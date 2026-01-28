"""Genesis REST API server.

This module provides a FastAPI-based REST API for synthetic data generation.

Example:
    # Start the server
    uvicorn genesis.api.server:app --host 0.0.0.0 --port 8000

    # Or programmatically
    from genesis.api import create_app
    app = create_app()
"""

import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from genesis.api.schemas import (
    ClarificationRequest,
    ErrorResponse,
    EvaluateRequest,
    EvaluateResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    JobStatus,
    JobStatusResponse,
    ListModelsResponse,
    ModelInfo,
    NaturalLanguageRequest,
    NaturalLanguageResponse,
    PrivacyMetrics,
    StatisticalMetrics,
)
from genesis.version import __version__

# Lazy import FastAPI to avoid mandatory dependency
try:
    from fastapi import BackgroundTasks, FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


# Global state
_startup_time: Optional[float] = None
_models: Dict[str, Any] = {}
_jobs: Dict[str, Dict[str, Any]] = {}
_nl_sessions: Dict[str, Any] = {}  # Natural language conversation sessions


def _check_fastapi() -> None:
    """Check if FastAPI is available."""
    if not HAS_FASTAPI:
        raise ImportError("FastAPI is not installed. Install with: pip install genesis[api]")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _startup_time
    _startup_time = time.time()
    yield
    # Cleanup
    _models.clear()
    _jobs.clear()


def create_app(allowed_origins: Optional[List[str]] = None) -> "FastAPI":
    """Create and configure the FastAPI application.

    Args:
        allowed_origins: List of allowed CORS origins. If None, reads from
            GENESIS_CORS_ORIGINS env var (comma-separated) or defaults to ["*"].

    Returns:
        Configured FastAPI app
    """
    _check_fastapi()

    # Determine CORS origins
    if allowed_origins is None:
        env_origins = os.environ.get("GENESIS_CORS_ORIGINS", "")
        if env_origins:
            origins = [o.strip() for o in env_origins.split(",") if o.strip()]
        else:
            origins = ["*"]
    else:
        origins = allowed_origins

    app = FastAPI(
        title="Genesis Synthetic Data API",
        description="REST API for generating privacy-safe synthetic data",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Add CORS middleware
    # Note: allow_credentials=True with allow_origins=["*"] is insecure.
    # In production, set GENESIS_CORS_ORIGINS to specific domains.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    _register_routes(app)

    return app


def _register_routes(app: "FastAPI") -> None:
    """Register API routes."""

    @app.get("/", response_model=Dict[str, str])
    async def root():
        """API root endpoint."""
        return {
            "name": "Genesis Synthetic Data API",
            "version": __version__,
            "docs": "/docs",
        }

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        uptime = time.time() - _startup_time if _startup_time else 0
        return HealthResponse(
            status="healthy",
            version=__version__,
            uptime_seconds=uptime,
            models_loaded=len(_models),
        )

    @app.post("/v1/generate", response_model=GenerateResponse)
    async def generate(
        request: GenerateRequest,
        background_tasks: BackgroundTasks,
    ):
        """Generate synthetic data.

        This endpoint accepts training data and generates synthetic samples.
        For large datasets or complex models, use async_mode=True.
        """
        start_time = time.time()

        try:
            # Validate request
            if request.data is None and request.model_id is None:
                raise HTTPException(
                    status_code=400, detail="Either 'data' or 'model_id' must be provided"
                )

            # Handle async mode
            if request.async_mode:
                job_id = str(uuid.uuid4())
                _jobs[job_id] = {
                    "status": JobStatus.PENDING,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                }
                background_tasks.add_task(_run_generation_job, job_id, request)

                return GenerateResponse(
                    success=True,
                    data=None,
                    n_samples=0,
                    job_id=job_id,
                    execution_time_ms=0,
                    message="Job submitted. Check status at /v1/jobs/{job_id}",
                )

            # Synchronous generation
            result = await _generate_sync(request)

            execution_time = (time.time() - start_time) * 1000

            return GenerateResponse(
                success=True,
                data=result["data"],
                n_samples=len(result["data"]),
                model_id=result.get("model_id"),
                execution_time_ms=execution_time,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/v1/evaluate", response_model=EvaluateResponse)
    async def evaluate(request: EvaluateRequest):
        """Evaluate synthetic data quality.

        Compare synthetic data against real data and compute quality metrics.
        """
        start_time = time.time()

        try:
            from genesis.evaluation.evaluator import QualityEvaluator

            real_df = pd.DataFrame(request.real_data)
            synthetic_df = pd.DataFrame(request.synthetic_data)

            evaluator = QualityEvaluator(
                real_data=real_df,
                synthetic_data=synthetic_df,
            )
            report = evaluator.evaluate()
            metrics = report.to_dict()

            execution_time = (time.time() - start_time) * 1000

            return EvaluateResponse(
                success=True,
                overall_score=metrics.get("overall_score", 0.0),
                statistical=StatisticalMetrics(
                    ks_test_avg=metrics.get("statistical", {}).get("ks_test_avg"),
                    correlation_diff=metrics.get("statistical", {}).get("correlation_diff"),
                ),
                privacy=PrivacyMetrics(
                    dcr_score=metrics.get("privacy", {}).get("dcr_score"),
                ),
                execution_time_ms=execution_time,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
    async def get_job_status(job_id: str):
        """Get status of an async generation job."""
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        job = _jobs[job_id]
        return JobStatusResponse(
            job_id=job_id,
            status=job["status"],
            progress=job.get("progress"),
            result=job.get("result"),
            error=job.get("error"),
            created_at=job["created_at"],
            updated_at=job["updated_at"],
        )

    @app.get("/v1/models", response_model=ListModelsResponse)
    async def list_models():
        """List all trained models."""
        models = [
            ModelInfo(
                model_id=model_id,
                method=model.get("method", "unknown"),
                created_at=model.get("created_at", datetime.utcnow()),
                n_columns=model.get("n_columns", 0),
                n_training_samples=model.get("n_training_samples", 0),
                discrete_columns=model.get("discrete_columns", []),
            )
            for model_id, model in _models.items()
        ]
        return ListModelsResponse(models=models, total=len(models))

    @app.delete("/v1/models/{model_id}")
    async def delete_model(model_id: str):
        """Delete a trained model."""
        if model_id not in _models:
            raise HTTPException(status_code=404, detail="Model not found")

        del _models[model_id]
        return {"success": True, "message": f"Model {model_id} deleted"}

    @app.post("/v1/generate/natural-language", response_model=NaturalLanguageResponse)
    async def generate_natural_language(request: NaturalLanguageRequest):
        """Generate synthetic data using natural language.

        Describe your data needs in plain English and the AI will
        configure and run the appropriate generators.

        Example:
            POST /v1/generate/natural-language
            {
                "prompt": "Generate 10000 customer records with names, ages 18-65,
                          US addresses, and income normally distributed around $50k",
                "n_samples": 10000
            }
        """
        import os

        start_time = time.time()

        try:
            from genesis.agents import SyntheticDataAgent

            # Get API key
            api_key = request.api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise HTTPException(
                    status_code=400,
                    detail="API key required. Provide 'api_key' or set OPENAI_API_KEY",
                )

            # Load reference data if provided
            base_data = None
            if request.reference_data:
                base_data = pd.DataFrame(request.reference_data)

            # Create agent
            agent = SyntheticDataAgent(
                api_key=api_key,
                model=request.model,
                base_data=base_data,
            )

            # Process request
            response = agent.request(request.prompt, n_samples=request.n_samples)

            # Create session for potential follow-up
            session_id = str(uuid.uuid4())
            _nl_sessions[session_id] = {
                "agent": agent,
                "created_at": datetime.utcnow(),
            }

            if response.needs_clarification:
                return NaturalLanguageResponse(
                    success=True,
                    needs_clarification=True,
                    clarification_question=response.clarification_question,
                    session_id=session_id,
                    interpretation=response.explanation,
                )

            # Generate data
            data = response.generate()
            execution_time = (time.time() - start_time) * 1000

            # Clean up session
            del _nl_sessions[session_id]

            return NaturalLanguageResponse(
                success=True,
                needs_clarification=False,
                session_id=None,
                interpretation=response.explanation,
                config_summary={
                    "method": response.config.generator_method if response.config else None,
                    "n_columns": len(response.config.columns) if response.config else None,
                },
                data=data.to_dict(orient="records"),
                n_samples=len(data),
                execution_time_ms=execution_time,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/v1/generate/natural-language/clarify", response_model=NaturalLanguageResponse)
    async def clarify_natural_language(request: ClarificationRequest):
        """Provide clarification for a natural language generation request.

        Use this endpoint to answer follow-up questions from the AI.
        """
        start_time = time.time()

        if request.session_id not in _nl_sessions:
            raise HTTPException(
                status_code=404, detail="Session not found. Sessions expire after 10 minutes."
            )

        try:
            session = _nl_sessions[request.session_id]
            agent = session["agent"]

            # Provide answer
            response = agent.respond(request.answer)

            if response.needs_clarification:
                return NaturalLanguageResponse(
                    success=True,
                    needs_clarification=True,
                    clarification_question=response.clarification_question,
                    session_id=request.session_id,
                    interpretation=response.explanation,
                )

            # Generate data
            data = response.generate()
            execution_time = (time.time() - start_time) * 1000

            # Clean up session
            del _nl_sessions[request.session_id]

            return NaturalLanguageResponse(
                success=True,
                needs_clarification=False,
                session_id=None,
                interpretation=response.explanation,
                config_summary={
                    "method": response.config.generator_method if response.config else None,
                    "n_columns": len(response.config.columns) if response.config else None,
                },
                data=data.to_dict(orient="records"),
                n_samples=len(data),
                execution_time_ms=execution_time,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler."""
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=str(exc),
                error_code="INTERNAL_ERROR",
            ).model_dump(),
        )


async def _generate_sync(request: GenerateRequest) -> Dict[str, Any]:
    """Run synchronous generation."""
    from genesis import GeneratorConfig, PrivacyConfig, SyntheticGenerator

    # Get or create generator
    if request.model_id and request.model_id in _models:
        generator = _models[request.model_id]["generator"]
    else:
        # Create new generator
        gen_config = None
        if request.generator_config:
            gen_config = GeneratorConfig(
                method=request.generator_config.method.value,
                epochs=request.generator_config.epochs,
                batch_size=request.generator_config.batch_size,
            )

        priv_config = None
        if request.privacy_config:
            priv_config = PrivacyConfig(
                preset=request.privacy_config.level.value,
            )

        generator = SyntheticGenerator(
            method=gen_config.method if gen_config else "auto",
            config=gen_config,
            privacy=priv_config,
        )

        # Fit the generator
        if request.data:
            df = pd.DataFrame(request.data)
            generator.fit(df, discrete_columns=request.discrete_columns)

            # Store model
            model_id = str(uuid.uuid4())
            _models[model_id] = {
                "generator": generator,
                "method": gen_config.method if gen_config else "auto",
                "created_at": datetime.utcnow(),
                "n_columns": len(df.columns),
                "n_training_samples": len(df),
                "discrete_columns": request.discrete_columns or [],
            }

    # Generate
    conditions = None
    if request.conditions:
        conditions = {c.column: c.value for c in request.conditions}

    synthetic_df = generator.generate(request.n_samples, conditions=conditions)

    return {
        "data": synthetic_df.to_dict(orient="records"),
        "model_id": model_id if "model_id" in dir() else None,
    }


async def _run_generation_job(job_id: str, request: GenerateRequest) -> None:
    """Run generation job in background."""
    try:
        _jobs[job_id]["status"] = JobStatus.RUNNING
        _jobs[job_id]["updated_at"] = datetime.utcnow()

        result = await _generate_sync(request)

        _jobs[job_id]["status"] = JobStatus.COMPLETED
        _jobs[job_id]["result"] = GenerateResponse(
            success=True,
            data=result["data"],
            n_samples=len(result["data"]),
            model_id=result.get("model_id"),
            execution_time_ms=0,
        )
    except Exception as e:
        _jobs[job_id]["status"] = JobStatus.FAILED
        _jobs[job_id]["error"] = str(e)
    finally:
        _jobs[job_id]["updated_at"] = datetime.utcnow()


# Create default app instance
if HAS_FASTAPI:
    app = create_app()
else:
    app = None
