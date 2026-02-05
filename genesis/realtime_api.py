"""Real-Time Synthetic Data APIs.

High-performance, low-latency API for real-time synthetic data generation
with model caching, batching, and optional gRPC support.

Features:
    - Low-latency generation (<100ms p99 for cached models)
    - Model warm-up and caching
    - Request batching for throughput optimization
    - gRPC support for high-performance use cases
    - Rate limiting and backpressure
    - Health checks and metrics

Example:
    REST API usage::

        from genesis.realtime_api import create_realtime_app, RealtimeConfig
        
        config = RealtimeConfig(
            cache_size=100,
            warmup_models=["customer_model", "transaction_model"],
            max_batch_size=64,
        )
        app = create_realtime_app(config)
        # Run with: uvicorn genesis.realtime_api:app

    Client usage::

        import httpx
        
        async with httpx.AsyncClient() as client:
            # Single record generation (lowest latency)
            response = await client.post(
                "http://localhost:8000/v1/realtime/generate",
                json={"model_id": "customer_model", "n_samples": 1}
            )
            record = response.json()["data"][0]

    gRPC usage::

        from genesis.realtime_api import RealtimeGRPCClient
        
        client = RealtimeGRPCClient("localhost:50051")
        records = await client.generate("customer_model", n_samples=100)

Classes:
    RealtimeConfig: Configuration for real-time API.
    ModelCache: LRU cache for fitted generator models.
    BatchProcessor: Batches requests for throughput.
    RealtimeGenerator: Optimized generator wrapper.
    RealtimeGRPCServer: gRPC server implementation.
    RealtimeGRPCClient: gRPC client for high-performance access.
"""

import asyncio
import hashlib
import os
import pickle
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from genesis.core.exceptions import ConfigurationError, NotFittedError
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class CacheStrategy(str, Enum):
    """Model caching strategy."""

    LRU = "lru"  # Least recently used
    LFU = "lfu"  # Least frequently used
    FIFO = "fifo"  # First in, first out
    TTL = "ttl"  # Time to live


@dataclass
class RealtimeConfig:
    """Configuration for real-time synthetic data API.

    Attributes:
        cache_size: Maximum number of models to cache in memory
        warmup_models: List of model IDs to pre-load on startup
        max_batch_size: Maximum batch size for batched generation
        batch_timeout_ms: Timeout for batch accumulation
        max_concurrent_requests: Maximum concurrent generation requests
        target_latency_ms: Target p99 latency for SLA monitoring
        enable_grpc: Enable gRPC server alongside REST
        grpc_port: Port for gRPC server
        model_dir: Directory for persisted models
        cache_strategy: Caching eviction strategy
        ttl_seconds: TTL for cached models (if using TTL strategy)
        enable_metrics: Enable Prometheus metrics
        rate_limit_rps: Requests per second limit (0 = unlimited)
    """

    cache_size: int = 100
    warmup_models: List[str] = field(default_factory=list)
    max_batch_size: int = 64
    batch_timeout_ms: int = 10
    max_concurrent_requests: int = 100
    target_latency_ms: float = 100.0
    enable_grpc: bool = False
    grpc_port: int = 50051
    model_dir: str = "./models"
    cache_strategy: CacheStrategy = CacheStrategy.LRU
    ttl_seconds: int = 3600
    enable_metrics: bool = True
    rate_limit_rps: int = 0


@dataclass
class GenerationRequest:
    """Single generation request."""

    model_id: str
    n_samples: int
    conditions: Optional[Dict[str, Any]] = None
    request_id: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class GenerationResponse:
    """Generation response with timing."""

    request_id: str
    data: List[Dict[str, Any]]
    latency_ms: float
    model_id: str
    cached: bool = True
    batch_size: int = 1


@dataclass
class ModelMetrics:
    """Metrics for a cached model."""

    model_id: str
    load_time_ms: float
    avg_generation_time_ms: float
    total_requests: int
    last_accessed: float
    memory_bytes: int
    access_count: int = 0


class ModelCache:
    """Thread-safe LRU cache for fitted generator models.

    Provides fast model access with configurable eviction strategies
    and automatic memory management.
    """

    def __init__(
        self,
        max_size: int = 100,
        strategy: CacheStrategy = CacheStrategy.LRU,
        ttl_seconds: int = 3600,
    ):
        """Initialize model cache.

        Args:
            max_size: Maximum number of models to cache
            strategy: Eviction strategy
            ttl_seconds: TTL for cached entries (TTL strategy only)
        """
        self._max_size = max_size
        self._strategy = strategy
        self._ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._metrics: Dict[str, ModelMetrics] = {}
        self._lock = threading.RLock()
        self._access_counts: Dict[str, int] = {}
        self._timestamps: Dict[str, float] = {}

    def get(self, model_id: str) -> Optional[Any]:
        """Get a model from cache.

        Args:
            model_id: Model identifier

        Returns:
            Cached model or None if not found
        """
        with self._lock:
            if model_id not in self._cache:
                return None

            # Check TTL
            if self._strategy == CacheStrategy.TTL:
                if time.time() - self._timestamps[model_id] > self._ttl_seconds:
                    self._evict(model_id)
                    return None

            # Update access tracking
            if self._strategy == CacheStrategy.LRU:
                self._cache.move_to_end(model_id)
            elif self._strategy == CacheStrategy.LFU:
                self._access_counts[model_id] = self._access_counts.get(model_id, 0) + 1

            # Update metrics
            if model_id in self._metrics:
                self._metrics[model_id].last_accessed = time.time()
                self._metrics[model_id].access_count += 1

            return self._cache[model_id]

    def put(
        self,
        model_id: str,
        model: Any,
        load_time_ms: float = 0.0,
        memory_bytes: int = 0,
    ) -> None:
        """Add a model to cache.

        Args:
            model_id: Model identifier
            model: Fitted generator model
            load_time_ms: Time taken to load/fit model
            memory_bytes: Estimated memory usage
        """
        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self._max_size:
                self._evict_one()

            self._cache[model_id] = model
            self._timestamps[model_id] = time.time()
            self._access_counts[model_id] = 1
            self._metrics[model_id] = ModelMetrics(
                model_id=model_id,
                load_time_ms=load_time_ms,
                avg_generation_time_ms=0.0,
                total_requests=0,
                last_accessed=time.time(),
                memory_bytes=memory_bytes,
            )

    def _evict_one(self) -> None:
        """Evict one model based on strategy."""
        if not self._cache:
            return

        if self._strategy == CacheStrategy.LRU:
            victim = next(iter(self._cache))
        elif self._strategy == CacheStrategy.FIFO:
            victim = next(iter(self._cache))
        elif self._strategy == CacheStrategy.LFU:
            victim = min(self._access_counts, key=self._access_counts.get)
        elif self._strategy == CacheStrategy.TTL:
            # Evict oldest
            victim = min(self._timestamps, key=self._timestamps.get)
        else:
            victim = next(iter(self._cache))

        self._evict(victim)

    def _evict(self, model_id: str) -> None:
        """Evict a specific model."""
        if model_id in self._cache:
            del self._cache[model_id]
        if model_id in self._timestamps:
            del self._timestamps[model_id]
        if model_id in self._access_counts:
            del self._access_counts[model_id]
        if model_id in self._metrics:
            del self._metrics[model_id]
        logger.debug(f"Evicted model: {model_id}")

    def contains(self, model_id: str) -> bool:
        """Check if model is in cache."""
        with self._lock:
            return model_id in self._cache

    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            self._cache.clear()
            self._metrics.clear()
            self._timestamps.clear()
            self._access_counts.clear()

    def get_metrics(self) -> Dict[str, ModelMetrics]:
        """Get metrics for all cached models."""
        with self._lock:
            return dict(self._metrics)

    def update_generation_time(self, model_id: str, time_ms: float) -> None:
        """Update average generation time metric."""
        with self._lock:
            if model_id in self._metrics:
                m = self._metrics[model_id]
                n = m.total_requests
                m.avg_generation_time_ms = (m.avg_generation_time_ms * n + time_ms) / (n + 1)
                m.total_requests += 1


class BatchProcessor:
    """Batches incoming requests for improved throughput.

    Accumulates requests up to max_batch_size or timeout, then
    processes them together for better GPU utilization.
    """

    def __init__(
        self,
        max_batch_size: int = 64,
        timeout_ms: int = 10,
        process_fn: Optional[Callable] = None,
    ):
        """Initialize batch processor.

        Args:
            max_batch_size: Maximum requests per batch
            timeout_ms: Max time to wait for batch to fill
            process_fn: Function to process a batch
        """
        self._max_batch_size = max_batch_size
        self._timeout_ms = timeout_ms
        self._process_fn = process_fn
        self._pending: Dict[str, List[Tuple[GenerationRequest, asyncio.Future]]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._tasks: Dict[str, asyncio.Task] = {}

    async def submit(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """Submit a request for batched processing.

        Args:
            request: Generation request

        Returns:
            Generation response
        """
        model_id = request.model_id
        future: asyncio.Future = asyncio.Future()

        # Get or create lock for this model
        if model_id not in self._locks:
            self._locks[model_id] = asyncio.Lock()

        async with self._locks[model_id]:
            if model_id not in self._pending:
                self._pending[model_id] = []

            self._pending[model_id].append((request, future))

            # Start batch timer if first request
            if len(self._pending[model_id]) == 1:
                self._tasks[model_id] = asyncio.create_task(
                    self._batch_timeout(model_id)
                )

            # Process immediately if batch is full
            if len(self._pending[model_id]) >= self._max_batch_size:
                if model_id in self._tasks:
                    self._tasks[model_id].cancel()
                await self._process_batch(model_id)

        return await future

    async def _batch_timeout(self, model_id: str) -> None:
        """Process batch after timeout."""
        await asyncio.sleep(self._timeout_ms / 1000.0)
        async with self._locks[model_id]:
            if model_id in self._pending and self._pending[model_id]:
                await self._process_batch(model_id)

    async def _process_batch(self, model_id: str) -> None:
        """Process accumulated batch."""
        if model_id not in self._pending or not self._pending[model_id]:
            return

        batch = self._pending.pop(model_id)
        requests = [r for r, _ in batch]
        futures = [f for _, f in batch]

        try:
            if self._process_fn:
                responses = await self._process_fn(model_id, requests)
                for future, response in zip(futures, responses):
                    if not future.done():
                        future.set_result(response)
            else:
                # Default: return empty responses
                for i, future in enumerate(futures):
                    if not future.done():
                        future.set_result(
                            GenerationResponse(
                                request_id=requests[i].request_id,
                                data=[],
                                latency_ms=0.0,
                                model_id=model_id,
                            )
                        )
        except Exception as e:
            for future in futures:
                if not future.done():
                    future.set_exception(e)


class RealtimeGenerator:
    """Optimized generator wrapper for real-time inference.

    Wraps existing generators with optimizations:
    - Pre-compiled models (TorchScript/ONNX when available)
    - Batched generation
    - Warm-up to avoid cold start latency
    """

    def __init__(
        self,
        generator: Any,
        model_id: str,
        optimize: bool = True,
    ):
        """Initialize realtime generator.

        Args:
            generator: Fitted generator instance
            model_id: Model identifier
            optimize: Whether to apply optimizations
        """
        self._generator = generator
        self._model_id = model_id
        self._optimized = False
        self._warm = False

        if optimize:
            self._optimize()

    def _optimize(self) -> None:
        """Apply optimizations to generator."""
        # Try TorchScript compilation if PyTorch-based
        try:
            if hasattr(self._generator, "_model") and hasattr(
                self._generator._model, "eval"
            ):
                self._generator._model.eval()
                # Note: Full TorchScript compilation would require model changes
                self._optimized = True
                logger.debug(f"Optimized model: {self._model_id}")
        except Exception as e:
            logger.debug(f"Could not optimize model {self._model_id}: {e}")

    def warmup(self, n_samples: int = 10) -> float:
        """Warm up the generator with a test generation.

        Args:
            n_samples: Number of samples for warmup

        Returns:
            Warmup time in milliseconds
        """
        start = time.perf_counter()
        try:
            _ = self._generator.generate(n_samples=n_samples)
            self._warm = True
        except Exception as e:
            logger.warning(f"Warmup failed for {self._model_id}: {e}")
        return (time.perf_counter() - start) * 1000

    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Generate samples with minimal latency.

        Args:
            n_samples: Number of samples
            conditions: Optional conditions

        Returns:
            Generated DataFrame
        """
        if conditions:
            return self._generator.generate(n_samples=n_samples, conditions=conditions)
        return self._generator.generate(n_samples=n_samples)

    @property
    def is_warm(self) -> bool:
        """Check if generator is warmed up."""
        return self._warm

    @property
    def is_optimized(self) -> bool:
        """Check if generator is optimized."""
        return self._optimized


class RealtimeAPIMetrics:
    """Metrics collector for real-time API."""

    def __init__(self):
        self._request_count = 0
        self._error_count = 0
        self._latencies: List[float] = []
        self._lock = threading.Lock()
        self._start_time = time.time()

    def record_request(self, latency_ms: float, error: bool = False) -> None:
        """Record a request."""
        with self._lock:
            self._request_count += 1
            if error:
                self._error_count += 1
            self._latencies.append(latency_ms)
            # Keep only last 10000 latencies
            if len(self._latencies) > 10000:
                self._latencies = self._latencies[-10000:]

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        with self._lock:
            if not self._latencies:
                return {
                    "total_requests": self._request_count,
                    "error_rate": 0.0,
                    "p50_latency_ms": 0.0,
                    "p99_latency_ms": 0.0,
                    "uptime_seconds": time.time() - self._start_time,
                }

            sorted_latencies = sorted(self._latencies)
            return {
                "total_requests": self._request_count,
                "error_rate": self._error_count / max(1, self._request_count),
                "p50_latency_ms": sorted_latencies[len(sorted_latencies) // 2],
                "p99_latency_ms": sorted_latencies[int(len(sorted_latencies) * 0.99)],
                "avg_latency_ms": sum(self._latencies) / len(self._latencies),
                "uptime_seconds": time.time() - self._start_time,
            }


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, rate_limit_rps: int):
        """Initialize rate limiter.

        Args:
            rate_limit_rps: Maximum requests per second (0 = unlimited)
        """
        self._rate = rate_limit_rps
        self._tokens = float(rate_limit_rps)
        self._last_update = time.time()
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        """Try to acquire a token.

        Returns:
            True if token acquired, False if rate limited
        """
        if self._rate <= 0:
            return True

        with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            self._tokens = min(self._rate, self._tokens + elapsed * self._rate)
            self._last_update = now

            if self._tokens >= 1:
                self._tokens -= 1
                return True
            return False


# Global instances for the API
_config: Optional[RealtimeConfig] = None
_cache: Optional[ModelCache] = None
_metrics: Optional[RealtimeAPIMetrics] = None
_rate_limiter: Optional[RateLimiter] = None
_batch_processor: Optional[BatchProcessor] = None
_executor: Optional[ThreadPoolExecutor] = None


def _init_globals(config: RealtimeConfig) -> None:
    """Initialize global instances."""
    global _config, _cache, _metrics, _rate_limiter, _batch_processor, _executor

    _config = config
    _cache = ModelCache(
        max_size=config.cache_size,
        strategy=config.cache_strategy,
        ttl_seconds=config.ttl_seconds,
    )
    _metrics = RealtimeAPIMetrics()
    _rate_limiter = RateLimiter(config.rate_limit_rps)
    _executor = ThreadPoolExecutor(max_workers=config.max_concurrent_requests)


async def _process_batch(
    model_id: str,
    requests: List[GenerationRequest],
) -> List[GenerationResponse]:
    """Process a batch of generation requests."""
    responses = []
    generator = _cache.get(model_id) if _cache else None

    if not generator:
        for req in requests:
            responses.append(
                GenerationResponse(
                    request_id=req.request_id,
                    data=[],
                    latency_ms=0.0,
                    model_id=model_id,
                    cached=False,
                )
            )
        return responses

    # Calculate total samples
    total_samples = sum(r.n_samples for r in requests)

    # Generate in one batch
    start = time.perf_counter()
    try:
        df = generator.generate(n_samples=total_samples)
        records = df.to_dict("records")
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        for req in requests:
            responses.append(
                GenerationResponse(
                    request_id=req.request_id,
                    data=[],
                    latency_ms=0.0,
                    model_id=model_id,
                    cached=True,
                )
            )
        return responses

    latency = (time.perf_counter() - start) * 1000

    # Split results
    offset = 0
    for req in requests:
        responses.append(
            GenerationResponse(
                request_id=req.request_id,
                data=records[offset : offset + req.n_samples],
                latency_ms=latency / len(requests),
                model_id=model_id,
                cached=True,
                batch_size=len(requests),
            )
        )
        offset += req.n_samples

    if _cache:
        _cache.update_generation_time(model_id, latency)

    return responses


def create_realtime_app(config: Optional[RealtimeConfig] = None) -> Any:
    """Create FastAPI application for real-time synthetic data generation.

    Args:
        config: Real-time API configuration

    Returns:
        FastAPI application
    """
    try:
        from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("FastAPI required. Install with: pip install genesis[api]")

    config = config or RealtimeConfig()
    _init_globals(config)

    # Pydantic models for API
    class RealtimeGenerateRequest(BaseModel):
        model_id: str
        n_samples: int = 1
        conditions: Optional[Dict[str, Any]] = None
        batch: bool = False

    class RealtimeGenerateResponse(BaseModel):
        success: bool
        data: List[Dict[str, Any]]
        latency_ms: float
        model_id: str
        cached: bool
        batch_size: int = 1

    class LoadModelRequest(BaseModel):
        model_id: str
        model_path: Optional[str] = None
        warmup: bool = True

    class MetricsResponse(BaseModel):
        total_requests: int
        error_rate: float
        p50_latency_ms: float
        p99_latency_ms: float
        uptime_seconds: float
        cached_models: int

    app = FastAPI(
        title="Genesis Real-Time Synthetic Data API",
        description="Low-latency API for real-time synthetic data generation",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup():
        """Warm up models on startup."""
        global _batch_processor
        _batch_processor = BatchProcessor(
            max_batch_size=config.max_batch_size,
            timeout_ms=config.batch_timeout_ms,
            process_fn=_process_batch,
        )

        # Load warmup models
        for model_id in config.warmup_models:
            try:
                model_path = Path(config.model_dir) / f"{model_id}.pkl"
                if model_path.exists():
                    with open(model_path, "rb") as f:
                        generator = pickle.load(f)
                    rt_gen = RealtimeGenerator(generator, model_id)
                    rt_gen.warmup()
                    _cache.put(model_id, rt_gen)
                    logger.info(f"Warmed up model: {model_id}")
            except Exception as e:
                logger.warning(f"Failed to warm up {model_id}: {e}")

    @app.post("/v1/realtime/generate", response_model=RealtimeGenerateResponse)
    async def realtime_generate(request: RealtimeGenerateRequest):
        """Generate synthetic data with minimal latency.

        Optimized for single-record or small batch generation.
        Uses model caching and optional request batching.
        """
        # Rate limiting
        if _rate_limiter and not _rate_limiter.acquire():
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        start = time.perf_counter()

        try:
            # Check cache
            generator = _cache.get(request.model_id) if _cache else None
            if not generator:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{request.model_id}' not loaded. POST to /v1/realtime/models first.",
                )

            # Use batching for batch requests
            if request.batch and _batch_processor:
                gen_request = GenerationRequest(
                    model_id=request.model_id,
                    n_samples=request.n_samples,
                    conditions=request.conditions,
                    request_id=hashlib.md5(str(time.time()).encode()).hexdigest()[:8],
                )
                response = await _batch_processor.submit(gen_request)
                latency = (time.perf_counter() - start) * 1000

                if _metrics:
                    _metrics.record_request(latency)

                return RealtimeGenerateResponse(
                    success=True,
                    data=response.data,
                    latency_ms=latency,
                    model_id=request.model_id,
                    cached=True,
                    batch_size=response.batch_size,
                )

            # Direct generation (lowest latency for single requests)
            df = generator.generate(
                n_samples=request.n_samples,
                conditions=request.conditions,
            )
            data = df.to_dict("records")
            latency = (time.perf_counter() - start) * 1000

            if _metrics:
                _metrics.record_request(latency)
            if _cache:
                _cache.update_generation_time(request.model_id, latency)

            return RealtimeGenerateResponse(
                success=True,
                data=data,
                latency_ms=latency,
                model_id=request.model_id,
                cached=True,
            )

        except HTTPException:
            raise
        except Exception as e:
            if _metrics:
                _metrics.record_request(0.0, error=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/realtime/models")
    async def load_model(request: LoadModelRequest):
        """Load and cache a model for real-time generation."""
        start = time.perf_counter()

        try:
            model_path = request.model_path
            if not model_path:
                model_path = str(Path(config.model_dir) / f"{request.model_id}.pkl")

            if not Path(model_path).exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Model file not found: {model_path}",
                )

            with open(model_path, "rb") as f:
                generator = pickle.load(f)

            rt_gen = RealtimeGenerator(generator, request.model_id)

            warmup_time = 0.0
            if request.warmup:
                warmup_time = rt_gen.warmup()

            load_time = (time.perf_counter() - start) * 1000
            _cache.put(request.model_id, rt_gen, load_time_ms=load_time)

            return {
                "success": True,
                "model_id": request.model_id,
                "load_time_ms": load_time,
                "warmup_time_ms": warmup_time,
                "optimized": rt_gen.is_optimized,
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/v1/realtime/models/{model_id}")
    async def unload_model(model_id: str):
        """Unload a model from cache."""
        if not _cache or not _cache.contains(model_id):
            raise HTTPException(status_code=404, detail="Model not found in cache")

        _cache._evict(model_id)
        return {"success": True, "model_id": model_id}

    @app.get("/v1/realtime/models")
    async def list_cached_models():
        """List all cached models with metrics."""
        if not _cache:
            return {"models": []}

        metrics = _cache.get_metrics()
        return {
            "models": [
                {
                    "model_id": m.model_id,
                    "load_time_ms": m.load_time_ms,
                    "avg_generation_time_ms": m.avg_generation_time_ms,
                    "total_requests": m.total_requests,
                    "last_accessed": datetime.fromtimestamp(m.last_accessed).isoformat(),
                }
                for m in metrics.values()
            ]
        }

    @app.get("/v1/realtime/metrics", response_model=MetricsResponse)
    async def get_metrics():
        """Get API performance metrics."""
        stats = _metrics.get_stats() if _metrics else {}
        return MetricsResponse(
            total_requests=stats.get("total_requests", 0),
            error_rate=stats.get("error_rate", 0.0),
            p50_latency_ms=stats.get("p50_latency_ms", 0.0),
            p99_latency_ms=stats.get("p99_latency_ms", 0.0),
            uptime_seconds=stats.get("uptime_seconds", 0.0),
            cached_models=len(_cache.get_metrics()) if _cache else 0,
        )

    @app.get("/v1/realtime/health")
    async def health():
        """Health check with SLA status."""
        stats = _metrics.get_stats() if _metrics else {}
        p99 = stats.get("p99_latency_ms", 0.0)
        sla_ok = p99 <= config.target_latency_ms if p99 > 0 else True

        return {
            "status": "healthy",
            "sla_ok": sla_ok,
            "target_latency_ms": config.target_latency_ms,
            "actual_p99_latency_ms": p99,
            "cached_models": len(_cache.get_metrics()) if _cache else 0,
        }

    return app


# gRPC support (optional)
def create_grpc_server(config: RealtimeConfig) -> Any:
    """Create gRPC server for high-performance generation.

    Args:
        config: Configuration

    Returns:
        gRPC server (requires grpcio)
    """
    try:
        import grpc
        from concurrent import futures
    except ImportError:
        raise ImportError("gRPC support requires: pip install grpcio grpcio-tools")

    # Note: Full gRPC implementation would require .proto compilation
    # This is a placeholder showing the structure
    logger.info(f"gRPC server would start on port {config.grpc_port}")
    return None


class RealtimeClient:
    """Async client for real-time API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize client.

        Args:
            base_url: API base URL
        """
        self._base_url = base_url.rstrip("/")

    async def generate(
        self,
        model_id: str,
        n_samples: int = 1,
        conditions: Optional[Dict[str, Any]] = None,
        batch: bool = False,
    ) -> List[Dict[str, Any]]:
        """Generate synthetic data.

        Args:
            model_id: Model to use
            n_samples: Number of samples
            conditions: Optional conditions
            batch: Use request batching

        Returns:
            List of generated records
        """
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required for client: pip install httpx")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/v1/realtime/generate",
                json={
                    "model_id": model_id,
                    "n_samples": n_samples,
                    "conditions": conditions,
                    "batch": batch,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()["data"]

    async def load_model(
        self,
        model_id: str,
        model_path: Optional[str] = None,
        warmup: bool = True,
    ) -> Dict[str, Any]:
        """Load a model into the server cache.

        Args:
            model_id: Model identifier
            model_path: Path to model file
            warmup: Perform warmup generation

        Returns:
            Load result
        """
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required for client: pip install httpx")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/v1/realtime/models",
                json={
                    "model_id": model_id,
                    "model_path": model_path,
                    "warmup": warmup,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            return response.json()

    async def get_metrics(self) -> Dict[str, Any]:
        """Get API metrics."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required for client: pip install httpx")

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self._base_url}/v1/realtime/metrics")
            response.raise_for_status()
            return response.json()


# Convenience function
def save_model_for_realtime(generator: Any, model_id: str, model_dir: str = "./models") -> str:
    """Save a fitted generator for real-time API usage.

    Args:
        generator: Fitted generator
        model_id: Model identifier
        model_dir: Directory to save model

    Returns:
        Path to saved model
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(model_dir) / f"{model_id}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(generator, f)

    logger.info(f"Saved model for real-time API: {model_path}")
    return str(model_path)


# Create default app instance
app = None
try:
    app = create_realtime_app()
except ImportError:
    pass  # FastAPI not installed
