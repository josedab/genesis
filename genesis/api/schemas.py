"""Pydantic schemas for the Genesis API."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class GeneratorMethod(str, Enum):
    """Available generator methods."""

    AUTO = "auto"
    CTGAN = "ctgan"
    TVAE = "tvae"
    GAUSSIAN_COPULA = "gaussian_copula"


class PrivacyLevel(str, Enum):
    """Privacy level presets."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class JobStatus(str, Enum):
    """Job status states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ConditionOperator(str, Enum):
    """Condition operators."""

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GE = "ge"
    LT = "lt"
    LE = "le"
    IN = "in"
    BETWEEN = "between"


class Condition(BaseModel):
    """A generation condition."""

    column: str = Field(..., description="Column name")
    operator: ConditionOperator = Field(default=ConditionOperator.EQ)
    value: Any = Field(..., description="Value to compare against")


class ConstraintSpec(BaseModel):
    """A constraint specification."""

    type: str = Field(..., description="Constraint type: positive, range, unique, regex, one_of")
    column: str = Field(..., description="Column to apply constraint to")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Constraint parameters")


class GeneratorConfig(BaseModel):
    """Generator configuration."""

    method: GeneratorMethod = Field(default=GeneratorMethod.AUTO)
    epochs: int = Field(default=300, ge=1, le=10000)
    batch_size: int = Field(default=500, ge=1, le=10000)
    learning_rate: float = Field(default=0.0002, gt=0)
    embedding_dim: int = Field(default=128, ge=1)


class PrivacyConfig(BaseModel):
    """Privacy configuration."""

    level: PrivacyLevel = Field(default=PrivacyLevel.NONE)
    epsilon: Optional[float] = Field(default=None, gt=0)
    delta: Optional[float] = Field(default=None, gt=0, lt=1)
    k_anonymity: Optional[int] = Field(default=None, ge=2)


class GenerateRequest(BaseModel):
    """Request to generate synthetic data."""

    data: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Training data as list of records. Either 'data' or 'model_id' required.",
    )
    model_id: Optional[str] = Field(
        default=None, description="ID of pre-trained model. Either 'data' or 'model_id' required."
    )
    n_samples: int = Field(
        default=1000, ge=1, le=10_000_000, description="Number of samples to generate"
    )
    discrete_columns: Optional[List[str]] = Field(
        default=None, description="List of categorical column names"
    )
    conditions: Optional[List[Condition]] = Field(default=None, description="Generation conditions")
    constraints: Optional[List[ConstraintSpec]] = Field(
        default=None, description="Constraints to enforce"
    )
    generator_config: Optional[GeneratorConfig] = Field(
        default=None, description="Generator configuration"
    )
    privacy_config: Optional[PrivacyConfig] = Field(
        default=None, description="Privacy configuration"
    )
    async_mode: bool = Field(default=False, description="Run generation asynchronously")

    @field_validator("data", "model_id")
    @classmethod
    def check_data_or_model(cls, v, info):
        """Validate that data or model_id is provided."""
        return v


class GenerateResponse(BaseModel):
    """Response from generate endpoint."""

    success: bool
    data: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Generated data as list of records"
    )
    n_samples: int = Field(description="Number of samples generated")
    model_id: Optional[str] = Field(default=None, description="ID of trained model (for reuse)")
    job_id: Optional[str] = Field(default=None, description="Job ID for async requests")
    execution_time_ms: float = Field(description="Execution time in milliseconds")
    message: Optional[str] = Field(default=None)


class EvaluateRequest(BaseModel):
    """Request to evaluate synthetic data quality."""

    real_data: List[Dict[str, Any]] = Field(..., description="Real data as list of records")
    synthetic_data: List[Dict[str, Any]] = Field(
        ..., description="Synthetic data as list of records"
    )
    metrics: Optional[List[str]] = Field(default=None, description="Specific metrics to compute")


class StatisticalMetrics(BaseModel):
    """Statistical quality metrics."""

    ks_test_avg: Optional[float] = None
    chi_squared_avg: Optional[float] = None
    correlation_diff: Optional[float] = None


class MLUtilityMetrics(BaseModel):
    """ML utility metrics."""

    tstr_score: Optional[float] = None
    trts_score: Optional[float] = None


class PrivacyMetrics(BaseModel):
    """Privacy metrics."""

    dcr_score: Optional[float] = None
    reid_risk: Optional[float] = None


class EvaluateResponse(BaseModel):
    """Response from evaluate endpoint."""

    success: bool
    overall_score: float = Field(ge=0, le=1)
    statistical: Optional[StatisticalMetrics] = None
    ml_utility: Optional[MLUtilityMetrics] = None
    privacy: Optional[PrivacyMetrics] = None
    execution_time_ms: float


class JobStatusResponse(BaseModel):
    """Response for job status check."""

    job_id: str
    status: JobStatus
    progress: Optional[float] = Field(default=None, ge=0, le=100)
    result: Optional[GenerateResponse] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str
    uptime_seconds: float
    models_loaded: int = 0


class ModelInfo(BaseModel):
    """Information about a trained model."""

    model_id: str
    method: str
    created_at: datetime
    n_columns: int
    n_training_samples: int
    discrete_columns: List[str]


class ListModelsResponse(BaseModel):
    """Response for listing models."""

    models: List[ModelInfo]
    total: int


class ErrorResponse(BaseModel):
    """Error response."""

    success: bool = False
    error: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class NaturalLanguageRequest(BaseModel):
    """Request for natural language generation."""

    prompt: str = Field(
        ...,
        description="Natural language description of desired data",
        min_length=10,
        max_length=2000,
    )
    reference_data: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Optional reference data to base generation on"
    )
    n_samples: Optional[int] = Field(
        default=None,
        ge=1,
        le=10_000_000,
        description="Override number of samples (LLM will estimate if not provided)",
    )
    model: str = Field(default="gpt-4o-mini", description="LLM model to use for interpretation")
    api_key: Optional[str] = Field(
        default=None, description="API key for LLM provider (or use server default)"
    )


class NaturalLanguageResponse(BaseModel):
    """Response from natural language generation."""

    success: bool
    needs_clarification: bool = Field(default=False, description="Whether clarification is needed")
    clarification_question: Optional[str] = Field(
        default=None, description="Question to ask user if clarification needed"
    )
    session_id: Optional[str] = Field(
        default=None, description="Session ID for multi-turn conversation"
    )
    interpretation: Optional[str] = Field(
        default=None, description="How the request was interpreted"
    )
    config_summary: Optional[Dict[str, Any]] = Field(
        default=None, description="Summary of generation configuration"
    )
    data: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Generated data (if not needing clarification)"
    )
    n_samples: Optional[int] = Field(default=None)
    execution_time_ms: Optional[float] = Field(default=None)


class ClarificationRequest(BaseModel):
    """Request to provide clarification in a conversation."""

    session_id: str = Field(..., description="Session ID from previous response")
    answer: str = Field(..., description="Answer to the clarification question")
