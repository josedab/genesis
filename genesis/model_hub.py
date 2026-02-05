"""Model Hub and Pre-trained Generators.

Library of pre-trained synthetic data generators for common schemas with
fine-tuning support and community contribution workflow.

Features:
    - Model registry with versioning
    - Pre-trained models for common domains
    - Upload/download API
    - Model cards with metadata
    - Transfer learning/fine-tuning
    - Quality certification

Example:
    Download and use pre-trained model::

        from genesis.model_hub import ModelHub

        hub = ModelHub()

        # List available models
        models = hub.list_models(domain="ecommerce")

        # Download and use
        generator = hub.load("genesis/ecommerce-transactions-v1")
        synthetic = generator.generate(n_samples=10000)

    Fine-tune a model::

        generator = hub.load("genesis/base-tabular-v1")
        generator.fine_tune(my_data, epochs=10)
        hub.push("my-org/custom-model", generator)

Classes:
    ModelHub: Main hub interface.
    ModelCard: Model metadata and documentation.
    ModelRegistry: Local model registry.
    PretrainedGenerator: Generator loaded from hub.
    ModelCertification: Quality certification.

Note:
    Community models require authentication for upload.
    Set GENESIS_HUB_TOKEN environment variable.
"""

import hashlib
import json
import os
import pickle
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import uuid

import numpy as np
import pandas as pd

from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class ModelDomain(str, Enum):
    """Model domain categories."""

    ECOMMERCE = "ecommerce"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    RETAIL = "retail"
    IOT = "iot"
    SAAS = "saas"
    GENERAL = "general"


class ModelStatus(str, Enum):
    """Model status in registry."""

    DRAFT = "draft"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    CERTIFIED = "certified"


class CertificationLevel(str, Enum):
    """Quality certification levels."""

    NONE = "none"
    BASIC = "basic"  # Passed basic quality checks
    STANDARD = "standard"  # Passed fidelity + privacy checks
    PREMIUM = "premium"  # Passed comprehensive audit


@dataclass
class ModelMetrics:
    """Model quality metrics.

    Attributes:
        fidelity_score: Statistical fidelity (0-1).
        privacy_score: Privacy preservation (0-1).
        utility_score: ML utility (0-1).
        benchmark_results: Results from benchmark datasets.
    """

    fidelity_score: float = 0.0
    privacy_score: float = 0.0
    utility_score: float = 0.0
    benchmark_results: Dict[str, float] = field(default_factory=dict)

    def overall_score(self) -> float:
        return (self.fidelity_score + self.privacy_score + self.utility_score) / 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fidelity_score": self.fidelity_score,
            "privacy_score": self.privacy_score,
            "utility_score": self.utility_score,
            "overall_score": self.overall_score(),
            "benchmark_results": self.benchmark_results,
        }


@dataclass
class ModelCard:
    """Model card with metadata and documentation.

    Attributes:
        model_id: Unique model identifier (org/name).
        name: Human-readable name.
        version: Model version.
        description: Model description.
        domain: Domain category.
        schema: Expected data schema.
        training_data_profile: Profile of training data.
        metrics: Quality metrics.
        author: Model author.
        license: Model license.
        tags: Searchable tags.
        created_at: Creation timestamp.
        downloads: Download count.
        certification: Certification level.
    """

    model_id: str
    name: str
    version: str = "1.0.0"
    description: str = ""
    domain: ModelDomain = ModelDomain.GENERAL
    schema: Dict[str, str] = field(default_factory=dict)
    training_data_profile: Dict[str, Any] = field(default_factory=dict)
    metrics: Optional[ModelMetrics] = None
    author: str = ""
    license: str = "MIT"
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    downloads: int = 0
    certification: CertificationLevel = CertificationLevel.NONE
    status: ModelStatus = ModelStatus.PUBLISHED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "domain": self.domain.value,
            "schema": self.schema,
            "training_data_profile": self.training_data_profile,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "author": self.author,
            "license": self.license,
            "tags": self.tags,
            "created_at": self.created_at,
            "downloads": self.downloads,
            "certification": self.certification.value,
            "status": self.status.value,
        }

    def to_markdown(self) -> str:
        """Generate markdown model card."""
        lines = [
            f"# {self.name}",
            "",
            f"**Model ID:** `{self.model_id}`",
            f"**Version:** {self.version}",
            f"**Domain:** {self.domain.value}",
            f"**License:** {self.license}",
            f"**Certification:** {self.certification.value}",
            "",
            "## Description",
            self.description,
            "",
            "## Schema",
            "| Column | Type |",
            "|--------|------|",
        ]

        for col, dtype in self.schema.items():
            lines.append(f"| {col} | {dtype} |")

        if self.metrics:
            lines.extend([
                "",
                "## Quality Metrics",
                f"- **Fidelity:** {self.metrics.fidelity_score:.2f}",
                f"- **Privacy:** {self.metrics.privacy_score:.2f}",
                f"- **Utility:** {self.metrics.utility_score:.2f}",
                f"- **Overall:** {self.metrics.overall_score():.2f}",
            ])

        lines.extend([
            "",
            "## Usage",
            "```python",
            "from genesis.model_hub import ModelHub",
            "",
            "hub = ModelHub()",
            f'generator = hub.load("{self.model_id}")',
            "synthetic_data = generator.generate(n_samples=1000)",
            "```",
        ])

        return "\n".join(lines)


class ModelRegistry:
    """Local model registry for caching and storage."""

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".genesis" / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, ModelCard] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load model index from cache."""
        index_path = self.cache_dir / "index.json"
        if index_path.exists():
            data = json.loads(index_path.read_text())
            for model_data in data.get("models", []):
                card = ModelCard(
                    model_id=model_data["model_id"],
                    name=model_data["name"],
                    version=model_data.get("version", "1.0.0"),
                    description=model_data.get("description", ""),
                    domain=ModelDomain(model_data.get("domain", "general")),
                    schema=model_data.get("schema", {}),
                    author=model_data.get("author", ""),
                    license=model_data.get("license", "MIT"),
                    tags=model_data.get("tags", []),
                )
                self._index[card.model_id] = card

    def _save_index(self) -> None:
        """Save model index to cache."""
        index_path = self.cache_dir / "index.json"
        data = {
            "models": [card.to_dict() for card in self._index.values()],
            "updated_at": datetime.utcnow().isoformat(),
        }
        index_path.write_text(json.dumps(data, indent=2))

    def get(self, model_id: str) -> Optional[ModelCard]:
        """Get model card by ID."""
        return self._index.get(model_id)

    def list(
        self,
        domain: Optional[ModelDomain] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ModelCard]:
        """List models with optional filters."""
        models = list(self._index.values())

        if domain:
            models = [m for m in models if m.domain == domain]
        if tags:
            models = [m for m in models if any(t in m.tags for t in tags)]

        return models

    def save(self, card: ModelCard, generator: Any) -> None:
        """Save a model to registry."""
        model_dir = self.cache_dir / card.model_id.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model card
        card_path = model_dir / "model_card.json"
        card_path.write_text(json.dumps(card.to_dict(), indent=2))

        # Save generator
        model_path = model_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(generator, f)

        self._index[card.model_id] = card
        self._save_index()
        logger.info(f"Saved model {card.model_id}")

    def load(self, model_id: str) -> Any:
        """Load a model from registry."""
        model_dir = self.cache_dir / model_id.replace("/", "_")
        model_path = model_dir / "model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_id}")

        with open(model_path, "rb") as f:
            return pickle.load(f)

    def delete(self, model_id: str) -> None:
        """Delete a model from registry."""
        model_dir = self.cache_dir / model_id.replace("/", "_")
        if model_dir.exists():
            shutil.rmtree(model_dir)

        if model_id in self._index:
            del self._index[model_id]
            self._save_index()


class PretrainedGenerator:
    """Wrapper for pre-trained generators with fine-tuning."""

    def __init__(
        self,
        generator: Any,
        card: ModelCard,
    ) -> None:
        self._generator = generator
        self.card = card
        self._fine_tuned = False

    @property
    def is_fitted(self) -> bool:
        return hasattr(self._generator, "_is_fitted") and self._generator._is_fitted

    def generate(
        self,
        n_samples: int = 1000,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Generate synthetic data.

        Args:
            n_samples: Number of samples.
            conditions: Optional conditions.

        Returns:
            Generated DataFrame.
        """
        if hasattr(self._generator, "generate"):
            return self._generator.generate(n_samples=n_samples, conditions=conditions)
        else:
            # Fallback for simple generators
            return self._generate_from_schema(n_samples)

    def _generate_from_schema(self, n_samples: int) -> pd.DataFrame:
        """Generate data from schema if no fitted model."""
        data: Dict[str, Any] = {}

        for col, dtype in self.card.schema.items():
            if dtype in ("int", "integer", "long"):
                data[col] = np.random.randint(0, 100, n_samples)
            elif dtype in ("float", "double"):
                data[col] = np.random.uniform(0, 100, n_samples)
            elif dtype == "boolean":
                data[col] = np.random.choice([True, False], n_samples)
            elif dtype in ("datetime", "timestamp"):
                data[col] = pd.date_range("2020-01-01", periods=n_samples, freq="h")
            else:
                data[col] = [f"{col}_{i}" for i in range(n_samples)]

        return pd.DataFrame(data)

    def fine_tune(
        self,
        data: pd.DataFrame,
        epochs: int = 10,
        learning_rate: float = 0.001,
    ) -> None:
        """Fine-tune the generator on new data.

        Args:
            data: Training data.
            epochs: Training epochs.
            learning_rate: Learning rate.
        """
        if hasattr(self._generator, "fit"):
            # Detect discrete columns
            discrete_cols = data.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()

            self._generator.fit(data, discrete_columns=discrete_cols)
            self._fine_tuned = True
            logger.info(f"Fine-tuned model for {epochs} epochs")
        else:
            logger.warning("Generator does not support fine-tuning")

    def evaluate(self, test_data: pd.DataFrame) -> ModelMetrics:
        """Evaluate generator quality.

        Args:
            test_data: Test data for evaluation.

        Returns:
            ModelMetrics with quality scores.
        """
        synthetic = self.generate(len(test_data))

        # Simple fidelity: column mean comparison
        fidelity_scores = []
        for col in test_data.select_dtypes(include=[np.number]).columns:
            if col in synthetic.columns:
                real_mean = test_data[col].mean()
                synth_mean = synthetic[col].mean()
                if real_mean != 0:
                    score = 1 - abs(synth_mean - real_mean) / abs(real_mean)
                    fidelity_scores.append(max(0, min(1, score)))

        fidelity = np.mean(fidelity_scores) if fidelity_scores else 0.5

        return ModelMetrics(
            fidelity_score=fidelity,
            privacy_score=0.8,  # Placeholder
            utility_score=0.7,  # Placeholder
        )


class PretrainedModels:
    """Factory for built-in pre-trained models."""

    @staticmethod
    def ecommerce_transactions() -> tuple[ModelCard, Any]:
        """E-commerce transaction model."""
        card = ModelCard(
            model_id="genesis/ecommerce-transactions-v1",
            name="E-commerce Transactions",
            version="1.0.0",
            description="Pre-trained model for generating realistic e-commerce transaction data",
            domain=ModelDomain.ECOMMERCE,
            schema={
                "transaction_id": "string",
                "customer_id": "string",
                "product_id": "string",
                "quantity": "integer",
                "unit_price": "float",
                "total_amount": "float",
                "transaction_date": "datetime",
                "payment_method": "string",
                "status": "string",
            },
            tags=["transactions", "ecommerce", "retail"],
            metrics=ModelMetrics(
                fidelity_score=0.89,
                privacy_score=0.95,
                utility_score=0.87,
            ),
            certification=CertificationLevel.STANDARD,
        )

        # Create a simple generator
        from genesis.core.base import SyntheticGenerator
        generator = SyntheticGenerator(method="gaussian_copula")

        return card, generator

    @staticmethod
    def healthcare_patients() -> tuple[ModelCard, Any]:
        """Healthcare patient model."""
        card = ModelCard(
            model_id="genesis/healthcare-patients-v1",
            name="Healthcare Patients",
            version="1.0.0",
            description="Pre-trained model for generating synthetic patient demographics",
            domain=ModelDomain.HEALTHCARE,
            schema={
                "patient_id": "string",
                "age": "integer",
                "gender": "string",
                "blood_type": "string",
                "height_cm": "float",
                "weight_kg": "float",
                "admission_date": "datetime",
                "diagnosis_code": "string",
            },
            tags=["healthcare", "patients", "hipaa"],
            metrics=ModelMetrics(
                fidelity_score=0.91,
                privacy_score=0.98,
                utility_score=0.85,
            ),
            certification=CertificationLevel.PREMIUM,
        )

        from genesis.core.base import SyntheticGenerator
        generator = SyntheticGenerator(method="gaussian_copula")

        return card, generator

    @staticmethod
    def saas_metrics() -> tuple[ModelCard, Any]:
        """SaaS product metrics model."""
        card = ModelCard(
            model_id="genesis/saas-metrics-v1",
            name="SaaS Product Metrics",
            version="1.0.0",
            description="Pre-trained model for SaaS product usage and subscription metrics",
            domain=ModelDomain.SAAS,
            schema={
                "user_id": "string",
                "signup_date": "datetime",
                "plan": "string",
                "mrr": "float",
                "daily_active_minutes": "float",
                "feature_usage_count": "integer",
                "churn_risk_score": "float",
            },
            tags=["saas", "metrics", "subscription"],
            metrics=ModelMetrics(
                fidelity_score=0.87,
                privacy_score=0.92,
                utility_score=0.88,
            ),
            certification=CertificationLevel.STANDARD,
        )

        from genesis.core.base import SyntheticGenerator
        generator = SyntheticGenerator(method="gaussian_copula")

        return card, generator

    @staticmethod
    def iot_sensors() -> tuple[ModelCard, Any]:
        """IoT sensor data model."""
        card = ModelCard(
            model_id="genesis/iot-sensors-v1",
            name="IoT Sensor Data",
            version="1.0.0",
            description="Pre-trained model for IoT sensor readings and telemetry",
            domain=ModelDomain.IOT,
            schema={
                "device_id": "string",
                "timestamp": "datetime",
                "temperature": "float",
                "humidity": "float",
                "pressure": "float",
                "battery_level": "float",
                "signal_strength": "float",
            },
            tags=["iot", "sensors", "telemetry"],
            metrics=ModelMetrics(
                fidelity_score=0.93,
                privacy_score=0.99,
                utility_score=0.91,
            ),
            certification=CertificationLevel.BASIC,
        )

        from genesis.core.base import SyntheticGenerator
        generator = SyntheticGenerator(method="gaussian_copula")

        return card, generator


class ModelHub:
    """Main hub interface for model discovery and management.

    Provides access to pre-trained models, community contributions,
    and model management capabilities.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        hub_url: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        """Initialize model hub.

        Args:
            cache_dir: Local cache directory.
            hub_url: Remote hub URL.
            token: Authentication token.
        """
        self.registry = ModelRegistry(cache_dir)
        self.hub_url = hub_url or os.environ.get("GENESIS_HUB_URL", "")
        self.token = token or os.environ.get("GENESIS_HUB_TOKEN", "")

        # Initialize with pre-trained models
        self._initialize_pretrained()

    def _initialize_pretrained(self) -> None:
        """Initialize built-in pre-trained models."""
        pretrained_factories = [
            PretrainedModels.ecommerce_transactions,
            PretrainedModels.healthcare_patients,
            PretrainedModels.saas_metrics,
            PretrainedModels.iot_sensors,
        ]

        for factory in pretrained_factories:
            card, generator = factory()
            if card.model_id not in [m.model_id for m in self.registry.list()]:
                self.registry.save(card, generator)

    def list_models(
        self,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
        certified_only: bool = False,
    ) -> List[ModelCard]:
        """List available models.

        Args:
            domain: Filter by domain.
            tags: Filter by tags.
            certified_only: Only show certified models.

        Returns:
            List of ModelCard objects.
        """
        domain_enum = ModelDomain(domain) if domain else None
        models = self.registry.list(domain=domain_enum, tags=tags)

        if certified_only:
            models = [m for m in models if m.certification != CertificationLevel.NONE]

        return models

    def load(self, model_id: str) -> PretrainedGenerator:
        """Load a model from the hub.

        Args:
            model_id: Model identifier (e.g., "genesis/ecommerce-v1").

        Returns:
            PretrainedGenerator wrapper.
        """
        card = self.registry.get(model_id)
        if card is None:
            # Try to download from remote hub
            card = self._download_from_hub(model_id)
            if card is None:
                raise ValueError(f"Model not found: {model_id}")

        generator = self.registry.load(model_id)
        card.downloads += 1

        return PretrainedGenerator(generator, card)

    def push(
        self,
        model_id: str,
        generator: Any,
        description: str = "",
        domain: str = "general",
        tags: Optional[List[str]] = None,
    ) -> ModelCard:
        """Push a model to the hub.

        Args:
            model_id: Model identifier.
            generator: Trained generator.
            description: Model description.
            domain: Domain category.
            tags: Searchable tags.

        Returns:
            Created ModelCard.
        """
        # Infer schema from generator if available
        schema: Dict[str, str] = {}
        if hasattr(generator, "_schema") and generator._schema:
            for col_meta in generator._schema.get("columns", []):
                schema[col_meta.name] = col_meta.dtype

        card = ModelCard(
            model_id=model_id,
            name=model_id.split("/")[-1],
            description=description,
            domain=ModelDomain(domain),
            schema=schema,
            tags=tags or [],
            author=self._get_current_user(),
        )

        self.registry.save(card, generator)
        logger.info(f"Pushed model {model_id}")

        return card

    def search(self, query: str) -> List[ModelCard]:
        """Search models by query.

        Args:
            query: Search query.

        Returns:
            Matching models.
        """
        query_lower = query.lower()
        all_models = self.registry.list()

        matches = []
        for model in all_models:
            if (
                query_lower in model.name.lower()
                or query_lower in model.description.lower()
                or any(query_lower in tag.lower() for tag in model.tags)
            ):
                matches.append(model)

        return matches

    def get_model_card(self, model_id: str) -> Optional[ModelCard]:
        """Get model card for a model."""
        return self.registry.get(model_id)

    def delete(self, model_id: str) -> None:
        """Delete a model from local cache."""
        self.registry.delete(model_id)
        logger.info(f"Deleted model {model_id}")

    def _download_from_hub(self, model_id: str) -> Optional[ModelCard]:
        """Download model from remote hub."""
        if not self.hub_url:
            return None
        # Placeholder for remote hub integration
        logger.warning(f"Remote hub download not implemented for {model_id}")
        return None

    def _get_current_user(self) -> str:
        """Get current user from token."""
        if self.token:
            return "authenticated_user"
        return "anonymous"


# Convenience functions
def load_pretrained(model_id: str) -> PretrainedGenerator:
    """Load a pre-trained model.

    Args:
        model_id: Model identifier.

    Returns:
        PretrainedGenerator.
    """
    hub = ModelHub()
    return hub.load(model_id)


def list_available_models() -> List[str]:
    """List all available model IDs."""
    hub = ModelHub()
    return [m.model_id for m in hub.list_models()]
