"""Cross-modal synthetic data generation.

This module provides capabilities for generating paired data across
different modalities (tabular + text, tabular + image, etc.).

Example:
    >>> from genesis.crossmodal import CrossModalGenerator
    >>>
    >>> gen = CrossModalGenerator()
    >>> gen.fit(
    ...     tabular=patient_records,
    ...     text=clinical_notes,
    ...     pairing="patient_id",
    ... )
    >>>
    >>> result = gen.generate(n_samples=1000)
    >>> print(result.tabular)  # Synthetic patient records
    >>> print(result.text)     # Matching clinical notes
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class Modality(Enum):
    """Supported data modalities."""

    TABULAR = "tabular"
    TEXT = "text"
    IMAGE = "image"
    TIMESERIES = "timeseries"
    AUDIO = "audio"


@dataclass
class ModalityData:
    """Container for data of a specific modality."""

    modality: Modality
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        if isinstance(self.data, (pd.DataFrame, list)):
            return len(self.data)
        return 0


@dataclass
class CrossModalResult:
    """Result of cross-modal generation."""

    tabular: Optional[pd.DataFrame] = None
    text: Optional[List[str]] = None
    images: Optional[List[Any]] = None
    timeseries: Optional[List[pd.DataFrame]] = None
    n_samples: int = 0
    pairings: Optional[Dict[int, Dict[str, int]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "has_tabular": self.tabular is not None,
            "has_text": self.text is not None,
            "has_images": self.images is not None,
            "has_timeseries": self.timeseries is not None,
        }


class ModalityEncoder(ABC):
    """Base class for encoding modality-specific data."""

    @abstractmethod
    def fit(self, data: Any) -> "ModalityEncoder":
        """Fit encoder to data."""
        pass

    @abstractmethod
    def encode(self, data: Any) -> np.ndarray:
        """Encode data to latent representation."""
        pass

    @abstractmethod
    def decode(self, latent: np.ndarray) -> Any:
        """Decode latent representation to data."""
        pass


class TabularEncoder(ModalityEncoder):
    """Encoder for tabular data using learned embeddings."""

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self._means: Optional[pd.Series] = None
        self._stds: Optional[pd.Series] = None
        self._columns: List[str] = []
        self._categorical_encoders: Dict[str, Dict[Any, int]] = {}

    def fit(self, data: pd.DataFrame) -> "TabularEncoder":
        """Fit encoder to tabular data."""
        self._columns = list(data.columns)

        # Compute normalization stats for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        self._means = data[numeric_cols].mean()
        self._stds = data[numeric_cols].std().replace(0, 1)

        # Encode categorical columns
        for col in data.columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                unique_vals = data[col].unique()
                self._categorical_encoders[col] = {val: idx for idx, val in enumerate(unique_vals)}

        return self

    def encode(self, data: pd.DataFrame) -> np.ndarray:
        """Encode tabular data to normalized representation."""
        encoded = data.copy()

        # Normalize numeric columns
        for col in self._means.index:
            if col in encoded.columns:
                encoded[col] = (encoded[col] - self._means[col]) / self._stds[col]

        # Encode categoricals
        for col, mapping in self._categorical_encoders.items():
            if col in encoded.columns:
                encoded[col] = encoded[col].map(mapping).fillna(-1)

        return encoded.values.astype(np.float32)

    def decode(self, latent: np.ndarray) -> pd.DataFrame:
        """Decode latent representation to tabular data."""
        df = pd.DataFrame(latent, columns=self._columns)

        # Denormalize numeric columns
        for col in self._means.index:
            if col in df.columns:
                df[col] = df[col] * self._stds[col] + self._means[col]

        # Decode categoricals
        for col, mapping in self._categorical_encoders.items():
            if col in df.columns:
                reverse_mapping = {v: k for k, v in mapping.items()}
                df[col] = df[col].round().astype(int).map(reverse_mapping)

        return df


class TextEncoder(ModalityEncoder):
    """Encoder for text data using embeddings."""

    def __init__(self, embedding_dim: int = 768, model: str = "tfidf"):
        self.embedding_dim = embedding_dim
        self.model = model
        self._vectorizer = None
        self._vocab: List[str] = []

    def fit(self, data: List[str]) -> "TextEncoder":
        """Fit encoder to text data."""
        if self.model == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(max_features=self.embedding_dim)
            self._vectorizer.fit(data)
            self._vocab = self._vectorizer.get_feature_names_out().tolist()
        return self

    def encode(self, data: List[str]) -> np.ndarray:
        """Encode text to vector representation."""
        if self._vectorizer is not None:
            return self._vectorizer.transform(data).toarray()
        return np.zeros((len(data), self.embedding_dim))

    def decode(self, latent: np.ndarray) -> List[str]:
        """Decode latent representation to text (approximation)."""
        # Simple keyword-based reconstruction
        texts = []
        for row in latent:
            top_indices = np.argsort(row)[-10:]
            keywords = [self._vocab[i] for i in top_indices if i < len(self._vocab)]
            texts.append(" ".join(keywords))
        return texts


class CrossModalJointSpace:
    """Learns a joint latent space across modalities."""

    def __init__(self, latent_dim: int = 128):
        self.latent_dim = latent_dim
        self._modality_encoders: Dict[Modality, ModalityEncoder] = {}
        self._joint_mean: Optional[np.ndarray] = None
        self._joint_cov: Optional[np.ndarray] = None

    def add_modality(
        self,
        modality: Modality,
        encoder: ModalityEncoder,
    ) -> None:
        """Add a modality encoder."""
        self._modality_encoders[modality] = encoder

    def fit(
        self,
        modality_data: Dict[Modality, Any],
        pairing_indices: Optional[np.ndarray] = None,
    ) -> "CrossModalJointSpace":
        """Fit joint space to multi-modal data.

        Args:
            modality_data: Dictionary mapping modality to data
            pairing_indices: Indices indicating paired samples
        """
        # Encode each modality
        encoded = {}
        for modality, data in modality_data.items():
            if modality in self._modality_encoders:
                encoder = self._modality_encoders[modality]
                encoder.fit(data)
                encoded[modality] = encoder.encode(data)

        # Concatenate encoded representations
        all_encoded = []
        for modality in self._modality_encoders:
            if modality in encoded:
                # Pad or truncate to latent_dim
                enc = encoded[modality]
                if enc.shape[1] > self.latent_dim:
                    enc = enc[:, : self.latent_dim]
                elif enc.shape[1] < self.latent_dim:
                    padding = np.zeros((enc.shape[0], self.latent_dim - enc.shape[1]))
                    enc = np.concatenate([enc, padding], axis=1)
                all_encoded.append(enc)

        if all_encoded:
            # Average across modalities for joint representation
            joint = np.mean(all_encoded, axis=0)
            self._joint_mean = np.mean(joint, axis=0)
            self._joint_cov = np.cov(joint.T)

        return self

    def sample(self, n_samples: int) -> np.ndarray:
        """Sample from the joint latent space."""
        if self._joint_mean is None or self._joint_cov is None:
            raise RuntimeError("Joint space not fitted")

        return np.random.multivariate_normal(
            self._joint_mean,
            self._joint_cov,
            size=n_samples,
        )

    def decode(
        self,
        latent: np.ndarray,
        modalities: Optional[List[Modality]] = None,
    ) -> Dict[Modality, Any]:
        """Decode latent samples to each modality.

        Args:
            latent: Latent samples
            modalities: Modalities to decode (default: all)

        Returns:
            Dictionary mapping modality to decoded data
        """
        modalities = modalities or list(self._modality_encoders.keys())

        decoded = {}
        for modality in modalities:
            if modality in self._modality_encoders:
                encoder = self._modality_encoders[modality]
                decoded[modality] = encoder.decode(latent)

        return decoded


class CrossModalGenerator:
    """Generator for paired multi-modal synthetic data.

    Learns joint distributions across modalities and generates
    coherent paired data.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        modalities: Optional[List[Modality]] = None,
    ):
        """Initialize cross-modal generator.

        Args:
            latent_dim: Dimension of joint latent space
            modalities: Modalities to support
        """
        self.latent_dim = latent_dim
        self.modalities = modalities or [Modality.TABULAR, Modality.TEXT]

        self._joint_space = CrossModalJointSpace(latent_dim)
        self._is_fitted = False
        self._pairing_column: Optional[str] = None

    def fit(
        self,
        tabular: Optional[pd.DataFrame] = None,
        text: Optional[List[str]] = None,
        images: Optional[List[Any]] = None,
        timeseries: Optional[List[pd.DataFrame]] = None,
        pairing: Optional[str] = None,
    ) -> "CrossModalGenerator":
        """Fit generator to multi-modal data.

        Args:
            tabular: Tabular data (DataFrame)
            text: Text data (list of strings)
            images: Image data (list of arrays/paths)
            timeseries: Time series data (list of DataFrames)
            pairing: Column name for pairing samples across modalities

        Returns:
            Self
        """
        self._pairing_column = pairing
        modality_data = {}

        # Add modality encoders and data
        if tabular is not None:
            encoder = TabularEncoder(self.latent_dim)
            self._joint_space.add_modality(Modality.TABULAR, encoder)
            modality_data[Modality.TABULAR] = tabular
            logger.info(f"Added tabular data: {len(tabular)} rows")

        if text is not None:
            encoder = TextEncoder(self.latent_dim)
            self._joint_space.add_modality(Modality.TEXT, encoder)
            modality_data[Modality.TEXT] = text
            logger.info(f"Added text data: {len(text)} documents")

        if images is not None:
            # Placeholder for image encoder
            logger.info(f"Added image data: {len(images)} images")

        if timeseries is not None:
            logger.info(f"Added time series data: {len(timeseries)} sequences")

        # Fit joint space
        self._joint_space.fit(modality_data)
        self._is_fitted = True

        logger.info("Cross-modal generator fitted successfully")
        return self

    def generate(
        self,
        n_samples: int,
        modalities: Optional[List[Modality]] = None,
    ) -> CrossModalResult:
        """Generate paired multi-modal synthetic data.

        Args:
            n_samples: Number of samples to generate
            modalities: Modalities to generate (default: all fitted)

        Returns:
            CrossModalResult with generated data
        """
        if not self._is_fitted:
            raise RuntimeError("Generator not fitted. Call fit() first.")

        logger.info(f"Generating {n_samples} cross-modal samples")

        # Sample from joint space
        latent = self._joint_space.sample(n_samples)

        # Decode to each modality
        decoded = self._joint_space.decode(latent, modalities)

        # Build result
        result = CrossModalResult(
            n_samples=n_samples,
        )

        if Modality.TABULAR in decoded:
            result.tabular = decoded[Modality.TABULAR]
        if Modality.TEXT in decoded:
            result.text = decoded[Modality.TEXT]

        # Create pairing indices
        result.pairings = {i: {m.value: i for m in decoded.keys()} for i in range(n_samples)}

        return result

    def generate_conditional(
        self,
        n_samples: int,
        condition_modality: Modality,
        condition_data: Any,
    ) -> CrossModalResult:
        """Generate other modalities conditioned on one modality.

        Args:
            n_samples: Number of samples per condition
            condition_modality: Modality to condition on
            condition_data: Conditioning data

        Returns:
            CrossModalResult with generated data
        """
        if not self._is_fitted:
            raise RuntimeError("Generator not fitted. Call fit() first.")

        # Encode conditioning data
        if condition_modality in self._joint_space._modality_encoders:
            encoder = self._joint_space._modality_encoders[condition_modality]

            if condition_modality == Modality.TABULAR:
                condition_latent = encoder.encode(condition_data)
            elif condition_modality == Modality.TEXT:
                condition_latent = encoder.encode(condition_data)
            else:
                raise ValueError(f"Unsupported condition modality: {condition_modality}")

            # Add noise to latent
            noise = np.random.normal(0, 0.1, size=condition_latent.shape)
            latent = condition_latent + noise

            # Expand to n_samples if single condition
            if len(latent) == 1:
                latent = np.tile(latent, (n_samples, 1))
                noise = np.random.normal(0, 0.2, size=latent.shape)
                latent = latent + noise

            # Decode other modalities
            other_modalities = [
                m for m in self._joint_space._modality_encoders.keys() if m != condition_modality
            ]
            decoded = self._joint_space.decode(latent, other_modalities)

            # Build result
            result = CrossModalResult(n_samples=len(latent))

            if Modality.TABULAR in decoded:
                result.tabular = decoded[Modality.TABULAR]
            if Modality.TEXT in decoded:
                result.text = decoded[Modality.TEXT]

            return result

        raise ValueError(f"Modality not fitted: {condition_modality}")


class TabularTextGenerator:
    """Specialized generator for paired tabular and text data.

    Example use cases:
    - Patient records + clinical notes
    - Product data + descriptions
    - Transaction data + comments
    """

    def __init__(
        self,
        text_template: Optional[str] = None,
        use_llm: bool = False,
    ):
        """Initialize tabular-text generator.

        Args:
            text_template: Template for generating text from tabular
            use_llm: Whether to use LLM for text generation
        """
        self.text_template = text_template
        self.use_llm = use_llm
        self._tabular_generator = None
        self._text_patterns: Dict[str, List[str]] = {}
        self._column_to_text_map: Dict[str, str] = {}

    def fit(
        self,
        tabular: pd.DataFrame,
        text: List[str],
        text_column_mapping: Optional[Dict[str, str]] = None,
        discrete_columns: Optional[List[str]] = None,
    ) -> "TabularTextGenerator":
        """Fit generator to paired tabular and text data.

        Args:
            tabular: Tabular data
            text: Corresponding text data (same length as tabular)
            text_column_mapping: Map from tabular columns to text patterns
            discrete_columns: Categorical columns in tabular

        Returns:
            Self
        """
        from genesis import SyntheticGenerator

        # Fit tabular generator
        self._tabular_generator = SyntheticGenerator(method="gaussian_copula")
        self._tabular_generator.fit(tabular, discrete_columns=discrete_columns)

        # Learn text patterns per column value
        self._column_to_text_map = text_column_mapping or {}

        # Extract patterns from text (simple approach)
        for i, txt in enumerate(text):
            if i < len(tabular):
                row = tabular.iloc[i]
                for col in tabular.columns:
                    val = str(row[col])
                    if val not in self._text_patterns:
                        self._text_patterns[val] = []
                    # Store text snippets associated with this value
                    self._text_patterns[val].append(txt)

        logger.info(f"Fitted TabularTextGenerator on {len(tabular)} samples")
        return self

    def generate(
        self,
        n_samples: int,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Generate paired tabular and text data.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Tuple of (tabular DataFrame, list of texts)
        """
        if self._tabular_generator is None:
            raise RuntimeError("Generator not fitted. Call fit() first.")

        # Generate tabular data
        tabular = self._tabular_generator.generate(n_samples)

        # Generate corresponding text
        texts = []
        for i in range(len(tabular)):
            row = tabular.iloc[i]

            if self.text_template:
                # Use template
                text = self.text_template.format(**row.to_dict())
            else:
                # Pattern-based generation
                text_parts = []
                for _col, val in row.items():
                    val_str = str(val)
                    if val_str in self._text_patterns and self._text_patterns[val_str]:
                        # Sample from associated texts
                        sample_text = np.random.choice(self._text_patterns[val_str])
                        # Extract relevant portion
                        words = sample_text.split()[:10]
                        text_parts.append(" ".join(words))

                text = " ".join(text_parts) if text_parts else f"Record {i}"

            texts.append(text)

        return tabular, texts


__all__ = [
    # Main generators
    "CrossModalGenerator",
    "TabularTextGenerator",
    # Result types
    "CrossModalResult",
    "ModalityData",
    # Encoders
    "ModalityEncoder",
    "TabularEncoder",
    "TextEncoder",
    # Types
    "Modality",
    "CrossModalJointSpace",
]
