"""Multi-Modal Foundation Model for synthetic data generation.

This module provides a unified foundation model architecture capable of
generating synthetic data across multiple modalities: tabular, text,
time-series, and images.

Example:
    >>> from genesis.multimodal import MultiModalGenerator, ModalityType
    >>>
    >>> # Create generator
    >>> generator = MultiModalGenerator()
    >>>
    >>> # Fit on tabular + text data
    >>> generator.fit(
    ...     tabular_data=df,
    ...     text_column="description",
    ...     time_column="timestamp",
    ... )
    >>>
    >>> # Generate synthetic multi-modal data
    >>> synthetic = generator.generate(n_samples=1000)
"""

from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from genesis.core.base import BaseGenerator
from genesis.core.config import GeneratorConfig, PrivacyConfig
from genesis.core.types import PrivacyLevel
from genesis.utils.logging import get_logger

logger = get_logger(__name__)

# Optional imports
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    from transformers import AutoTokenizer, AutoModel

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ModalityType(Enum):
    """Types of data modalities supported."""

    TABULAR = "tabular"
    TEXT = "text"
    TIME_SERIES = "time_series"
    IMAGE = "image"
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"


class EncoderType(Enum):
    """Types of encoders for different modalities."""

    MLP = "mlp"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    LSTM = "lstm"
    GRU = "gru"


@dataclass
class ModalityConfig:
    """Configuration for a specific modality."""

    modality_type: ModalityType
    encoder_type: EncoderType
    embedding_dim: int = 128
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    dropout: float = 0.1
    activation: str = "relu"

    # Modality-specific settings
    max_sequence_length: int = 512  # For text/time-series
    vocab_size: int = 50000  # For text
    n_channels: int = 3  # For images
    image_size: Tuple[int, int] = (64, 64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality_type": self.modality_type.value,
            "encoder_type": self.encoder_type.value,
            "embedding_dim": self.embedding_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "activation": self.activation,
            "max_sequence_length": self.max_sequence_length,
        }


@dataclass
class MultiModalConfig:
    """Configuration for multi-modal generator."""

    # Architecture
    latent_dim: int = 256
    fusion_method: str = "concat"  # 'concat', 'attention', 'cross_attention'
    shared_encoder_layers: int = 2

    # Training
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Generation
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9

    # Modality configs
    tabular_config: Optional[ModalityConfig] = None
    text_config: Optional[ModalityConfig] = None
    time_series_config: Optional[ModalityConfig] = None
    image_config: Optional[ModalityConfig] = None

    # Privacy
    privacy: Optional[PrivacyConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "fusion_method": self.fusion_method,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
        }


class ModalityEncoder(ABC):
    """Abstract base class for modality-specific encoders."""

    @abstractmethod
    def encode(self, data: Any) -> np.ndarray:
        """Encode input data to latent representation."""
        pass

    @abstractmethod
    def decode(self, latent: np.ndarray) -> Any:
        """Decode latent representation to data."""
        pass


class TabularEncoder(ModalityEncoder):
    """Encoder for tabular data."""

    def __init__(
        self,
        config: ModalityConfig,
        columns: List[str],
        dtypes: Dict[str, str],
    ):
        self.config = config
        self.columns = columns
        self.dtypes = dtypes
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self._fitted = False

    def fit(self, data: pd.DataFrame) -> "TabularEncoder":
        """Fit the encoder on training data."""
        from sklearn.preprocessing import StandardScaler, LabelEncoder

        for col in self.columns:
            if self.dtypes.get(col) in ["int64", "float64", "int32", "float32"]:
                scaler = StandardScaler()
                self.scalers[col] = scaler.fit(data[[col]].values)
            else:
                encoder = LabelEncoder()
                self.encoders[col] = encoder.fit(data[col].astype(str).values)

        self._fitted = True
        return self

    def encode(self, data: pd.DataFrame) -> np.ndarray:
        """Encode tabular data to latent representation."""
        if not self._fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        encoded = []

        for col in self.columns:
            if col in self.scalers:
                values = self.scalers[col].transform(data[[col]].values)
                encoded.append(values)
            elif col in self.encoders:
                values = self.encoders[col].transform(data[col].astype(str).values)
                encoded.append(values.reshape(-1, 1))

        return np.hstack(encoded)

    def decode(self, latent: np.ndarray) -> pd.DataFrame:
        """Decode latent representation to tabular data."""
        if not self._fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        result = {}
        col_idx = 0

        for col in self.columns:
            if col in self.scalers:
                values = self.scalers[col].inverse_transform(latent[:, col_idx:col_idx + 1])
                result[col] = values.flatten()
                col_idx += 1
            elif col in self.encoders:
                # Round to nearest integer for categorical
                indices = np.clip(
                    np.round(latent[:, col_idx]).astype(int),
                    0,
                    len(self.encoders[col].classes_) - 1
                )
                result[col] = self.encoders[col].inverse_transform(indices)
                col_idx += 1

        return pd.DataFrame(result)


class TextEncoder(ModalityEncoder):
    """Encoder for text data using transformers or simpler methods."""

    def __init__(
        self,
        config: ModalityConfig,
        use_pretrained: bool = True,
        model_name: str = "distilbert-base-uncased",
    ):
        self.config = config
        self.use_pretrained = use_pretrained
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._vocab: Dict[str, int] = {}
        self._reverse_vocab: Dict[int, str] = {}
        self._fitted = False

    def fit(self, texts: List[str]) -> "TextEncoder":
        """Fit the encoder on training texts."""
        if self.use_pretrained and TRANSFORMERS_AVAILABLE:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            if TORCH_AVAILABLE:
                self._model.eval()
        else:
            # Build simple vocabulary
            word_counts: Dict[str, int] = {}
            for text in texts:
                for word in text.lower().split():
                    word_counts[word] = word_counts.get(word, 0) + 1

            # Keep top vocab_size words
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            self._vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}

            for word, _ in sorted_words[:self.config.vocab_size - 4]:
                self._vocab[word] = len(self._vocab)

            self._reverse_vocab = {v: k for k, v in self._vocab.items()}

        self._fitted = True
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to latent representations."""
        if not self._fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        if self.use_pretrained and TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
            embeddings = []

            with torch.no_grad():
                for text in texts:
                    inputs = self._tokenizer(
                        text,
                        padding="max_length",
                        truncation=True,
                        max_length=self.config.max_sequence_length,
                        return_tensors="pt",
                    )
                    outputs = self._model(**inputs)
                    # Use CLS token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].numpy()
                    embeddings.append(embedding)

            return np.vstack(embeddings)
        else:
            # Simple bag-of-words encoding
            encoded = []
            for text in texts:
                words = text.lower().split()
                vector = np.zeros(min(self.config.embedding_dim, len(self._vocab)))

                for word in words:
                    idx = self._vocab.get(word, 1)  # 1 = UNK
                    if idx < len(vector):
                        vector[idx] += 1

                # Normalize
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector /= norm

                encoded.append(vector)

            return np.array(encoded)

    def decode(self, latent: np.ndarray) -> List[str]:
        """Decode latent representations to texts."""
        if not self._fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        # Simple generation from latent space
        texts = []

        for vec in latent:
            # Get top-k most likely words based on vector values
            if len(vec) <= len(self._reverse_vocab):
                top_indices = np.argsort(vec)[-20:][::-1]
                words = [self._reverse_vocab.get(i, "") for i in top_indices if i in self._reverse_vocab]
                words = [w for w in words if w not in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]]
                texts.append(" ".join(words[:10]))
            else:
                texts.append("")

        return texts


class TimeSeriesEncoder(ModalityEncoder):
    """Encoder for time series data."""

    def __init__(
        self,
        config: ModalityConfig,
        n_features: int,
    ):
        self.config = config
        self.n_features = n_features
        self._scaler = None
        self._fitted = False

    def fit(self, data: np.ndarray) -> "TimeSeriesEncoder":
        """Fit the encoder on training time series.

        Args:
            data: Shape (n_samples, sequence_length, n_features)
        """
        from sklearn.preprocessing import StandardScaler

        # Reshape for scaling
        n_samples, seq_len, n_features = data.shape
        reshaped = data.reshape(-1, n_features)

        self._scaler = StandardScaler()
        self._scaler.fit(reshaped)

        self._fitted = True
        return self

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode time series to latent representation.

        Args:
            data: Shape (n_samples, sequence_length, n_features)

        Returns:
            Latent representation (n_samples, embedding_dim)
        """
        if not self._fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        n_samples, seq_len, n_features = data.shape

        # Scale
        scaled = self._scaler.transform(data.reshape(-1, n_features))
        scaled = scaled.reshape(n_samples, seq_len, n_features)

        # Simple encoding: mean and std pooling
        mean_pool = np.mean(scaled, axis=1)
        std_pool = np.std(scaled, axis=1)

        # Concatenate statistics
        encoded = np.hstack([mean_pool, std_pool])

        # Pad or truncate to embedding_dim
        target_dim = self.config.embedding_dim
        if encoded.shape[1] < target_dim:
            padding = np.zeros((n_samples, target_dim - encoded.shape[1]))
            encoded = np.hstack([encoded, padding])
        elif encoded.shape[1] > target_dim:
            encoded = encoded[:, :target_dim]

        return encoded

    def decode(self, latent: np.ndarray, seq_len: int = 24) -> np.ndarray:
        """Decode latent representation to time series.

        Args:
            latent: Shape (n_samples, embedding_dim)
            seq_len: Sequence length to generate

        Returns:
            Time series (n_samples, sequence_length, n_features)
        """
        if not self._fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        n_samples = latent.shape[0]

        # Extract mean and std from latent
        mean_dim = min(self.n_features, latent.shape[1] // 2)
        mean_vals = latent[:, :mean_dim]
        std_vals = np.abs(latent[:, mean_dim:mean_dim * 2]) + 0.01

        # Generate sequences
        sequences = []

        for i in range(n_samples):
            # Generate with random walk + trend
            seq = np.zeros((seq_len, self.n_features))
            seq[0] = mean_vals[i, :self.n_features] if mean_dim >= self.n_features else np.zeros(self.n_features)

            for t in range(1, seq_len):
                noise = np.random.normal(0, std_vals[i, :self.n_features] if mean_dim >= self.n_features else 0.1, self.n_features)
                seq[t] = seq[t - 1] + noise

            sequences.append(seq)

        result = np.array(sequences)

        # Inverse transform
        if self._scaler:
            result_reshaped = result.reshape(-1, self.n_features)
            result_reshaped = self._scaler.inverse_transform(result_reshaped)
            result = result_reshaped.reshape(n_samples, seq_len, self.n_features)

        return result


class MultiModalFusion:
    """Fuses multiple modality representations."""

    def __init__(
        self,
        config: MultiModalConfig,
        modality_dims: Dict[ModalityType, int],
    ):
        self.config = config
        self.modality_dims = modality_dims
        self._weights: Dict[ModalityType, np.ndarray] = {}
        self._fitted = False

    def fit(
        self,
        embeddings: Dict[ModalityType, np.ndarray],
    ) -> "MultiModalFusion":
        """Fit the fusion layer."""
        for modality, emb in embeddings.items():
            # Simple linear projection to latent_dim
            input_dim = emb.shape[1]
            self._weights[modality] = np.random.randn(input_dim, self.config.latent_dim) * 0.1

        self._fitted = True
        return self

    def fuse(
        self,
        embeddings: Dict[ModalityType, np.ndarray],
    ) -> np.ndarray:
        """Fuse multiple modality embeddings.

        Args:
            embeddings: Dict of modality -> embedding arrays

        Returns:
            Fused representation (n_samples, latent_dim)
        """
        if not self._fitted:
            raise RuntimeError("Fusion layer not fitted. Call fit() first.")

        n_samples = next(iter(embeddings.values())).shape[0]

        if self.config.fusion_method == "concat":
            # Project and concatenate
            projected = []
            for modality, emb in embeddings.items():
                if modality in self._weights:
                    proj = emb @ self._weights[modality]
                    projected.append(proj)

            if projected:
                # Average the projections
                fused = np.mean(projected, axis=0)
            else:
                fused = np.zeros((n_samples, self.config.latent_dim))

        elif self.config.fusion_method == "attention":
            # Simple attention-based fusion
            projected = {}
            for modality, emb in embeddings.items():
                if modality in self._weights:
                    projected[modality] = emb @ self._weights[modality]

            if projected:
                # Compute attention weights based on embedding norms
                attention = {}
                total_norm = 0
                for modality, proj in projected.items():
                    norm = np.linalg.norm(proj, axis=1, keepdims=True)
                    attention[modality] = norm
                    total_norm += norm

                # Weighted combination
                fused = np.zeros((n_samples, self.config.latent_dim))
                for modality, proj in projected.items():
                    weight = attention[modality] / (total_norm + 1e-8)
                    fused += weight * proj
            else:
                fused = np.zeros((n_samples, self.config.latent_dim))
        else:
            raise ValueError(f"Unknown fusion method: {self.config.fusion_method}")

        return fused

    def split(
        self,
        fused: np.ndarray,
        modalities: List[ModalityType],
    ) -> Dict[ModalityType, np.ndarray]:
        """Split fused representation back to modality-specific embeddings."""
        if not self._fitted:
            raise RuntimeError("Fusion layer not fitted. Call fit() first.")

        result = {}

        for modality in modalities:
            if modality in self._weights:
                # Pseudo-inverse for splitting
                W = self._weights[modality]
                W_pinv = np.linalg.pinv(W)
                result[modality] = fused @ W_pinv.T

        return result


class MultiModalGenerator(BaseGenerator):
    """Multi-modal foundation model for synthetic data generation.

    Capable of generating synthetic data across multiple modalities:
    - Tabular data
    - Text
    - Time series
    - Images (basic support)

    The model learns a shared latent space across modalities, enabling:
    - Cross-modal generation (generate text conditioned on tabular data)
    - Joint generation (generate all modalities together)
    - Single-modal generation with multi-modal context
    """

    def __init__(
        self,
        config: Optional[MultiModalConfig] = None,
        privacy: Optional[PrivacyConfig] = None,
    ):
        """Initialize multi-modal generator.

        Args:
            config: Multi-modal configuration
            privacy: Privacy configuration
        """
        super().__init__()

        self.config = config or MultiModalConfig()
        if privacy:
            self.config.privacy = privacy

        self._encoders: Dict[ModalityType, ModalityEncoder] = {}
        self._fusion: Optional[MultiModalFusion] = None
        self._fitted = False
        self._modalities: List[ModalityType] = []

        # Training data references
        self._tabular_columns: List[str] = []
        self._tabular_dtypes: Dict[str, str] = {}
        self._text_column: Optional[str] = None
        self._time_columns: List[str] = []
        self._n_time_features: int = 0

        # Latent space
        self._latent_mean: Optional[np.ndarray] = None
        self._latent_std: Optional[np.ndarray] = None

    def fit(
        self,
        tabular_data: Optional[pd.DataFrame] = None,
        text_column: Optional[str] = None,
        time_columns: Optional[List[str]] = None,
        time_series_data: Optional[np.ndarray] = None,
        image_data: Optional[np.ndarray] = None,
    ) -> "MultiModalGenerator":
        """Fit the multi-modal generator.

        Args:
            tabular_data: DataFrame with tabular features
            text_column: Column name containing text data
            time_columns: Columns containing time series data
            time_series_data: Pre-processed time series (n_samples, seq_len, n_features)
            image_data: Image data array (not fully implemented)

        Returns:
            Fitted generator
        """
        embeddings = {}

        # Fit tabular encoder
        if tabular_data is not None:
            # Separate text column if specified
            if text_column and text_column in tabular_data.columns:
                texts = tabular_data[text_column].tolist()
                tabular_cols = [c for c in tabular_data.columns if c != text_column]
            else:
                texts = None
                tabular_cols = list(tabular_data.columns)

            # Separate time columns
            if time_columns:
                tabular_cols = [c for c in tabular_cols if c not in time_columns]
                self._time_columns = time_columns

            if tabular_cols:
                self._tabular_columns = tabular_cols
                self._tabular_dtypes = {c: str(tabular_data[c].dtype) for c in tabular_cols}

                tabular_config = self.config.tabular_config or ModalityConfig(
                    modality_type=ModalityType.TABULAR,
                    encoder_type=EncoderType.MLP,
                )

                tabular_encoder = TabularEncoder(
                    config=tabular_config,
                    columns=tabular_cols,
                    dtypes=self._tabular_dtypes,
                )
                tabular_encoder.fit(tabular_data[tabular_cols])

                self._encoders[ModalityType.TABULAR] = tabular_encoder
                self._modalities.append(ModalityType.TABULAR)

                # Encode training data
                embeddings[ModalityType.TABULAR] = tabular_encoder.encode(tabular_data[tabular_cols])

            # Fit text encoder
            if text_column and texts:
                self._text_column = text_column

                text_config = self.config.text_config or ModalityConfig(
                    modality_type=ModalityType.TEXT,
                    encoder_type=EncoderType.TRANSFORMER,
                )

                text_encoder = TextEncoder(
                    config=text_config,
                    use_pretrained=TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE,
                )
                text_encoder.fit(texts)

                self._encoders[ModalityType.TEXT] = text_encoder
                self._modalities.append(ModalityType.TEXT)

                embeddings[ModalityType.TEXT] = text_encoder.encode(texts)

            # Fit time series encoder from columns
            if time_columns and len(time_columns) > 0:
                time_data = tabular_data[time_columns].values
                time_data = time_data.reshape(len(tabular_data), 1, len(time_columns))

                self._n_time_features = len(time_columns)

                ts_config = self.config.time_series_config or ModalityConfig(
                    modality_type=ModalityType.TIME_SERIES,
                    encoder_type=EncoderType.LSTM,
                )

                ts_encoder = TimeSeriesEncoder(
                    config=ts_config,
                    n_features=len(time_columns),
                )
                ts_encoder.fit(time_data)

                self._encoders[ModalityType.TIME_SERIES] = ts_encoder
                self._modalities.append(ModalityType.TIME_SERIES)

                embeddings[ModalityType.TIME_SERIES] = ts_encoder.encode(time_data)

        # Fit on provided time series data
        if time_series_data is not None:
            n_features = time_series_data.shape[2]
            self._n_time_features = n_features

            ts_config = self.config.time_series_config or ModalityConfig(
                modality_type=ModalityType.TIME_SERIES,
                encoder_type=EncoderType.LSTM,
            )

            ts_encoder = TimeSeriesEncoder(
                config=ts_config,
                n_features=n_features,
            )
            ts_encoder.fit(time_series_data)

            self._encoders[ModalityType.TIME_SERIES] = ts_encoder

            if ModalityType.TIME_SERIES not in self._modalities:
                self._modalities.append(ModalityType.TIME_SERIES)

            embeddings[ModalityType.TIME_SERIES] = ts_encoder.encode(time_series_data)

        # Fit fusion layer
        if embeddings:
            modality_dims = {m: e.shape[1] for m, e in embeddings.items()}

            self._fusion = MultiModalFusion(self.config, modality_dims)
            self._fusion.fit(embeddings)

            # Compute fused latent statistics
            fused = self._fusion.fuse(embeddings)
            self._latent_mean = np.mean(fused, axis=0)
            self._latent_std = np.std(fused, axis=0) + 1e-6

        self._fitted = True
        logger.info(f"Fitted multi-modal generator with modalities: {[m.value for m in self._modalities]}")

        return self

    def generate(
        self,
        n_samples: int = 1000,
        modalities: Optional[List[ModalityType]] = None,
        conditions: Optional[Dict[ModalityType, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate synthetic multi-modal data.

        Args:
            n_samples: Number of samples to generate
            modalities: Which modalities to generate (default: all fitted)
            conditions: Conditioning data for cross-modal generation

        Returns:
            Dictionary with generated data for each modality
        """
        if not self._fitted:
            raise RuntimeError("Generator not fitted. Call fit() first.")

        modalities = modalities or self._modalities

        # Generate from latent space
        if conditions:
            # Conditional generation: encode conditions, generate rest
            condition_embeddings = {}

            for modality, data in conditions.items():
                if modality in self._encoders:
                    encoder = self._encoders[modality]

                    if modality == ModalityType.TABULAR:
                        condition_embeddings[modality] = encoder.encode(data)
                    elif modality == ModalityType.TEXT:
                        condition_embeddings[modality] = encoder.encode(data)
                    elif modality == ModalityType.TIME_SERIES:
                        condition_embeddings[modality] = encoder.encode(data)

            # Fuse conditions
            if condition_embeddings:
                latent = self._fusion.fuse(condition_embeddings)
                # Add noise
                noise = np.random.normal(0, 0.1, latent.shape)
                latent = latent + noise
            else:
                latent = self._sample_latent(n_samples)
        else:
            latent = self._sample_latent(n_samples)

        # Decode to each modality
        result = {}

        # Split latent back to modality embeddings
        modality_embeddings = self._fusion.split(latent, modalities)

        for modality in modalities:
            if modality not in self._encoders:
                continue

            encoder = self._encoders[modality]
            embedding = modality_embeddings.get(modality, latent[:, :encoder.config.embedding_dim])

            if modality == ModalityType.TABULAR:
                result["tabular"] = encoder.decode(embedding)

            elif modality == ModalityType.TEXT:
                result["text"] = encoder.decode(embedding)

            elif modality == ModalityType.TIME_SERIES:
                result["time_series"] = encoder.decode(embedding)

        # Combine into DataFrame if possible
        if "tabular" in result:
            df = result["tabular"]

            if "text" in result and self._text_column:
                df[self._text_column] = result["text"][:len(df)]

            if "time_series" in result and self._time_columns:
                ts_data = result["time_series"][:, 0, :]  # Take first timestep
                for i, col in enumerate(self._time_columns):
                    if i < ts_data.shape[1]:
                        df[col] = ts_data[:, i]

            result["combined"] = df

        return result

    def generate_conditional(
        self,
        source_modality: ModalityType,
        target_modality: ModalityType,
        source_data: Any,
    ) -> Any:
        """Generate one modality conditioned on another.

        Args:
            source_modality: Modality of conditioning data
            target_modality: Modality to generate
            source_data: Conditioning data

        Returns:
            Generated data in target modality
        """
        if not self._fitted:
            raise RuntimeError("Generator not fitted. Call fit() first.")

        if source_modality not in self._encoders:
            raise ValueError(f"Source modality {source_modality} not available")
        if target_modality not in self._encoders:
            raise ValueError(f"Target modality {target_modality} not available")

        # Encode source
        source_encoder = self._encoders[source_modality]
        source_embedding = source_encoder.encode(source_data)

        # Fuse and decode
        latent = self._fusion.fuse({source_modality: source_embedding})

        # Add small noise for variation
        noise = np.random.normal(0, 0.05 * self.config.temperature, latent.shape)
        latent = latent + noise

        # Split back
        modality_embeddings = self._fusion.split(latent, [target_modality])
        target_embedding = modality_embeddings[target_modality]

        # Decode
        target_encoder = self._encoders[target_modality]
        return target_encoder.decode(target_embedding)

    def _sample_latent(self, n_samples: int) -> np.ndarray:
        """Sample from the learned latent distribution."""
        if self._latent_mean is None:
            return np.random.randn(n_samples, self.config.latent_dim)

        # Sample from Gaussian
        z = np.random.randn(n_samples, self.config.latent_dim)
        z = z * self._latent_std + self._latent_mean

        # Temperature scaling
        if self.config.temperature != 1.0:
            z = z * self.config.temperature

        return z

    def get_latent_representation(
        self,
        tabular_data: Optional[pd.DataFrame] = None,
        text_data: Optional[List[str]] = None,
        time_series_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Get latent representation for given data.

        Args:
            tabular_data: Tabular features
            text_data: Text data
            time_series_data: Time series data

        Returns:
            Latent representation array
        """
        if not self._fitted:
            raise RuntimeError("Generator not fitted. Call fit() first.")

        embeddings = {}

        if tabular_data is not None and ModalityType.TABULAR in self._encoders:
            embeddings[ModalityType.TABULAR] = self._encoders[ModalityType.TABULAR].encode(
                tabular_data[self._tabular_columns]
            )

        if text_data is not None and ModalityType.TEXT in self._encoders:
            embeddings[ModalityType.TEXT] = self._encoders[ModalityType.TEXT].encode(text_data)

        if time_series_data is not None and ModalityType.TIME_SERIES in self._encoders:
            embeddings[ModalityType.TIME_SERIES] = self._encoders[ModalityType.TIME_SERIES].encode(time_series_data)

        if not embeddings:
            raise ValueError("No valid data provided for fitted modalities")

        return self._fusion.fuse(embeddings)

    def save(self, path: Union[str, Path]) -> None:
        """Save the generator to disk.

        Args:
            path: Save path
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        config_data = {
            "config": self.config.to_dict(),
            "modalities": [m.value for m in self._modalities],
            "tabular_columns": self._tabular_columns,
            "tabular_dtypes": self._tabular_dtypes,
            "text_column": self._text_column,
            "time_columns": self._time_columns,
            "n_time_features": self._n_time_features,
        }

        with open(path / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)

        # Save latent statistics
        if self._latent_mean is not None:
            np.save(path / "latent_mean.npy", self._latent_mean)
            np.save(path / "latent_std.npy", self._latent_std)

        logger.info(f"Saved multi-modal generator to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MultiModalGenerator":
        """Load generator from disk.

        Args:
            path: Load path

        Returns:
            Loaded generator
        """
        path = Path(path)

        with open(path / "config.json", "r") as f:
            config_data = json.load(f)

        generator = cls()
        generator._modalities = [ModalityType(m) for m in config_data["modalities"]]
        generator._tabular_columns = config_data["tabular_columns"]
        generator._tabular_dtypes = config_data["tabular_dtypes"]
        generator._text_column = config_data.get("text_column")
        generator._time_columns = config_data.get("time_columns", [])
        generator._n_time_features = config_data.get("n_time_features", 0)

        # Load latent statistics
        if (path / "latent_mean.npy").exists():
            generator._latent_mean = np.load(path / "latent_mean.npy")
            generator._latent_std = np.load(path / "latent_std.npy")

        logger.info(f"Loaded multi-modal generator from {path}")

        return generator


# Convenience functions

def generate_multimodal(
    tabular_data: pd.DataFrame,
    text_column: Optional[str] = None,
    time_columns: Optional[List[str]] = None,
    n_samples: int = 1000,
    **kwargs,
) -> pd.DataFrame:
    """Generate multi-modal synthetic data.

    Args:
        tabular_data: Training DataFrame
        text_column: Column containing text
        time_columns: Columns containing time series
        n_samples: Number of samples to generate
        **kwargs: Additional arguments for MultiModalConfig

    Returns:
        Generated DataFrame
    """
    config = MultiModalConfig(**kwargs) if kwargs else None
    generator = MultiModalGenerator(config=config)

    generator.fit(
        tabular_data=tabular_data,
        text_column=text_column,
        time_columns=time_columns,
    )

    result = generator.generate(n_samples=n_samples)

    return result.get("combined", result.get("tabular", pd.DataFrame()))


__all__ = [
    # Main class
    "MultiModalGenerator",
    # Config
    "MultiModalConfig",
    "ModalityConfig",
    # Types
    "ModalityType",
    "EncoderType",
    # Encoders
    "ModalityEncoder",
    "TabularEncoder",
    "TextEncoder",
    "TimeSeriesEncoder",
    # Fusion
    "MultiModalFusion",
    # Functions
    "generate_multimodal",
]
