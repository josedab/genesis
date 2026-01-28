"""Base class for time series generators."""

from typing import List, Optional

import numpy as np
import pandas as pd

from genesis.core.base import BaseGenerator
from genesis.core.config import GeneratorConfig, PrivacyConfig, TimeSeriesConfig


class BaseTimeSeriesGenerator(BaseGenerator):
    """Base class for time series generators."""

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        privacy: Optional[PrivacyConfig] = None,
        ts_config: Optional[TimeSeriesConfig] = None,
    ) -> None:
        super().__init__(config, privacy)
        self.ts_config = ts_config or TimeSeriesConfig()

        self._n_features: Optional[int] = None
        self._feature_names: List[str] = []
        self._sequence_length: int = self.ts_config.sequence_length

    def _prepare_sequences(
        self,
        data: pd.DataFrame,
        sequence_length: Optional[int] = None,
    ) -> np.ndarray:
        """Prepare sequences from DataFrame.

        Args:
            data: DataFrame with time series data
            sequence_length: Length of each sequence (uses config if None)

        Returns:
            Array of shape (n_sequences, sequence_length, n_features)
        """
        seq_len = sequence_length or self._sequence_length
        self._feature_names = list(data.columns)
        self._n_features = len(self._feature_names)

        values = data.values.astype(np.float32)

        # Normalize to [0, 1]
        self._min_vals = values.min(axis=0)
        self._max_vals = values.max(axis=0)
        self._range = self._max_vals - self._min_vals
        self._range[self._range == 0] = 1  # Avoid division by zero

        normalized = (values - self._min_vals) / self._range

        # Create sequences
        n_sequences = len(normalized) - seq_len + 1
        if n_sequences <= 0:
            raise ValueError(f"Data too short for sequence length {seq_len}")

        sequences = np.zeros((n_sequences, seq_len, self._n_features))
        for i in range(n_sequences):
            sequences[i] = normalized[i : i + seq_len]

        return sequences

    def _sequences_to_dataframe(
        self,
        sequences: np.ndarray,
        denormalize: bool = True,
    ) -> pd.DataFrame:
        """Convert generated sequences back to DataFrame.

        Args:
            sequences: Array of shape (n_sequences, sequence_length, n_features)
            denormalize: Whether to denormalize values

        Returns:
            DataFrame with generated time series
        """
        # Flatten sequences (take all timesteps)
        n_seq, seq_len, n_feat = sequences.shape
        flat = sequences.reshape(-1, n_feat)

        if denormalize:
            flat = flat * self._range + self._min_vals

        return pd.DataFrame(flat, columns=self._feature_names)
