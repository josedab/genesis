"""Data transformers for encoding and decoding data."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class BaseTransformer(ABC):
    """Abstract base class for data transformers."""

    def __init__(self) -> None:
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @abstractmethod
    def fit(self, data: pd.Series) -> "BaseTransformer":
        """Fit the transformer to data."""
        pass

    @abstractmethod
    def transform(self, data: pd.Series) -> np.ndarray:
        """Transform data."""
        pass

    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> pd.Series:
        """Reverse transform data."""
        pass

    def fit_transform(self, data: pd.Series) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)


class NumericalTransformer(BaseTransformer):
    """Transformer for numerical columns using mode-specific normalization."""

    def __init__(
        self,
        n_modes: int = 10,
        clip: bool = True,
        epsilon: float = 1e-6,
    ) -> None:
        """Initialize numerical transformer.

        Args:
            n_modes: Number of Gaussian modes for VGM
            clip: Whether to clip values during inverse transform
            epsilon: Small value to avoid division by zero
        """
        super().__init__()
        self.n_modes = n_modes
        self.clip = clip
        self.epsilon = epsilon

        self._min_value: Optional[float] = None
        self._max_value: Optional[float] = None
        self._mean: Optional[float] = None
        self._std: Optional[float] = None
        self._modes: Optional[np.ndarray] = None
        self._mode_stds: Optional[np.ndarray] = None
        self._mode_weights: Optional[np.ndarray] = None

    def fit(self, data: pd.Series) -> "NumericalTransformer":
        """Fit the transformer to numerical data."""
        values = data.dropna().values.astype(float)

        self._min_value = float(np.min(values))
        self._max_value = float(np.max(values))
        self._mean = float(np.mean(values))
        self._std = float(np.std(values)) + self.epsilon

        # Fit Gaussian Mixture for mode-specific normalization
        if len(values) > self.n_modes:
            try:
                from sklearn.mixture import BayesianGaussianMixture

                bgm = BayesianGaussianMixture(
                    n_components=self.n_modes,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=0.001,
                    max_iter=100,
                    n_init=1,
                    random_state=42,
                )
                bgm.fit(values.reshape(-1, 1))

                self._modes = bgm.means_.flatten()
                self._mode_stds = np.sqrt(bgm.covariances_.flatten()) + self.epsilon
                self._mode_weights = bgm.weights_
            except Exception:
                # Fallback to simple normalization
                self._modes = np.array([self._mean])
                self._mode_stds = np.array([self._std])
                self._mode_weights = np.array([1.0])
        else:
            self._modes = np.array([self._mean])
            self._mode_stds = np.array([self._std])
            self._mode_weights = np.array([1.0])

        self._is_fitted = True
        return self

    def transform(self, data: pd.Series) -> np.ndarray:
        """Transform numerical data to normalized form.

        Returns array of shape (n_samples, n_modes + 1) where:
        - First column: normalized value
        - Remaining columns: mode probabilities (one-hot)
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted")

        values = data.fillna(self._mean).values.astype(float)
        n_samples = len(values)
        n_modes = len(self._modes)

        # Calculate mode assignments
        if n_modes > 1:
            # Calculate probability of each mode for each sample
            probs = np.zeros((n_samples, n_modes))
            for i, (mode, std, weight) in enumerate(
                zip(self._modes, self._mode_stds, self._mode_weights)
            ):
                probs[:, i] = weight * stats.norm.pdf(values, mode, std)

            # Normalize probabilities
            probs = probs / (probs.sum(axis=1, keepdims=True) + self.epsilon)

            # Get mode assignment
            mode_assignments = np.argmax(probs, axis=1)
        else:
            mode_assignments = np.zeros(n_samples, dtype=int)

        # Normalize values using assigned mode
        normalized = np.zeros(n_samples)
        for i in range(n_samples):
            mode_idx = mode_assignments[i]
            normalized[i] = (values[i] - self._modes[mode_idx]) / (4 * self._mode_stds[mode_idx])

        # Clip normalized values
        normalized = np.clip(normalized, -0.99, 0.99)

        # Create output: normalized value + one-hot mode
        output = np.zeros((n_samples, 1 + n_modes))
        output[:, 0] = normalized
        output[np.arange(n_samples), 1 + mode_assignments] = 1

        return output

    def inverse_transform(self, data: np.ndarray) -> pd.Series:
        """Inverse transform from normalized form back to original scale."""
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted")

        # Handle different input shapes
        if data.ndim == 1:
            normalized = data
            mode_assignments = np.zeros(len(data), dtype=int)
        else:
            normalized = data[:, 0]
            if data.shape[1] > 1:
                mode_assignments = np.argmax(data[:, 1:], axis=1)
            else:
                mode_assignments = np.zeros(len(data), dtype=int)

        # Inverse transform
        values = np.zeros(len(normalized))
        for i in range(len(normalized)):
            mode_idx = min(mode_assignments[i], len(self._modes) - 1)
            values[i] = normalized[i] * 4 * self._mode_stds[mode_idx] + self._modes[mode_idx]

        # Clip if requested
        if self.clip:
            values = np.clip(values, self._min_value, self._max_value)

        return pd.Series(values)

    def get_output_dim(self) -> int:
        """Get the output dimension."""
        if not self._is_fitted:
            return 1 + self.n_modes
        return 1 + len(self._modes)


class CategoricalTransformer(BaseTransformer):
    """Transformer for categorical columns using one-hot encoding."""

    def __init__(
        self,
        handle_unknown: str = "ignore",
        max_categories: Optional[int] = None,
    ) -> None:
        """Initialize categorical transformer.

        Args:
            handle_unknown: How to handle unknown categories ('ignore', 'error')
            max_categories: Maximum number of categories to keep
        """
        super().__init__()
        self.handle_unknown = handle_unknown
        self.max_categories = max_categories

        self._categories: Optional[List[Any]] = None
        self._category_map: Dict[Any, int] = {}
        self._unknown_value: Any = "<UNKNOWN>"

    def fit(self, data: pd.Series) -> "CategoricalTransformer":
        """Fit the transformer to categorical data."""
        # Get unique categories
        value_counts = data.value_counts()
        categories = value_counts.index.tolist()

        # Limit categories if specified
        if self.max_categories and len(categories) > self.max_categories:
            categories = categories[: self.max_categories]

        self._categories = categories
        self._category_map = {cat: i for i, cat in enumerate(categories)}

        self._is_fitted = True
        return self

    def transform(self, data: pd.Series) -> np.ndarray:
        """Transform categorical data to one-hot encoded form."""
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted")

        n_samples = len(data)
        n_categories = len(self._categories)

        output = np.zeros((n_samples, n_categories))

        for i, val in enumerate(data):
            if pd.isna(val):
                continue
            if val in self._category_map:
                output[i, self._category_map[val]] = 1
            elif self.handle_unknown == "ignore":
                # Assign to first category as fallback
                output[i, 0] = 1

        return output

    def inverse_transform(self, data: np.ndarray) -> pd.Series:
        """Inverse transform from one-hot back to categories."""
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted")

        # Get category indices from one-hot or probabilities
        if data.ndim == 1:
            indices = data.astype(int)
        else:
            indices = np.argmax(data, axis=1)

        # Map back to categories
        values = []
        for idx in indices:
            if 0 <= idx < len(self._categories):
                values.append(self._categories[idx])
            else:
                values.append(self._categories[0])

        return pd.Series(values)

    def get_output_dim(self) -> int:
        """Get the output dimension."""
        if not self._is_fitted:
            return 0
        return len(self._categories)


class DatetimeTransformer(BaseTransformer):
    """Transformer for datetime columns."""

    def __init__(
        self,
        features: Optional[List[str]] = None,
        cyclical: bool = True,
    ) -> None:
        """Initialize datetime transformer.

        Args:
            features: Features to extract ('year', 'month', 'day', 'hour', etc.)
            cyclical: Whether to use cyclical encoding for periodic features
        """
        super().__init__()
        self.features = features or ["year", "month", "day", "dayofweek", "hour"]
        self.cyclical = cyclical

        self._min_datetime: Optional[pd.Timestamp] = None
        self._max_datetime: Optional[pd.Timestamp] = None

    def fit(self, data: pd.Series) -> "DatetimeTransformer":
        """Fit the transformer to datetime data."""
        dt_data = pd.to_datetime(data, errors="coerce")
        self._min_datetime = dt_data.min()
        self._max_datetime = dt_data.max()
        self._is_fitted = True
        return self

    def transform(self, data: pd.Series) -> np.ndarray:
        """Transform datetime to numerical features."""
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted")

        dt_data = pd.to_datetime(data, errors="coerce")
        features = []

        for feat in self.features:
            if feat == "year":
                values = dt_data.dt.year.values
                features.append(self._normalize(values))
            elif feat == "month":
                values = dt_data.dt.month.values
                if self.cyclical:
                    features.extend(self._cyclical_encode(values, 12))
                else:
                    features.append(values / 12)
            elif feat == "day":
                values = dt_data.dt.day.values
                if self.cyclical:
                    features.extend(self._cyclical_encode(values, 31))
                else:
                    features.append(values / 31)
            elif feat == "dayofweek":
                values = dt_data.dt.dayofweek.values
                if self.cyclical:
                    features.extend(self._cyclical_encode(values, 7))
                else:
                    features.append(values / 7)
            elif feat == "hour":
                values = dt_data.dt.hour.values
                if self.cyclical:
                    features.extend(self._cyclical_encode(values, 24))
                else:
                    features.append(values / 24)

        return np.column_stack(features)

    def inverse_transform(self, data: np.ndarray) -> pd.Series:
        """Inverse transform from features back to datetime."""
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted")

        # Simplified inverse: generate datetimes within learned range
        time_range = (self._max_datetime - self._min_datetime).total_seconds()

        # Use first feature as relative position
        positions = data[:, 0] if data.ndim > 1 else data
        positions = np.clip(positions, 0, 1)

        timestamps = [
            self._min_datetime + pd.Timedelta(seconds=pos * time_range) for pos in positions
        ]

        return pd.Series(timestamps)

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        """Normalize values to [0, 1] range."""
        min_val = np.nanmin(values)
        max_val = np.nanmax(values)
        if max_val == min_val:
            return np.zeros_like(values, dtype=float)
        return (values - min_val) / (max_val - min_val)

    def _cyclical_encode(self, values: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
        """Encode values as sin/cos for cyclical features."""
        sin_vals = np.sin(2 * np.pi * values / period)
        cos_vals = np.cos(2 * np.pi * values / period)
        return sin_vals, cos_vals

    def get_output_dim(self) -> int:
        """Get the output dimension."""
        dim = 0
        for feat in self.features:
            if feat in ["month", "day", "dayofweek", "hour"] and self.cyclical:
                dim += 2
            else:
                dim += 1
        return dim


class DataTransformer:
    """High-level transformer that handles entire DataFrames."""

    def __init__(
        self,
        n_modes: int = 10,
        max_categories: int = 100,
    ) -> None:
        """Initialize data transformer.

        Args:
            n_modes: Number of modes for numerical transformation
            max_categories: Maximum categories for categorical columns
        """
        self.n_modes = n_modes
        self.max_categories = max_categories

        self._transformers: Dict[str, BaseTransformer] = {}
        self._column_order: List[str] = []
        self._output_dims: Dict[str, int] = {}
        self._is_fitted = False

    def fit(
        self,
        data: pd.DataFrame,
        discrete_columns: Optional[List[str]] = None,
    ) -> "DataTransformer":
        """Fit transformers for all columns.

        Args:
            data: Input DataFrame
            discrete_columns: List of columns to treat as categorical

        Returns:
            Self for method chaining
        """
        discrete_columns = set(discrete_columns or [])
        self._column_order = list(data.columns)

        for col in data.columns:
            dtype = data[col].dtype

            if col in discrete_columns or dtype == "object" or dtype.name == "category":
                transformer = CategoricalTransformer(max_categories=self.max_categories)
            elif np.issubdtype(dtype, np.datetime64):
                transformer = DatetimeTransformer()
            elif np.issubdtype(dtype, np.number):
                transformer = NumericalTransformer(n_modes=self.n_modes)
            else:
                # Treat as categorical by default
                transformer = CategoricalTransformer(max_categories=self.max_categories)

            transformer.fit(data[col])
            self._transformers[col] = transformer
            self._output_dims[col] = transformer.get_output_dim()

        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transform all columns to numerical representation.

        Args:
            data: Input DataFrame

        Returns:
            Transformed numpy array
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted")

        transformed = []
        for col in self._column_order:
            if col in data.columns:
                col_data = self._transformers[col].transform(data[col])
                transformed.append(col_data)

        return np.hstack(transformed)

    def inverse_transform(self, data: np.ndarray) -> pd.DataFrame:
        """Inverse transform from numerical representation.

        Args:
            data: Transformed numpy array

        Returns:
            Reconstructed DataFrame
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted")

        result = {}
        offset = 0

        for col in self._column_order:
            dim = self._output_dims[col]
            col_data = data[:, offset : offset + dim]
            result[col] = self._transformers[col].inverse_transform(col_data)
            offset += dim

        return pd.DataFrame(result)

    def get_output_dimensions(self) -> int:
        """Get total output dimension."""
        return sum(self._output_dims.values())

    def get_column_dimensions(self) -> Dict[str, int]:
        """Get output dimensions for each column."""
        return self._output_dims.copy()
