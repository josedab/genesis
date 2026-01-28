"""Base class for text generators."""

from typing import List, Optional

import pandas as pd

from genesis.core.base import BaseGenerator
from genesis.core.config import GeneratorConfig, PrivacyConfig, TextGenerationConfig


class BaseTextGenerator(BaseGenerator):
    """Base class for text generators."""

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        privacy: Optional[PrivacyConfig] = None,
        text_config: Optional[TextGenerationConfig] = None,
    ) -> None:
        super().__init__(config, privacy)
        self.text_config = text_config or TextGenerationConfig()

        self._text_column: Optional[str] = None
        self._sample_texts: List[str] = []

    def _prepare_texts(
        self,
        data: pd.DataFrame,
        text_column: Optional[str] = None,
    ) -> List[str]:
        """Extract text samples from DataFrame.

        Args:
            data: DataFrame containing text data
            text_column: Name of column containing text

        Returns:
            List of text samples
        """
        # Auto-detect text column if not specified
        if text_column is None:
            object_cols = data.select_dtypes(include=["object"]).columns
            if len(object_cols) == 0:
                raise ValueError("No text columns found in data")
            text_column = object_cols[0]

        self._text_column = text_column
        texts = data[text_column].dropna().astype(str).tolist()

        return texts

    def _filter_pii(self, text: str) -> str:
        """Filter potential PII from generated text.

        Args:
            text: Text to filter

        Returns:
            Filtered text
        """
        import re

        # Simple PII patterns
        patterns = [
            (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]"),  # Phone numbers
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),  # Emails
            (r"\b\d{3}[-]?\d{2}[-]?\d{4}\b", "[SSN]"),  # SSN
            (r"\b\d{16}\b", "[CARD]"),  # Credit card
        ]

        filtered = text
        for pattern, replacement in patterns:
            filtered = re.sub(pattern, replacement, filtered)

        return filtered
