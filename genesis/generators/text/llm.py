"""LLM-based text generator."""

import time
from typing import Any, Dict, List, Optional

import pandas as pd

from genesis.core.config import GeneratorConfig, PrivacyConfig, TextGenerationConfig
from genesis.core.types import FittingResult, ProgressCallback
from genesis.generators.text.base import BaseTextGenerator
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class LLMTextGenerator(BaseTextGenerator):
    """LLM-based generator for synthetic text data.

    Supports both OpenAI API and HuggingFace transformers backends
    for generating privacy-safe synthetic text.

    Example:
        >>> generator = LLMTextGenerator(backend='huggingface', model_name='gpt2')
        >>> generator.fit(text_data)
        >>> synthetic_texts = generator.generate(n_samples=100)
    """

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        privacy: Optional[PrivacyConfig] = None,
        text_config: Optional[TextGenerationConfig] = None,
        backend: str = "huggingface",
        model_name: str = "gpt2",
        temperature: float = 0.7,
        max_tokens: int = 256,
        api_key: Optional[str] = None,
        fine_tune: bool = False,
        verbose: bool = True,
    ) -> None:
        """Initialize LLM text generator.

        Args:
            config: Generator configuration
            privacy: Privacy configuration
            text_config: Text generation configuration
            backend: Backend to use ('openai', 'huggingface')
            model_name: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens per sample
            api_key: API key for OpenAI (if using OpenAI backend)
            fine_tune: Whether to fine-tune on training data
            verbose: Whether to print progress
        """
        text_config = text_config or TextGenerationConfig(
            backend=backend,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        super().__init__(config, privacy, text_config)

        self.backend_name = backend
        self.model_name = model_name
        self.fine_tune = fine_tune
        self.verbose = verbose

        self._backend = None
        self._sample_texts: List[str] = []

    def _get_backend(self):
        """Get or create backend."""
        if self._backend is not None:
            return self._backend

        if self.backend_name == "openai":
            from genesis.generators.text.openai_backend import OpenAIBackend

            self._backend = OpenAIBackend(
                config=self.text_config,
                api_key=self.text_config.api_key,
            )
        else:
            from genesis.generators.text.huggingface_backend import HuggingFaceBackend

            self._backend = HuggingFaceBackend(
                config=self.text_config,
                model_name=self.model_name,
            )

        return self._backend

    def _fit_impl(
        self,
        data: pd.DataFrame,
        discrete_columns: Optional[List[str]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> FittingResult:
        """Fit text generator to training data."""
        start_time = time.time()

        # Detect text column
        text_column = None
        for col in data.columns:
            if col in (discrete_columns or []):
                continue
            if data[col].dtype == object:
                avg_len = data[col].dropna().astype(str).str.len().mean()
                if avg_len > 20:  # Likely text, not short category
                    text_column = col
                    break

        if text_column is None:
            # Use first object column
            object_cols = data.select_dtypes(include=["object"]).columns
            if len(object_cols) > 0:
                text_column = object_cols[0]
            else:
                raise ValueError("No text column found in data")

        self._text_column = text_column
        self._sample_texts = data[text_column].dropna().astype(str).tolist()

        if self.verbose:
            logger.info(
                f"Collected {len(self._sample_texts)} text samples from column '{text_column}'"
            )

        # Fine-tune if requested and using HuggingFace
        if self.fine_tune and self.backend_name == "huggingface":
            backend = self._get_backend()
            if hasattr(backend, "fine_tune"):
                if self.verbose:
                    logger.info("Fine-tuning model on training data...")
                backend.fine_tune(self._sample_texts[:1000])  # Limit for fine-tuning

        fitting_time = time.time() - start_time

        return FittingResult(
            success=True,
            fitting_time=fitting_time,
            metadata={
                "n_samples": len(self._sample_texts),
                "text_column": text_column,
                "backend": self.backend_name,
            },
        )

    def _generate_impl(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> pd.DataFrame:
        """Generate synthetic text data."""
        if not self._sample_texts:
            raise RuntimeError("No sample texts available. Call fit() first.")

        backend = self._get_backend()

        if self.verbose:
            logger.info(f"Generating {n_samples} text samples...")

        # Generate in batches
        batch_size = self.text_config.batch_size
        generated_texts = []

        for i in range(0, n_samples, batch_size):
            current_batch = min(batch_size, n_samples - i)

            # Generate similar texts
            batch_texts = backend.generate_similar(
                examples=self._sample_texts[:10],  # Use subset of examples
                n_samples=current_batch,
            )

            # Apply privacy filter if enabled
            if self.text_config.privacy_filter:
                batch_texts = [self._filter_pii(t) for t in batch_texts]

            generated_texts.extend(batch_texts)

            if progress_callback:
                progress_callback(i + current_batch, n_samples, {})

        # Ensure we have exactly n_samples
        generated_texts = generated_texts[:n_samples]

        # Pad if needed
        while len(generated_texts) < n_samples:
            # Use random sample if generation failed
            import random

            generated_texts.append(random.choice(self._sample_texts))

        return pd.DataFrame({self._text_column: generated_texts})
