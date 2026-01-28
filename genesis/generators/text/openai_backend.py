"""OpenAI backend for text generation."""

import os
from typing import List, Optional

from genesis.core.config import TextGenerationConfig
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class OpenAIBackend:
    """OpenAI API backend for text generation."""

    def __init__(
        self,
        config: Optional[TextGenerationConfig] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize OpenAI backend.

        Args:
            config: Text generation configuration
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
        """
        self.config = config or TextGenerationConfig()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        self._client = None

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except ImportError as e:
                raise ImportError(
                    "openai package required. Install with: pip install genesis-synth[llm]"
                ) from e

        return self._client

    def generate(
        self,
        prompt: str,
        n_samples: int = 1,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """Generate text using OpenAI API.

        Args:
            prompt: Prompt for generation
            n_samples: Number of samples to generate
            max_tokens: Maximum tokens per sample
            temperature: Sampling temperature

        Returns:
            List of generated texts
        """
        client = self._get_client()

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        results = []

        for _ in range(n_samples):
            try:
                response = client.chat.completions.create(
                    model=self.config.model_name or "gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that generates realistic synthetic data.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=1,
                )
                results.append(response.choices[0].message.content)
            except Exception as e:
                logger.warning(f"OpenAI generation failed: {e}")
                results.append("")

        return results

    def generate_similar(
        self,
        examples: List[str],
        n_samples: int = 1,
        context: Optional[str] = None,
    ) -> List[str]:
        """Generate text similar to provided examples.

        Args:
            examples: Example texts to base generation on
            n_samples: Number of samples to generate
            context: Additional context for generation

        Returns:
            List of generated texts
        """
        # Create prompt with examples
        example_text = "\n".join(f"- {ex[:200]}" for ex in examples[:5])

        prompt = f"""Generate {n_samples} new text samples that are similar in style and content to these examples, but completely new and different:

Examples:
{example_text}

{f'Context: {context}' if context else ''}

Generate {n_samples} new, unique samples (one per line):"""

        results = self.generate(prompt, n_samples=1, max_tokens=self.config.max_tokens * n_samples)

        # Parse results (split by lines)
        if results and results[0]:
            lines = [line.strip() for line in results[0].split("\n") if line.strip()]
            return lines[:n_samples]

        return []
