"""Base class for image generators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from genesis.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ImageConfig:
    """Configuration for image generation."""

    width: int = 512
    height: int = 512
    channels: int = 3
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    batch_size: int = 1
    output_format: str = "png"  # png, jpg, numpy


@dataclass
class GeneratedImage:
    """Container for a generated image."""

    data: np.ndarray  # HWC format, uint8
    prompt: str
    seed: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def channels(self) -> int:
        return self.data.shape[2] if len(self.data.shape) > 2 else 1

    def save(self, path: Union[str, Path]) -> None:
        """Save image to file."""
        try:
            from PIL import Image

            img = Image.fromarray(self.data)
            img.save(path)
            logger.info(f"Saved image to {path}")
        except ImportError as e:
            raise ImportError("Pillow required. Install with: pip install Pillow") from e

    def to_pil(self) -> Any:
        """Convert to PIL Image."""
        from PIL import Image

        return Image.fromarray(self.data)


class BaseImageGenerator(ABC):
    """Abstract base class for image generators.

    All image generators should inherit from this class and implement
    the generate() method.
    """

    def __init__(
        self,
        config: Optional[ImageConfig] = None,
    ) -> None:
        """Initialize the generator.

        Args:
            config: Image generation configuration
        """
        self.config = config or ImageConfig()
        self._is_initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the model (lazy loading)."""
        pass

    @abstractmethod
    def generate(
        self,
        prompts: Union[str, List[str]],
        **kwargs,
    ) -> List[GeneratedImage]:
        """Generate images from prompts.

        Args:
            prompts: Text prompt(s) describing desired images
            **kwargs: Additional generation parameters

        Returns:
            List of GeneratedImage objects
        """
        pass

    def generate_batch(
        self,
        prompts: List[str],
        n_per_prompt: int = 1,
        **kwargs,
    ) -> List[GeneratedImage]:
        """Generate multiple images per prompt.

        Args:
            prompts: List of prompts
            n_per_prompt: Number of images per prompt
            **kwargs: Additional parameters

        Returns:
            List of GeneratedImage objects
        """
        results = []
        for prompt in prompts:
            for _ in range(n_per_prompt):
                images = self.generate(prompt, **kwargs)
                results.extend(images)
        return results

    def generate_variations(
        self,
        base_prompt: str,
        variations: List[str],
        **kwargs,
    ) -> List[GeneratedImage]:
        """Generate variations of a base prompt.

        Args:
            base_prompt: Base prompt template with {variation} placeholder
            variations: List of variation strings
            **kwargs: Additional parameters

        Returns:
            List of GeneratedImage objects
        """
        prompts = [base_prompt.format(variation=v) for v in variations]
        return self.generate_batch(prompts, **kwargs)

    def save_batch(
        self,
        images: List[GeneratedImage],
        output_dir: Union[str, Path],
        prefix: str = "synthetic_",
    ) -> List[Path]:
        """Save a batch of images to directory.

        Args:
            images: List of GeneratedImage objects
            output_dir: Output directory
            prefix: Filename prefix

        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for i, img in enumerate(images):
            filename = f"{prefix}{i:04d}.{self.config.output_format}"
            path = output_dir / filename
            img.save(path)
            saved_paths.append(path)

        return saved_paths
