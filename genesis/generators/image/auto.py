"""Auto-selection for image generators."""

from typing import Optional

from genesis.generators.image.base import BaseImageGenerator, ImageConfig
from genesis.generators.image.diffusion import DiffusionImageGenerator
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


def create_image_generator(
    provider: str = "auto",
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
    config: Optional[ImageConfig] = None,
    **kwargs,
) -> BaseImageGenerator:
    """Create an image generator with automatic provider selection.

    Args:
        provider: Provider to use ('auto', 'huggingface', 'openai', 'replicate')
        model_id: Model identifier
        api_key: API key for external providers
        config: Image generation configuration
        **kwargs: Additional arguments passed to generator

    Returns:
        Configured image generator

    Example:
        >>> generator = create_image_generator(provider="auto")
        >>> images = generator.generate("A beautiful sunset over mountains")
    """
    import os

    if provider == "auto":
        # Auto-select based on available credentials/packages
        if os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
            logger.info("Auto-selected OpenAI DALL-E (API key found)")
        elif os.environ.get("REPLICATE_API_TOKEN"):
            provider = "replicate"
            logger.info("Auto-selected Replicate (API token found)")
        else:
            # Try HuggingFace (local)
            try:
                import diffusers  # noqa: F401
                import torch  # noqa: F401

                provider = "huggingface"
                logger.info("Auto-selected HuggingFace Diffusers (local)")
            except ImportError as e:
                raise ImportError(
                    "No image generation provider available. Either:\n"
                    "1. Set OPENAI_API_KEY for DALL-E\n"
                    "2. Set REPLICATE_API_TOKEN for Replicate\n"
                    "3. Install diffusers: pip install diffusers torch"
                ) from e

    # Set default model IDs based on provider
    if model_id is None:
        if provider == "huggingface":
            model_id = "stabilityai/stable-diffusion-2-1"
        elif provider == "openai":
            model_id = "dall-e-3"
        elif provider == "replicate":
            model_id = (
                "stability-ai/sdxl:c221b2b8ef527988fb59bf24a8b97c4561f1c671f73bd389f866bfb27c061316"
            )

    return DiffusionImageGenerator(
        provider=provider,
        model_id=model_id,
        api_key=api_key,
        config=config,
        **kwargs,
    )


def generate_images(
    prompts: list,
    n_per_prompt: int = 1,
    provider: str = "auto",
    output_dir: Optional[str] = None,
    **kwargs,
) -> list:
    """Convenience function for quick image generation.

    Args:
        prompts: List of text prompts
        n_per_prompt: Number of images per prompt
        provider: Provider to use
        output_dir: Optional directory to save images
        **kwargs: Additional generation parameters

    Returns:
        List of GeneratedImage objects
    """
    generator = create_image_generator(provider=provider, **kwargs)
    images = generator.generate_batch(prompts, n_per_prompt=n_per_prompt)

    if output_dir:
        generator.save_batch(images, output_dir)

    return images
