"""Diffusion-based image generation using Stable Diffusion or similar models.

This module provides image generation using diffusion models via:
- HuggingFace Diffusers
- OpenAI DALL-E API
- Local Stable Diffusion

Example:
    >>> from genesis.generators.image import DiffusionImageGenerator
    >>>
    >>> generator = DiffusionImageGenerator(provider="huggingface")
    >>> images = generator.generate("A photo of a cat wearing a hat")
    >>> images[0].save("cat_with_hat.png")
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from genesis.generators.image.base import (
    BaseImageGenerator,
    GeneratedImage,
    ImageConfig,
)
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class DiffusionImageGenerator(BaseImageGenerator):
    """Generate images using diffusion models.

    Supports multiple providers:
    - huggingface: Use local Stable Diffusion via diffusers
    - openai: Use DALL-E API
    - replicate: Use Replicate API (Stable Diffusion, SDXL, etc.)
    """

    def __init__(
        self,
        provider: str = "huggingface",
        model_id: str = "stabilityai/stable-diffusion-2-1",
        api_key: Optional[str] = None,
        config: Optional[ImageConfig] = None,
        device: str = "auto",
    ) -> None:
        """Initialize the diffusion generator.

        Args:
            provider: Provider to use ('huggingface', 'openai', 'replicate')
            model_id: Model identifier
            api_key: API key for external providers
            config: Image generation configuration
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        """
        super().__init__(config)

        self.provider = provider
        self.model_id = model_id
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.device = device

        self._pipeline = None
        self._client = None

    def initialize(self) -> None:
        """Initialize the model."""
        if self._is_initialized:
            return

        if self.provider == "huggingface":
            self._init_huggingface()
        elif self.provider == "openai":
            self._init_openai()
        elif self.provider == "replicate":
            self._init_replicate()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        self._is_initialized = True
        logger.info(f"Initialized {self.provider} image generator")

    def _init_huggingface(self) -> None:
        """Initialize HuggingFace diffusers pipeline."""
        try:
            import torch
            from diffusers import StableDiffusionPipeline
        except ImportError as e:
            raise ImportError(
                "diffusers and torch required. Install with: "
                "pip install diffusers torch transformers accelerate"
            ) from e

        # Determine device
        if self.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = self.device

        # Load pipeline
        self._pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        )
        self._pipeline = self._pipeline.to(device)

        # Enable memory optimizations
        if device == "cuda":
            try:
                self._pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                pass  # xformers not available

        self._device = device

    def _init_openai(self) -> None:
        """Initialize OpenAI DALL-E client."""
        try:
            import openai

            self._client = openai.OpenAI(api_key=self.api_key)
        except ImportError as e:
            raise ImportError("openai package required. Install with: pip install openai") from e

    def _init_replicate(self) -> None:
        """Initialize Replicate client."""
        try:
            import replicate

            self._client = replicate.Client(api_token=self.api_key)
        except ImportError as e:
            raise ImportError("replicate package required. Install with: pip install replicate") from e

    def generate(
        self,
        prompts: Union[str, List[str]],
        **kwargs,
    ) -> List[GeneratedImage]:
        """Generate images from prompts.

        Args:
            prompts: Text prompt(s)
            **kwargs: Additional parameters (overrides config)

        Returns:
            List of GeneratedImage objects
        """
        if not self._is_initialized:
            self.initialize()

        if isinstance(prompts, str):
            prompts = [prompts]

        if self.provider == "huggingface":
            return self._generate_huggingface(prompts, **kwargs)
        elif self.provider == "openai":
            return self._generate_openai(prompts, **kwargs)
        elif self.provider == "replicate":
            return self._generate_replicate(prompts, **kwargs)

        raise ValueError(f"Unknown provider: {self.provider}")

    def _generate_huggingface(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[GeneratedImage]:
        """Generate using HuggingFace diffusers."""
        import torch

        # Merge config with kwargs
        width = kwargs.get("width", self.config.width)
        height = kwargs.get("height", self.config.height)
        num_inference_steps = kwargs.get("num_inference_steps", self.config.num_inference_steps)
        guidance_scale = kwargs.get("guidance_scale", self.config.guidance_scale)
        negative_prompt = kwargs.get("negative_prompt", self.config.negative_prompt)
        seed = kwargs.get("seed", self.config.seed)

        results = []

        for prompt in prompts:
            # Set seed for reproducibility
            if seed is not None:
                generator = torch.Generator(device=self._device).manual_seed(seed)
            else:
                generator = None
                seed = np.random.randint(0, 2**32 - 1)

            # Generate
            output = self._pipeline(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                generator=generator,
            )

            # Convert to numpy
            image = output.images[0]
            image_np = np.array(image)

            results.append(
                GeneratedImage(
                    data=image_np,
                    prompt=prompt,
                    seed=seed,
                    metadata={
                        "provider": "huggingface",
                        "model_id": self.model_id,
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                    },
                )
            )

        return results

    def _generate_openai(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[GeneratedImage]:
        """Generate using OpenAI DALL-E."""
        from io import BytesIO

        import requests
        from PIL import Image

        # Determine size (DALL-E 3 supports specific sizes)
        width = kwargs.get("width", self.config.width)
        height = kwargs.get("height", self.config.height)

        # Map to supported sizes
        if width >= 1024 and height >= 1024:
            size = "1024x1024"
        elif width >= 1024:
            size = "1792x1024"
        else:
            size = "1024x1792"

        results = []

        for prompt in prompts:
            response = self._client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality="standard",
                n=1,
            )

            # Download image
            image_url = response.data[0].url
            img_response = requests.get(image_url)
            image = Image.open(BytesIO(img_response.content))
            image_np = np.array(image)

            results.append(
                GeneratedImage(
                    data=image_np,
                    prompt=prompt,
                    seed=0,  # DALL-E doesn't expose seed
                    metadata={
                        "provider": "openai",
                        "model": "dall-e-3",
                        "revised_prompt": response.data[0].revised_prompt,
                    },
                )
            )

        return results

    def _generate_replicate(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[GeneratedImage]:
        """Generate using Replicate API."""
        from io import BytesIO

        import requests
        from PIL import Image

        width = kwargs.get("width", self.config.width)
        height = kwargs.get("height", self.config.height)
        seed = kwargs.get("seed", self.config.seed) or np.random.randint(0, 2**32 - 1)

        results = []

        for prompt in prompts:
            output = self._client.run(
                self.model_id
                or "stability-ai/sdxl:c221b2b8ef527988fb59bf24a8b97c4561f1c671f73bd389f866bfb27c061316",
                input={
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "seed": seed,
                },
            )

            # Download image
            image_url = output[0] if isinstance(output, list) else output
            img_response = requests.get(image_url)
            image = Image.open(BytesIO(img_response.content))
            image_np = np.array(image)

            results.append(
                GeneratedImage(
                    data=image_np,
                    prompt=prompt,
                    seed=seed,
                    metadata={
                        "provider": "replicate",
                        "model_id": self.model_id,
                    },
                )
            )

        return results

    def generate_from_metadata(
        self,
        df_metadata: Any,
        prompt_column: str = "description",
        **kwargs,
    ) -> List[GeneratedImage]:
        """Generate images based on metadata from a DataFrame.

        Useful for generating images that correspond to tabular data.

        Args:
            df_metadata: DataFrame with metadata
            prompt_column: Column containing prompts or descriptions
            **kwargs: Additional generation parameters

        Returns:
            List of GeneratedImage objects
        """
        import pandas as pd

        if not isinstance(df_metadata, pd.DataFrame):
            raise ValueError("Expected pandas DataFrame")

        if prompt_column not in df_metadata.columns:
            raise ValueError(f"Column '{prompt_column}' not found")

        prompts = df_metadata[prompt_column].tolist()
        images = self.generate(prompts, **kwargs)

        # Add row index to metadata
        for i, img in enumerate(images):
            img.metadata["row_index"] = i
            for col in df_metadata.columns:
                if col != prompt_column:
                    img.metadata[col] = df_metadata.iloc[i][col]

        return images

    def generate_batch(
        self,
        prompts: List[str],
        batch_size: int = 4,
        show_progress: bool = True,
        **kwargs,
    ) -> List[GeneratedImage]:
        """Generate images in batches with progress tracking.

        Args:
            prompts: List of text prompts
            batch_size: Number of images to generate at once
            show_progress: Whether to show progress bar
            **kwargs: Additional generation parameters

        Returns:
            List of GeneratedImage objects
        """
        from tqdm import tqdm

        results = []
        n_batches = (len(prompts) + batch_size - 1) // batch_size

        iterator = range(0, len(prompts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches, desc="Generating images")

        for i in iterator:
            batch_prompts = prompts[i : i + batch_size]
            batch_results = self.generate(batch_prompts, **kwargs)
            results.extend(batch_results)

        return results


class TabularConditionedImageGenerator:
    """Generate images conditioned on tabular data.

    This generator creates images that correspond to rows in a tabular dataset,
    using column values to construct or modify prompts.

    Example:
        >>> from genesis.generators.image import TabularConditionedImageGenerator
        >>>
        >>> # Generate product images based on product catalog
        >>> generator = TabularConditionedImageGenerator(
        ...     prompt_template="A professional product photo of {name}, "
        ...                    "{color} color, {category} style",
        ... )
        >>> images = generator.generate(product_catalog_df)
    """

    def __init__(
        self,
        prompt_template: str,
        provider: str = "huggingface",
        model_id: str = "stabilityai/stable-diffusion-2-1",
        api_key: Optional[str] = None,
        config: Optional[ImageConfig] = None,
        negative_prompt_template: Optional[str] = None,
    ) -> None:
        """Initialize the tabular-conditioned image generator.

        Args:
            prompt_template: Template string with {column_name} placeholders
            provider: Image generation provider
            model_id: Model to use
            api_key: API key for external providers
            config: Image configuration
            negative_prompt_template: Optional negative prompt template
        """
        self.prompt_template = prompt_template
        self.negative_prompt_template = negative_prompt_template

        self._image_generator = DiffusionImageGenerator(
            provider=provider,
            model_id=model_id,
            api_key=api_key,
            config=config,
        )

    def _format_prompt(self, row: Dict[str, Any], template: str) -> str:
        """Format a prompt template with row values."""
        try:
            return template.format(**row)
        except KeyError as e:
            raise ValueError(f"Column {e} in template not found in data") from e

    def generate(
        self,
        data: Any,  # pd.DataFrame
        batch_size: int = 4,
        show_progress: bool = True,
        **kwargs,
    ) -> List[GeneratedImage]:
        """Generate images for each row in the DataFrame.

        Args:
            data: DataFrame with columns referenced in prompt_template
            batch_size: Batch size for generation
            show_progress: Whether to show progress
            **kwargs: Additional generation parameters

        Returns:
            List of GeneratedImage objects
        """
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Expected pandas DataFrame")

        # Build prompts from template
        prompts = []
        negative_prompts = []

        for _, row in data.iterrows():
            row_dict = row.to_dict()
            prompts.append(self._format_prompt(row_dict, self.prompt_template))

            if self.negative_prompt_template:
                negative_prompts.append(
                    self._format_prompt(row_dict, self.negative_prompt_template)
                )

        # Generate images
        if self.negative_prompt_template:
            # Generate one at a time to use per-image negative prompts
            results = []
            from tqdm import tqdm

            iterator = enumerate(zip(prompts, negative_prompts))
            if show_progress:
                iterator = tqdm(iterator, total=len(prompts), desc="Generating images")

            for i, (prompt, neg_prompt) in iterator:
                images = self._image_generator.generate(
                    prompt, negative_prompt=neg_prompt, **kwargs
                )
                # Add row metadata
                for img in images:
                    img.metadata["row_index"] = i
                    for col in data.columns:
                        img.metadata[col] = data.iloc[i][col]
                results.extend(images)
        else:
            results = self._image_generator.generate_batch(
                prompts,
                batch_size=batch_size,
                show_progress=show_progress,
                **kwargs,
            )
            # Add row metadata
            for i, img in enumerate(results):
                img.metadata["row_index"] = i
                for col in data.columns:
                    img.metadata[col] = data.iloc[i][col]

        return results

    def generate_variations(
        self,
        row: Dict[str, Any],
        n_variations: int = 5,
        temperature_range: tuple = (0.7, 1.3),
        **kwargs,
    ) -> List[GeneratedImage]:
        """Generate multiple variations for a single row.

        Args:
            row: Dictionary of column values
            n_variations: Number of variations to generate
            temperature_range: Range of guidance scale multipliers
            **kwargs: Additional generation parameters

        Returns:
            List of varied GeneratedImage objects
        """
        import numpy as np

        prompt = self._format_prompt(row, self.prompt_template)
        negative_prompt = None
        if self.negative_prompt_template:
            negative_prompt = self._format_prompt(row, self.negative_prompt_template)

        # Vary guidance scale for diversity
        base_guidance = kwargs.pop("guidance_scale", 7.5)
        variations = np.linspace(
            base_guidance * temperature_range[0],
            base_guidance * temperature_range[1],
            n_variations,
        )

        results = []
        for i, guidance in enumerate(variations):
            images = self._image_generator.generate(
                prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance,
                seed=kwargs.get("seed", 0) + i,
                **kwargs,
            )
            for img in images:
                img.metadata["variation"] = i
                img.metadata["guidance_scale"] = guidance
                img.metadata.update(row)
            results.extend(images)

        return results


class SyntheticImageDataset:
    """Generate a complete synthetic image dataset from tabular data.

    Creates paired (image, metadata) datasets useful for training
    vision models with specific attributes.
    """

    def __init__(
        self,
        image_generator: TabularConditionedImageGenerator,
        output_dir: str = "./synthetic_images",
    ) -> None:
        """Initialize the dataset generator.

        Args:
            image_generator: Configured TabularConditionedImageGenerator
            output_dir: Directory to save generated images
        """
        self.image_generator = image_generator
        self.output_dir = Path(output_dir)

    def generate_dataset(
        self,
        data: Any,  # pd.DataFrame
        images_per_row: int = 1,
        format: str = "png",
        save_metadata: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a complete image dataset.

        Args:
            data: Source DataFrame
            images_per_row: Number of images to generate per row
            format: Image format (png, jpg)
            save_metadata: Whether to save metadata JSON
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with dataset statistics
        """
        import json

        import pandas as pd

        self.output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = self.output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        metadata_records = []

        for idx, (_, row) in enumerate(data.iterrows()):
            row_dict = row.to_dict()

            for var in range(images_per_row):
                if images_per_row > 1:
                    images = self.image_generator.generate_variations(
                        row_dict,
                        n_variations=1,
                        seed=idx * 1000 + var,
                        **kwargs,
                    )
                else:
                    images = self.image_generator.generate(
                        pd.DataFrame([row_dict]),
                        show_progress=False,
                        **kwargs,
                    )

                for _i, img in enumerate(images):
                    # Save image
                    filename = f"{idx:06d}_{var:02d}.{format}"
                    filepath = images_dir / filename
                    img.save(str(filepath))

                    # Record metadata
                    record = {
                        "image_path": str(filepath.relative_to(self.output_dir)),
                        "row_index": idx,
                        "variation": var,
                        **row_dict,
                    }
                    metadata_records.append(record)

        # Save metadata
        if save_metadata:
            metadata_df = pd.DataFrame(metadata_records)
            metadata_df.to_csv(self.output_dir / "metadata.csv", index=False)

            with open(self.output_dir / "metadata.json", "w") as f:
                json.dump(metadata_records, f, indent=2, default=str)

        stats = {
            "total_images": len(metadata_records),
            "source_rows": len(data),
            "images_per_row": images_per_row,
            "output_dir": str(self.output_dir),
        }

        logger.info(f"Generated {stats['total_images']} images to {self.output_dir}")

        return stats
