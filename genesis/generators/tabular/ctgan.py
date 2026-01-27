"""CTGAN (Conditional Tabular GAN) implementation."""

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from genesis.backends import get_device, select_backend
from genesis.core.config import GeneratorConfig, PrivacyConfig
from genesis.core.types import ColumnList, FittingResult, ProgressCallback
from genesis.generators.tabular.base import BaseTabularGenerator
from genesis.utils.logging import TrainingLogger, get_logger

logger = get_logger(__name__)


class CTGANGenerator(BaseTabularGenerator):
    """CTGAN (Conditional Tabular GAN) for synthetic tabular data generation.

    CTGAN uses mode-specific normalization to handle multi-modal distributions
    and training-by-sampling to address class imbalance in categorical columns.

    Reference:
        Xu et al., "Modeling Tabular data using Conditional GAN" (2019)

    Example:
        >>> generator = CTGANGenerator(epochs=300, batch_size=500)
        >>> generator.fit(real_data, discrete_columns=['gender', 'city'])
        >>> synthetic_data = generator.generate(n_samples=1000)
    """

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        privacy: Optional[PrivacyConfig] = None,
        epochs: int = 300,
        batch_size: int = 500,
        generator_dim: Tuple[int, ...] = (256, 256),
        discriminator_dim: Tuple[int, ...] = (256, 256),
        embedding_dim: int = 128,
        learning_rate: float = 2e-4,
        pac: int = 10,
        n_critic: int = 1,
        lambda_gp: float = 10.0,
        device: str = "auto",
        verbose: bool = True,
    ) -> None:
        """Initialize CTGAN generator.

        Args:
            config: Generator configuration (overrides other args if provided)
            privacy: Privacy configuration
            epochs: Number of training epochs
            batch_size: Batch size for training
            generator_dim: Dimensions of generator hidden layers
            discriminator_dim: Dimensions of discriminator hidden layers
            embedding_dim: Dimension of embedding/noise vector
            learning_rate: Learning rate for Adam optimizer
            pac: Number of samples to pack together (PacGAN)
            n_critic: Number of discriminator updates per generator update
            lambda_gp: Gradient penalty coefficient
            device: Device to use ('auto', 'cpu', 'cuda')
            verbose: Whether to print training progress
        """
        super().__init__(config, privacy)

        # Override config with explicit arguments
        self.epochs = epochs if config is None else config.epochs
        self.batch_size = batch_size if config is None else config.batch_size
        self.generator_dim = generator_dim if config is None else config.generator_dim
        self.discriminator_dim = discriminator_dim if config is None else config.discriminator_dim
        self.embedding_dim = embedding_dim if config is None else config.embedding_dim
        self.learning_rate = learning_rate if config is None else config.learning_rate
        self.pac = pac if config is None else config.pac
        self.n_critic = n_critic if config is None else config.n_critics
        self.lambda_gp = lambda_gp
        self.device = device if config is None else config.device
        self.verbose = verbose if config is None else config.verbose

        # Will be set during training
        self._generator = None
        self._discriminator = None
        self._backend = None

    def _fit_impl(
        self,
        data: pd.DataFrame,
        discrete_columns: Optional[ColumnList] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> FittingResult:
        """Fit CTGAN to the training data."""
        start_time = time.time()

        # Select backend
        self._backend = select_backend(self.config.backend)
        device = get_device(self.device)

        if self.verbose:
            logger.info(f"Training CTGAN with {self._backend} backend on {device}")

        # Transform data
        transformed_data = self._fit_transformer(data, discrete_columns)
        data_dim = transformed_data.shape[1]

        if self.verbose:
            logger.info(f"Data dimension: {data_dim}, Samples: {len(transformed_data)}")

        # Train based on backend
        if self._backend == "pytorch":
            final_loss = self._train_pytorch(transformed_data, data_dim, device, progress_callback)
        else:
            final_loss = self._train_tensorflow(transformed_data, data_dim, progress_callback)

        fitting_time = time.time() - start_time

        if self.verbose:
            logger.info(f"Training completed in {fitting_time:.2f}s")

        return FittingResult(
            success=True,
            fitting_time=fitting_time,
            n_epochs=self.epochs,
            final_loss=final_loss,
            metadata={
                "backend": self._backend,
                "data_dim": data_dim,
                "n_samples": len(transformed_data),
            },
        )

    def _train_pytorch(
        self,
        data: np.ndarray,
        data_dim: int,
        device: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> float:
        """Train using PyTorch backend."""
        import torch.optim as optim

        from genesis.backends.pytorch import (
            Discriminator,
            GANTrainer,
            Generator,
            create_dataloader,
            set_seed,
        )

        if self.config.random_seed is not None:
            set_seed(self.config.random_seed)

        # Create models
        self._generator = Generator(
            embedding_dim=self.embedding_dim,
            generator_dim=self.generator_dim,
            data_dim=data_dim,
        ).to(device)

        self._discriminator = Discriminator(
            input_dim=data_dim,
            discriminator_dim=self.discriminator_dim,
            pac=self.pac,
        ).to(device)

        # Create optimizers
        g_optimizer = optim.Adam(
            self._generator.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.9),
        )
        d_optimizer = optim.Adam(
            self._discriminator.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.9),
        )

        # Create dataloader
        dataloader = create_dataloader(data, self.batch_size, shuffle=True)

        # Create trainer
        trainer = GANTrainer(
            generator=self._generator,
            discriminator=self._discriminator,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            device=device,
            n_critic=self.n_critic,
            lambda_gp=self.lambda_gp,
        )

        # Training loop
        training_logger = TrainingLogger(self.epochs, logger, log_every_n_epochs=10)

        for epoch in range(self.epochs):
            g_loss, d_loss = trainer.train_epoch(dataloader, self.embedding_dim)

            training_logger.log_epoch(epoch, {"g_loss": g_loss, "d_loss": d_loss})

            if progress_callback:
                progress_callback(epoch, self.epochs, {"g_loss": g_loss, "d_loss": d_loss})

        return trainer.history["g_loss"][-1] if trainer.history["g_loss"] else 0.0

    def _train_tensorflow(
        self,
        data: np.ndarray,
        data_dim: int,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> float:
        """Train using TensorFlow backend."""
        from tensorflow import keras

        from genesis.backends.tensorflow import (
            GANTrainer,
            create_dataset,
            create_discriminator,
            create_generator,
            set_seed,
        )

        if self.config.random_seed is not None:
            set_seed(self.config.random_seed)

        # Create models
        self._generator = create_generator(
            embedding_dim=self.embedding_dim,
            generator_dim=self.generator_dim,
            data_dim=data_dim,
        )

        self._discriminator = create_discriminator(
            input_dim=data_dim,
            discriminator_dim=self.discriminator_dim,
            pac=self.pac,
        )

        # Create optimizers
        g_optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.5,
            beta_2=0.9,
        )
        d_optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.5,
            beta_2=0.9,
        )

        # Create dataset
        dataset = create_dataset(data, self.batch_size, shuffle=True)

        # Create trainer
        trainer = GANTrainer(
            generator=self._generator,
            discriminator=self._discriminator,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            n_critic=self.n_critic,
            lambda_gp=self.lambda_gp,
        )

        # Training loop
        training_logger = TrainingLogger(self.epochs, logger, log_every_n_epochs=10)

        for epoch in range(self.epochs):
            g_loss, d_loss = trainer.train_epoch(dataset, self.embedding_dim)

            training_logger.log_epoch(epoch, {"g_loss": g_loss, "d_loss": d_loss})

            if progress_callback:
                progress_callback(epoch, self.epochs, {"g_loss": g_loss, "d_loss": d_loss})

        return trainer.history["g_loss"][-1] if trainer.history["g_loss"] else 0.0

    def _generate_impl(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> pd.DataFrame:
        """Generate synthetic data using trained CTGAN."""
        if self._generator is None:
            raise RuntimeError("Generator not trained")

        if self._backend == "pytorch":
            return self._generate_pytorch(n_samples, conditions)
        else:
            return self._generate_tensorflow(n_samples, conditions)

    def _generate_pytorch(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Generate using PyTorch backend."""
        import torch

        self._generator.eval()
        device = next(self._generator.parameters()).device

        generated_data = []
        batch_size = min(self.batch_size, n_samples)
        n_batches = (n_samples + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in range(n_batches):
                current_batch = min(batch_size, n_samples - i * batch_size)
                noise = torch.randn(current_batch, self.embedding_dim, device=device)

                fake_data = self._generator(noise)
                generated_data.append(fake_data.cpu().numpy())

        # Concatenate and process
        raw_output = np.vstack(generated_data)[:n_samples]

        # Apply activations and sampling
        activated = self._apply_activation(raw_output)
        sampled = self._sample_from_output(activated)

        # Inverse transform
        return self._inverse_transform(sampled)

    def _generate_tensorflow(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Generate using TensorFlow backend."""
        import tensorflow as tf

        generated_data = []
        batch_size = min(self.batch_size, n_samples)
        n_batches = (n_samples + batch_size - 1) // batch_size

        for i in range(n_batches):
            current_batch = min(batch_size, n_samples - i * batch_size)
            noise = tf.random.normal([current_batch, self.embedding_dim])

            fake_data = self._generator(noise, training=False)
            generated_data.append(fake_data.numpy())

        # Concatenate and process
        raw_output = np.vstack(generated_data)[:n_samples]

        # Apply activations and sampling
        activated = self._apply_activation(raw_output)
        sampled = self._sample_from_output(activated)

        # Inverse transform
        return self._inverse_transform(sampled)
