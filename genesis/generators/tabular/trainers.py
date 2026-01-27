"""Base training infrastructure with Template Method pattern.

This module provides a unified training framework for deep learning
generators (CTGAN, TVAE) that abstracts the backend-specific code
into a common structure.

The Template Method pattern allows subclasses to override specific
steps while keeping the overall algorithm structure consistent.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from genesis.backends import get_device, select_backend
from genesis.core.types import FittingResult, ProgressCallback
from genesis.utils.logging import TrainingLogger, get_logger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for the training process."""

    epochs: int = 300
    batch_size: int = 500
    learning_rate: float = 2e-4
    embedding_dim: int = 128
    device: str = "auto"
    verbose: bool = True
    random_seed: Optional[int] = None
    log_every_n_epochs: int = 10


@dataclass
class TrainingState:
    """State tracked during training."""

    epoch: int = 0
    losses: Dict[str, List[float]] = field(default_factory=dict)
    best_loss: float = float("inf")
    models: Dict[str, Any] = field(default_factory=dict)
    optimizers: Dict[str, Any] = field(default_factory=dict)


class BaseDeepLearningTrainer(ABC):
    """Abstract base class for deep learning trainers using Template Method pattern.

    This class defines the skeleton of the training algorithm, with
    hooks that subclasses can override for model-specific behavior.

    Template Method: train()
        1. setup_backend() - Select PyTorch/TensorFlow
        2. create_models() - Create neural network models
        3. create_optimizers() - Create optimizers
        4. create_dataloader() - Prepare data for training
        5. training_loop() - Main training iterations
            - train_step() - Single training step (abstract)
        6. finalize() - Post-training cleanup
    """

    def __init__(self, config: TrainingConfig) -> None:
        """Initialize the trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.state = TrainingState()
        self._backend: Optional[str] = None
        self._device: Optional[str] = None
        self._dataloader: Optional[Any] = None
        self._training_logger: Optional[TrainingLogger] = None

    def train(
        self,
        data: np.ndarray,
        data_dim: int,
        backend_type: str = "auto",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> FittingResult:
        """Execute the training algorithm (Template Method).

        This method defines the overall training structure.
        Subclasses customize behavior by overriding hook methods.

        Args:
            data: Transformed training data
            data_dim: Dimension of the data
            backend_type: Backend to use ('auto', 'pytorch', 'tensorflow')
            progress_callback: Optional callback for progress updates

        Returns:
            FittingResult with training details
        """
        start_time = time.time()

        # Step 1: Setup backend
        self._backend = select_backend(backend_type)
        self._device = get_device(self.config.device)

        if self.config.verbose:
            logger.info(
                f"Training {self.__class__.__name__} with {self._backend} "
                f"backend on {self._device}"
            )
            logger.info(f"Data dimension: {data_dim}, Samples: {len(data)}")

        # Set random seed if specified
        if self.config.random_seed is not None:
            self._set_seed(self.config.random_seed)

        # Step 2: Create models
        self.state.models = self.create_models(data_dim)

        # Step 3: Create optimizers
        self.state.optimizers = self.create_optimizers()

        # Step 4: Create dataloader
        self._dataloader = self.create_dataloader(data)

        # Step 5: Training loop
        self._training_logger = TrainingLogger(
            self.config.epochs, logger, log_every_n_epochs=self.config.log_every_n_epochs
        )

        final_loss = self._training_loop(progress_callback)

        # Step 6: Finalize
        self.finalize()

        fitting_time = time.time() - start_time

        if self.config.verbose:
            logger.info(f"Training completed in {fitting_time:.2f}s")

        return FittingResult(
            success=True,
            fitting_time=fitting_time,
            n_epochs=self.config.epochs,
            final_loss=final_loss,
            metadata={
                "backend": self._backend,
                "data_dim": data_dim,
                "n_samples": len(data),
            },
        )

    def _training_loop(
        self,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> float:
        """Execute the main training loop.

        Args:
            progress_callback: Optional progress callback

        Returns:
            Final loss value
        """
        for epoch in range(self.config.epochs):
            self.state.epoch = epoch

            # Execute training step (backend-specific)
            losses = self.train_step(self._dataloader)

            # Track losses
            for name, value in losses.items():
                if name not in self.state.losses:
                    self.state.losses[name] = []
                self.state.losses[name].append(value)

            # Log progress
            if self._training_logger:
                self._training_logger.log_epoch(epoch, losses)

            # Callback
            if progress_callback:
                progress_callback(epoch, self.config.epochs, losses)

        # Return primary loss
        primary_loss_key = self.get_primary_loss_key()
        if primary_loss_key in self.state.losses and self.state.losses[primary_loss_key]:
            return self.state.losses[primary_loss_key][-1]
        return 0.0

    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        if self._backend == "pytorch":
            from genesis.backends.pytorch import set_seed

            set_seed(seed)
        else:
            from genesis.backends.tensorflow import set_seed

            set_seed(seed)

    @abstractmethod
    def create_models(self, data_dim: int) -> Dict[str, Any]:
        """Create neural network models.

        Args:
            data_dim: Dimension of the input data

        Returns:
            Dictionary of model name -> model instance
        """
        pass

    @abstractmethod
    def create_optimizers(self) -> Dict[str, Any]:
        """Create optimizers for the models.

        Returns:
            Dictionary of optimizer name -> optimizer instance
        """
        pass

    @abstractmethod
    def create_dataloader(self, data: np.ndarray) -> Any:
        """Create a dataloader for the training data.

        Args:
            data: Training data array

        Returns:
            Dataloader or dataset object
        """
        pass

    @abstractmethod
    def train_step(self, dataloader: Any) -> Dict[str, float]:
        """Execute a single training step/epoch.

        Args:
            dataloader: Data loader or dataset

        Returns:
            Dictionary of loss name -> loss value
        """
        pass

    def get_primary_loss_key(self) -> str:
        """Get the key for the primary loss metric.

        Returns:
            Loss key string (default: 'loss')
        """
        return "loss"

    def finalize(self) -> None:
        """Post-training cleanup and finalization.

        Override in subclasses if needed.
        """
        pass

    @property
    def backend(self) -> Optional[str]:
        """Get the current backend."""
        return self._backend

    @property
    def device(self) -> Optional[str]:
        """Get the current device."""
        return self._device

    @property
    def models(self) -> Dict[str, Any]:
        """Get the trained models."""
        return self.state.models

    @property
    def history(self) -> Dict[str, List[float]]:
        """Get the training history."""
        return self.state.losses


class GANTrainerMixin:
    """Mixin providing GAN-specific training utilities."""

    n_critic: int = 1
    lambda_gp: float = 10.0

    def compute_gradient_penalty(
        self,
        discriminator: Any,
        real_data: Any,
        fake_data: Any,
    ) -> Any:
        """Compute gradient penalty for WGAN-GP.

        Args:
            discriminator: Discriminator model
            real_data: Real data batch
            fake_data: Generated data batch

        Returns:
            Gradient penalty loss
        """
        if self._backend == "pytorch":
            return self._compute_gp_pytorch(discriminator, real_data, fake_data)
        else:
            return self._compute_gp_tensorflow(discriminator, real_data, fake_data)

    def _compute_gp_pytorch(
        self,
        discriminator: Any,
        real_data: Any,
        fake_data: Any,
    ) -> Any:
        """PyTorch gradient penalty computation."""
        import torch

        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, device=real_data.device)
        alpha = alpha.expand_as(real_data)

        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)

        d_interpolated = discriminator(interpolated)

        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()

        return gradient_penalty

    def _compute_gp_tensorflow(
        self,
        discriminator: Any,
        real_data: Any,
        fake_data: Any,
    ) -> Any:
        """TensorFlow gradient penalty computation."""
        import tensorflow as tf

        batch_size = tf.shape(real_data)[0]
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)

        interpolated = alpha * real_data + (1 - alpha) * fake_data

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            d_interpolated = discriminator(interpolated, training=True)

        gradients = tape.gradient(d_interpolated, interpolated)
        gradients = tf.reshape(gradients, [batch_size, -1])
        gradient_norm = tf.norm(gradients, axis=1)
        gradient_penalty = tf.reduce_mean((gradient_norm - 1.0) ** 2)

        return gradient_penalty


class VAETrainerMixin:
    """Mixin providing VAE-specific training utilities."""

    beta: float = 1.0

    def compute_kl_divergence(
        self,
        mean: Any,
        logvar: Any,
    ) -> Any:
        """Compute KL divergence for VAE.

        Args:
            mean: Mean of the latent distribution
            logvar: Log variance of the latent distribution

        Returns:
            KL divergence loss
        """
        if self._backend == "pytorch":
            import torch

            return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        else:
            import tensorflow as tf

            return -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar))

    def reparameterize(
        self,
        mean: Any,
        logvar: Any,
    ) -> Any:
        """Reparameterization trick for VAE.

        Args:
            mean: Mean of the latent distribution
            logvar: Log variance of the latent distribution

        Returns:
            Sampled latent vector
        """
        if self._backend == "pytorch":
            import torch

            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            import tensorflow as tf

            std = tf.exp(0.5 * logvar)
            eps = tf.random.normal(tf.shape(std))
            return mean + eps * std


class CTGANTrainer(GANTrainerMixin, BaseDeepLearningTrainer):
    """Concrete CTGAN trainer using the Template Method pattern.

    This trainer implements the complete CTGAN training algorithm
    with WGAN-GP loss and conditional generation support.

    Example:
        >>> trainer = CTGANTrainer(CTGANTrainerConfig(
        ...     epochs=300,
        ...     generator_dim=(256, 256),
        ...     discriminator_dim=(256, 256),
        ... ))
        >>> result = trainer.train(transformed_data, data_dim)
    """

    def __init__(
        self,
        config: TrainingConfig,
        generator_dim: tuple = (256, 256),
        discriminator_dim: tuple = (256, 256),
        n_critic: int = 1,
        lambda_gp: float = 10.0,
        pac: int = 10,
    ) -> None:
        """Initialize CTGAN trainer.

        Args:
            config: Base training configuration
            generator_dim: Dimensions of generator hidden layers
            discriminator_dim: Dimensions of discriminator hidden layers
            n_critic: Number of discriminator updates per generator update
            lambda_gp: Gradient penalty coefficient
            pac: PacGAN pack size
        """
        super().__init__(config)
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.pac = pac

    def create_models(self, data_dim: int) -> Dict[str, Any]:
        """Create generator and discriminator models."""
        if self._backend == "pytorch":
            from genesis.backends.pytorch import Discriminator, Generator

            generator = Generator(
                embedding_dim=self.config.embedding_dim,
                generator_dim=self.generator_dim,
                data_dim=data_dim,
            ).to(self._device)

            discriminator = Discriminator(
                input_dim=data_dim,
                discriminator_dim=self.discriminator_dim,
                pac=self.pac,
            ).to(self._device)

            return {"generator": generator, "discriminator": discriminator}
        else:
            from genesis.backends.tensorflow import create_discriminator, create_generator

            generator = create_generator(
                embedding_dim=self.config.embedding_dim,
                generator_dim=self.generator_dim,
                data_dim=data_dim,
            )

            discriminator = create_discriminator(
                input_dim=data_dim,
                discriminator_dim=self.discriminator_dim,
                pac=self.pac,
            )

            return {"generator": generator, "discriminator": discriminator}

    def create_optimizers(self) -> Dict[str, Any]:
        """Create Adam optimizers for generator and discriminator."""
        if self._backend == "pytorch":
            import torch.optim as optim

            g_optimizer = optim.Adam(
                self.state.models["generator"].parameters(),
                lr=self.config.learning_rate,
                betas=(0.5, 0.9),
            )
            d_optimizer = optim.Adam(
                self.state.models["discriminator"].parameters(),
                lr=self.config.learning_rate,
                betas=(0.5, 0.9),
            )
            return {"generator": g_optimizer, "discriminator": d_optimizer}
        else:
            from tensorflow import keras

            g_optimizer = keras.optimizers.Adam(
                learning_rate=self.config.learning_rate,
                beta_1=0.5,
                beta_2=0.9,
            )
            d_optimizer = keras.optimizers.Adam(
                learning_rate=self.config.learning_rate,
                beta_1=0.5,
                beta_2=0.9,
            )
            return {"generator": g_optimizer, "discriminator": d_optimizer}

    def create_dataloader(self, data: np.ndarray) -> Any:
        """Create dataloader from training data."""
        if self._backend == "pytorch":
            from genesis.backends.pytorch import create_dataloader

            return create_dataloader(data, self.config.batch_size, shuffle=True)
        else:
            from genesis.backends.tensorflow import create_dataset

            return create_dataset(data, self.config.batch_size, shuffle=True)

    def train_step(self, dataloader: Any) -> Dict[str, float]:
        """Execute one training epoch."""
        if self._backend == "pytorch":
            return self._train_step_pytorch(dataloader)
        else:
            return self._train_step_tensorflow(dataloader)

    def _train_step_pytorch(self, dataloader: Any) -> Dict[str, float]:
        """PyTorch training step."""
        import torch

        generator = self.state.models["generator"]
        discriminator = self.state.models["discriminator"]
        g_optimizer = self.state.optimizers["generator"]
        d_optimizer = self.state.optimizers["discriminator"]

        g_losses = []
        d_losses = []

        for batch in dataloader:
            real_data = batch[0].to(self._device).float()
            batch_size = real_data.size(0)

            # Train discriminator
            for _ in range(self.n_critic):
                d_optimizer.zero_grad()

                noise = torch.randn(batch_size, self.config.embedding_dim, device=self._device)
                fake_data = generator(noise)

                d_real = discriminator(real_data)
                d_fake = discriminator(fake_data.detach())

                gp = self.compute_gradient_penalty(discriminator, real_data, fake_data.detach())
                d_loss = -d_real.mean() + d_fake.mean() + self.lambda_gp * gp

                d_loss.backward()
                d_optimizer.step()

            # Train generator
            g_optimizer.zero_grad()

            noise = torch.randn(batch_size, self.config.embedding_dim, device=self._device)
            fake_data = generator(noise)
            d_fake = discriminator(fake_data)
            g_loss = -d_fake.mean()

            g_loss.backward()
            g_optimizer.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        return {"g_loss": np.mean(g_losses), "d_loss": np.mean(d_losses)}

    def _train_step_tensorflow(self, dataset: Any) -> Dict[str, float]:
        """TensorFlow training step."""
        import tensorflow as tf

        generator = self.state.models["generator"]
        discriminator = self.state.models["discriminator"]
        g_optimizer = self.state.optimizers["generator"]
        d_optimizer = self.state.optimizers["discriminator"]

        g_losses = []
        d_losses = []

        for batch in dataset:
            real_data = tf.cast(batch, tf.float32)
            batch_size = tf.shape(real_data)[0]

            # Train discriminator
            for _ in range(self.n_critic):
                noise = tf.random.normal([batch_size, self.config.embedding_dim])

                with tf.GradientTape() as tape:
                    fake_data = generator(noise, training=True)
                    d_real = discriminator(real_data, training=True)
                    d_fake = discriminator(fake_data, training=True)
                    gp = self.compute_gradient_penalty(discriminator, real_data, fake_data)
                    d_loss = -tf.reduce_mean(d_real) + tf.reduce_mean(d_fake) + self.lambda_gp * gp

                d_gradients = tape.gradient(d_loss, discriminator.trainable_variables)
                d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

            # Train generator
            noise = tf.random.normal([batch_size, self.config.embedding_dim])

            with tf.GradientTape() as tape:
                fake_data = generator(noise, training=True)
                d_fake = discriminator(fake_data, training=True)
                g_loss = -tf.reduce_mean(d_fake)

            g_gradients = tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

            g_losses.append(float(g_loss))
            d_losses.append(float(d_loss))

        return {"g_loss": np.mean(g_losses), "d_loss": np.mean(d_losses)}

    def get_primary_loss_key(self) -> str:
        """Return generator loss as primary metric."""
        return "g_loss"

    @property
    def generator(self) -> Any:
        """Get the trained generator model."""
        return self.state.models.get("generator")

    @property
    def discriminator(self) -> Any:
        """Get the trained discriminator model."""
        return self.state.models.get("discriminator")
