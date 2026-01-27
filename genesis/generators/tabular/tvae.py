"""TVAE (Tabular Variational Autoencoder) implementation."""

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


class TVAEGenerator(BaseTabularGenerator):
    """TVAE (Tabular Variational Autoencoder) for synthetic tabular data.

    TVAE uses a variational autoencoder architecture with mode-specific
    loss functions to generate realistic tabular data.

    Reference:
        Xu et al., "Modeling Tabular data using Conditional GAN" (2019)

    Example:
        >>> generator = TVAEGenerator(epochs=300)
        >>> generator.fit(real_data, discrete_columns=['gender'])
        >>> synthetic_data = generator.generate(n_samples=1000)
    """

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        privacy: Optional[PrivacyConfig] = None,
        epochs: int = 300,
        batch_size: int = 500,
        encoder_dim: Tuple[int, ...] = (128, 128),
        decoder_dim: Tuple[int, ...] = (128, 128),
        embedding_dim: int = 128,
        learning_rate: float = 1e-3,
        beta: float = 1.0,
        device: str = "auto",
        verbose: bool = True,
    ) -> None:
        """Initialize TVAE generator.

        Args:
            config: Generator configuration
            privacy: Privacy configuration
            epochs: Number of training epochs
            batch_size: Batch size for training
            encoder_dim: Dimensions of encoder hidden layers
            decoder_dim: Dimensions of decoder hidden layers
            embedding_dim: Dimension of latent space
            learning_rate: Learning rate for optimizer
            beta: Weight for KL divergence loss
            device: Device to use
            verbose: Whether to print progress
        """
        super().__init__(config, privacy)

        self.epochs = epochs if config is None else config.epochs
        self.batch_size = batch_size if config is None else config.batch_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.embedding_dim = embedding_dim if config is None else config.embedding_dim
        self.learning_rate = learning_rate if config is None else config.learning_rate
        self.beta = beta
        self.device = device if config is None else config.device
        self.verbose = verbose if config is None else config.verbose

        self._encoder = None
        self._decoder = None
        self._backend = None

    def _fit_impl(
        self,
        data: pd.DataFrame,
        discrete_columns: Optional[ColumnList] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> FittingResult:
        """Fit TVAE to the training data."""
        start_time = time.time()

        self._backend = select_backend(self.config.backend)
        device = get_device(self.device)

        if self.verbose:
            logger.info(f"Training TVAE with {self._backend} backend on {device}")

        # Transform data
        transformed_data = self._fit_transformer(data, discrete_columns)
        data_dim = transformed_data.shape[1]

        if self.verbose:
            logger.info(f"Data dimension: {data_dim}, Samples: {len(transformed_data)}")

        # Train
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
            Decoder,
            Encoder,
            VAETrainer,
            create_dataloader,
            set_seed,
        )

        if self.config.random_seed is not None:
            set_seed(self.config.random_seed)

        # Create models
        self._encoder = Encoder(
            data_dim=data_dim,
            encoder_dim=self.encoder_dim,
            embedding_dim=self.embedding_dim,
        ).to(device)

        self._decoder = Decoder(
            embedding_dim=self.embedding_dim,
            decoder_dim=self.decoder_dim,
            data_dim=data_dim,
        ).to(device)

        # Create optimizer
        optimizer = optim.Adam(
            list(self._encoder.parameters()) + list(self._decoder.parameters()),
            lr=self.learning_rate,
        )

        # Create dataloader
        dataloader = create_dataloader(data, self.batch_size, shuffle=True)

        # Create trainer
        trainer = VAETrainer(
            encoder=self._encoder,
            decoder=self._decoder,
            optimizer=optimizer,
            device=device,
            beta=self.beta,
        )

        # Training loop
        training_logger = TrainingLogger(self.epochs, logger, log_every_n_epochs=10)

        for epoch in range(self.epochs):
            loss, recon_loss, kl_loss = trainer.train_epoch(dataloader, self._output_info)

            training_logger.log_epoch(
                epoch,
                {"loss": loss, "recon": recon_loss, "kl": kl_loss},
            )

            if progress_callback:
                progress_callback(epoch, self.epochs, {"loss": loss})

        return trainer.history["loss"][-1] if trainer.history["loss"] else 0.0

    def _train_tensorflow(
        self,
        data: np.ndarray,
        data_dim: int,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> float:
        """Train using TensorFlow backend."""
        from tensorflow import keras

        from genesis.backends.tensorflow import (
            VAETrainer,
            create_dataset,
            create_decoder,
            create_encoder,
            set_seed,
        )

        if self.config.random_seed is not None:
            set_seed(self.config.random_seed)

        # Create models
        self._encoder = create_encoder(
            data_dim=data_dim,
            encoder_dim=self.encoder_dim,
            embedding_dim=self.embedding_dim,
        )

        self._decoder = create_decoder(
            embedding_dim=self.embedding_dim,
            decoder_dim=self.decoder_dim,
            data_dim=data_dim,
        )

        # Create optimizer
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Create dataset
        dataset = create_dataset(data, self.batch_size, shuffle=True)

        # Create trainer
        trainer = VAETrainer(
            encoder=self._encoder,
            decoder=self._decoder,
            optimizer=optimizer,
            beta=self.beta,
        )

        # Training loop
        training_logger = TrainingLogger(self.epochs, logger, log_every_n_epochs=10)

        for epoch in range(self.epochs):
            loss, recon_loss, kl_loss = trainer.train_epoch(dataset, self._output_info)

            training_logger.log_epoch(
                epoch,
                {"loss": loss, "recon": recon_loss, "kl": kl_loss},
            )

            if progress_callback:
                progress_callback(epoch, self.epochs, {"loss": loss})

        return trainer.history["loss"][-1] if trainer.history["loss"] else 0.0

    def _generate_impl(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> pd.DataFrame:
        """Generate synthetic data using trained TVAE."""
        if self._decoder is None:
            raise RuntimeError("Model not trained")

        if self._backend == "pytorch":
            return self._generate_pytorch(n_samples)
        else:
            return self._generate_tensorflow(n_samples)

    def _generate_pytorch(self, n_samples: int) -> pd.DataFrame:
        """Generate using PyTorch backend."""
        import torch

        self._decoder.eval()
        device = next(self._decoder.parameters()).device

        generated_data = []
        batch_size = min(self.batch_size, n_samples)
        n_batches = (n_samples + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in range(n_batches):
                current_batch = min(batch_size, n_samples - i * batch_size)
                z = torch.randn(current_batch, self.embedding_dim, device=device)

                fake_data = self._decoder(z)
                generated_data.append(fake_data.cpu().numpy())

        raw_output = np.vstack(generated_data)[:n_samples]
        activated = self._apply_activation(raw_output)
        sampled = self._sample_from_output(activated)

        return self._inverse_transform(sampled)

    def _generate_tensorflow(self, n_samples: int) -> pd.DataFrame:
        """Generate using TensorFlow backend."""
        import tensorflow as tf

        generated_data = []
        batch_size = min(self.batch_size, n_samples)
        n_batches = (n_samples + batch_size - 1) // batch_size

        for i in range(n_batches):
            current_batch = min(batch_size, n_samples - i * batch_size)
            z = tf.random.normal([current_batch, self.embedding_dim])

            fake_data = self._decoder(z, training=False)
            generated_data.append(fake_data.numpy())

        raw_output = np.vstack(generated_data)[:n_samples]
        activated = self._apply_activation(raw_output)
        sampled = self._sample_from_output(activated)

        return self._inverse_transform(sampled)
