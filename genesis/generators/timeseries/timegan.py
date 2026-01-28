"""TimeGAN implementation for time series synthesis."""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from genesis.backends import get_device, select_backend
from genesis.core.config import GeneratorConfig, PrivacyConfig, TimeSeriesConfig
from genesis.core.types import FittingResult, ProgressCallback
from genesis.generators.timeseries.base import BaseTimeSeriesGenerator
from genesis.utils.logging import TrainingLogger, get_logger

logger = get_logger(__name__)


class TimeGANGenerator(BaseTimeSeriesGenerator):
    """TimeGAN for synthetic time series generation.

    TimeGAN uses a combination of autoencoder and GAN to generate
    realistic time series data while preserving temporal dynamics.

    Reference:
        Yoon et al., "Time-series Generative Adversarial Networks" (2019)

    Example:
        >>> generator = TimeGANGenerator(sequence_length=24, epochs=1000)
        >>> generator.fit(time_series_data)
        >>> synthetic_data = generator.generate(n_samples=100)
    """

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        privacy: Optional[PrivacyConfig] = None,
        ts_config: Optional[TimeSeriesConfig] = None,
        epochs: int = 1000,
        batch_size: int = 128,
        sequence_length: int = 24,
        hidden_dim: int = 24,
        n_layers: int = 3,
        learning_rate: float = 1e-3,
        device: str = "auto",
        verbose: bool = True,
    ) -> None:
        ts_config = ts_config or TimeSeriesConfig(
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )
        super().__init__(config, privacy, ts_config)

        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.device = device
        self.verbose = verbose

        self._embedder = None
        self._recovery = None
        self._generator = None
        self._discriminator = None
        self._backend = None

    def _fit_impl(
        self,
        data: pd.DataFrame,
        discrete_columns: Optional[List[str]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> FittingResult:
        """Fit TimeGAN to time series data."""
        start_time = time.time()

        self._backend = select_backend(self.config.backend)
        device = get_device(self.device)

        if self.verbose:
            logger.info(f"Training TimeGAN with {self._backend} backend on {device}")

        # Prepare sequences
        sequences = self._prepare_sequences(data)
        n_sequences, seq_len, n_features = sequences.shape

        if self.verbose:
            logger.info(
                f"Prepared {n_sequences} sequences of length {seq_len} with {n_features} features"
            )

        # Train based on backend
        if self._backend == "pytorch":
            self._train_pytorch(sequences, device, progress_callback)
        else:
            self._train_tensorflow(sequences, progress_callback)

        fitting_time = time.time() - start_time

        if self.verbose:
            logger.info(f"Training completed in {fitting_time:.2f}s")

        return FittingResult(
            success=True,
            fitting_time=fitting_time,
            n_epochs=self.epochs,
        )

    def _train_pytorch(
        self,
        sequences: np.ndarray,
        device: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Train TimeGAN using PyTorch."""
        import torch
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        from genesis.backends.pytorch import (
            EmbeddingNetwork,
            RecoveryNetwork,
            TimeSeriesDiscriminator,
            TimeSeriesGenerator,
            set_seed,
        )

        if self.config.random_seed is not None:
            set_seed(self.config.random_seed)

        _, seq_len, n_features = sequences.shape
        z_dim = n_features

        # Create networks
        self._embedder = EmbeddingNetwork(n_features, self.hidden_dim, self.n_layers).to(device)
        self._recovery = RecoveryNetwork(self.hidden_dim, n_features, self.n_layers).to(device)
        self._generator = TimeSeriesGenerator(
            z_dim, self.hidden_dim, self.n_layers, self.hidden_dim
        ).to(device)
        self._discriminator = TimeSeriesDiscriminator(
            self.hidden_dim, self.hidden_dim, self.n_layers
        ).to(device)

        # Optimizers
        e_opt = optim.Adam(self._embedder.parameters(), lr=self.learning_rate)
        r_opt = optim.Adam(self._recovery.parameters(), lr=self.learning_rate)
        g_opt = optim.Adam(self._generator.parameters(), lr=self.learning_rate)
        d_opt = optim.Adam(self._discriminator.parameters(), lr=self.learning_rate)

        # DataLoader
        tensor_data = torch.FloatTensor(sequences).to(device)
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        training_logger = TrainingLogger(self.epochs, logger, log_every_n_epochs=100)

        # Phase 1: Autoencoder training
        for epoch in range(self.epochs // 2):
            e_loss_sum = 0.0
            for (batch,) in dataloader:
                e_opt.zero_grad()
                r_opt.zero_grad()

                # Embed and recover
                h = self._embedder(batch)
                x_tilde = self._recovery(h)

                # Reconstruction loss
                e_loss = torch.mean(torch.abs(batch - x_tilde))
                e_loss.backward()

                e_opt.step()
                r_opt.step()
                e_loss_sum += e_loss.item()

            if epoch % 100 == 0:
                training_logger.log_epoch(epoch, {"e_loss": e_loss_sum / len(dataloader)})

        # Phase 2: Joint training
        for epoch in range(self.epochs // 2, self.epochs):
            g_loss_sum = 0.0
            d_loss_sum = 0.0

            for (batch,) in dataloader:
                batch_size = batch.size(0)

                # Train discriminator
                d_opt.zero_grad()

                # Real embeddings
                h_real = self._embedder(batch)

                # Fake embeddings
                z = torch.randn(batch_size, seq_len, z_dim, device=device)
                h_fake = self._generator(z)

                # Discriminator loss
                d_real = self._discriminator(h_real)
                d_fake = self._discriminator(h_fake.detach())

                d_loss = -torch.mean(d_real) + torch.mean(d_fake)
                d_loss.backward()
                d_opt.step()

                # Train generator
                g_opt.zero_grad()

                h_fake = self._generator(z)
                d_fake = self._discriminator(h_fake)

                g_loss = -torch.mean(d_fake)
                g_loss.backward()
                g_opt.step()

                g_loss_sum += g_loss.item()
                d_loss_sum += d_loss.item()

            if epoch % 100 == 0:
                training_logger.log_epoch(
                    epoch,
                    {
                        "g_loss": g_loss_sum / len(dataloader),
                        "d_loss": d_loss_sum / len(dataloader),
                    },
                )

            if progress_callback:
                progress_callback(epoch, self.epochs, {"g_loss": g_loss_sum / len(dataloader)})

    def _train_tensorflow(
        self,
        sequences: np.ndarray,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Train TimeGAN using TensorFlow."""
        import tensorflow as tf

        from genesis.backends.tensorflow import (
            create_embedding_network,
            create_recovery_network,
            create_timeseries_discriminator,
            create_timeseries_generator,
            set_seed,
        )

        if self.config.random_seed is not None:
            set_seed(self.config.random_seed)

        _, seq_len, n_features = sequences.shape
        z_dim = n_features

        # Create networks
        self._embedder = create_embedding_network(n_features, self.hidden_dim, self.n_layers)
        self._recovery = create_recovery_network(self.hidden_dim, n_features, self.n_layers)
        self._generator = create_timeseries_generator(
            z_dim, self.hidden_dim, self.n_layers, self.hidden_dim
        )
        self._discriminator = create_timeseries_discriminator(
            self.hidden_dim, self.hidden_dim, self.n_layers
        )

        # Optimizers
        g_opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(sequences.astype(np.float32))
        dataset = dataset.shuffle(len(sequences)).batch(self.batch_size)

        training_logger = TrainingLogger(self.epochs, logger, log_every_n_epochs=100)

        # Simplified training loop
        for epoch in range(self.epochs):
            g_loss_sum = 0.0
            n_batches = 0

            for batch in dataset:
                batch_size = tf.shape(batch)[0]
                z = tf.random.normal([batch_size, seq_len, z_dim])

                with tf.GradientTape() as g_tape:
                    h_fake = self._generator(z, training=True)
                    d_fake = self._discriminator(h_fake, training=False)
                    g_loss = -tf.reduce_mean(d_fake)

                g_grads = g_tape.gradient(g_loss, self._generator.trainable_variables)
                g_opt.apply_gradients(zip(g_grads, self._generator.trainable_variables))

                g_loss_sum += float(g_loss)
                n_batches += 1

            if epoch % 100 == 0:
                training_logger.log_epoch(epoch, {"g_loss": g_loss_sum / max(n_batches, 1)})

    def _generate_impl(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> pd.DataFrame:
        """Generate synthetic time series using trained TimeGAN."""
        if self._generator is None or self._recovery is None:
            raise RuntimeError("Model not trained")

        if self._backend == "pytorch":
            return self._generate_pytorch(n_samples)
        else:
            return self._generate_tensorflow(n_samples)

    def _generate_pytorch(self, n_samples: int) -> pd.DataFrame:
        """Generate using PyTorch backend."""
        import torch

        self._generator.eval()
        self._recovery.eval()
        device = next(self._generator.parameters()).device

        seq_len = self._sequence_length
        z_dim = self._n_features

        with torch.no_grad():
            z = torch.randn(n_samples, seq_len, z_dim, device=device)
            h_fake = self._generator(z)
            x_fake = self._recovery(h_fake)

        sequences = x_fake.cpu().numpy()
        return self._sequences_to_dataframe(sequences)

    def _generate_tensorflow(self, n_samples: int) -> pd.DataFrame:
        """Generate using TensorFlow backend."""
        import tensorflow as tf

        seq_len = self._sequence_length
        z_dim = self._n_features

        z = tf.random.normal([n_samples, seq_len, z_dim])
        h_fake = self._generator(z, training=False)
        x_fake = self._recovery(h_fake, training=False)

        sequences = x_fake.numpy()
        return self._sequences_to_dataframe(sequences)
