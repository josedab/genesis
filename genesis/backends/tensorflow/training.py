"""TensorFlow training utilities."""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None


def _check_tensorflow():
    if not TENSORFLOW_AVAILABLE:
        raise ImportError(
            "TensorFlow is required. Install with: pip install genesis-synth[tensorflow]"
        )


class GANTrainer:
    """Trainer for GAN-based models using TensorFlow."""

    def __init__(
        self,
        generator: keras.Model,
        discriminator: keras.Model,
        g_optimizer: keras.optimizers.Optimizer,
        d_optimizer: keras.optimizers.Optimizer,
        n_critic: int = 1,
        lambda_gp: float = 10.0,
    ) -> None:
        _check_tensorflow()
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp

        self.history: Dict[str, List[float]] = {
            "g_loss": [],
            "d_loss": [],
        }

    @tf.function
    def _train_discriminator_step(
        self,
        real_data: tf.Tensor,
        noise: tf.Tensor,
        conditional: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """Train discriminator for one step."""
        # Generate fake data
        if conditional is not None:
            gen_input = tf.concat([noise, conditional], axis=1)
        else:
            gen_input = noise

        fake_data = self.generator(gen_input, training=False)

        # Prepare inputs
        if conditional is not None:
            real_input = tf.concat([real_data, conditional], axis=1)
            fake_input = tf.concat([fake_data, conditional], axis=1)
        else:
            real_input = real_data
            fake_input = fake_data

        with tf.GradientTape() as tape:
            real_validity = self.discriminator(real_input, training=True)
            fake_validity = self.discriminator(fake_input, training=True)

            # WGAN loss
            d_loss = -tf.reduce_mean(real_validity) + tf.reduce_mean(fake_validity)

            # Gradient penalty
            gp = self._gradient_penalty(real_input, fake_input)
            d_loss += self.lambda_gp * gp

        gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        return d_loss

    @tf.function
    def _train_generator_step(
        self,
        noise: tf.Tensor,
        conditional: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """Train generator for one step."""
        if conditional is not None:
            gen_input = tf.concat([noise, conditional], axis=1)
        else:
            gen_input = noise

        with tf.GradientTape() as tape:
            fake_data = self.generator(gen_input, training=True)

            if conditional is not None:
                fake_input = tf.concat([fake_data, conditional], axis=1)
            else:
                fake_input = fake_data

            fake_validity = self.discriminator(fake_input, training=False)
            g_loss = -tf.reduce_mean(fake_validity)

        gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        return g_loss

    def _gradient_penalty(
        self,
        real_data: tf.Tensor,
        fake_data: tf.Tensor,
    ) -> tf.Tensor:
        """Calculate gradient penalty."""
        batch_size = tf.shape(real_data)[0]
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)

        interpolates = alpha * real_data + (1 - alpha) * fake_data

        with tf.GradientTape() as tape:
            tape.watch(interpolates)
            pred = self.discriminator(interpolates, training=True)

        gradients = tape.gradient(pred, interpolates)
        gradients = tf.reshape(gradients, [batch_size, -1])
        gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-8)

        return tf.reduce_mean(tf.square(gradient_norm - 1.0))

    def train_epoch(
        self,
        dataset: tf.data.Dataset,
        embedding_dim: int,
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        total_g_loss = 0.0
        total_d_loss = 0.0
        n_batches = 0

        for batch in dataset:
            if isinstance(batch, (list, tuple)):
                if len(batch) > 1:
                    real_data, conditional = batch[0], batch[1]
                else:
                    real_data = batch[0]
                    conditional = None
            else:
                real_data = batch
                conditional = None

            batch_size = tf.shape(real_data)[0]

            # Train discriminator
            for _ in range(self.n_critic):
                noise = tf.random.normal([batch_size, embedding_dim])
                d_loss = self._train_discriminator_step(real_data, noise, conditional)

            # Train generator
            noise = tf.random.normal([batch_size, embedding_dim])
            g_loss = self._train_generator_step(noise, conditional)

            total_d_loss += float(d_loss)
            total_g_loss += float(g_loss)
            n_batches += 1

        avg_g_loss = total_g_loss / max(n_batches, 1)
        avg_d_loss = total_d_loss / max(n_batches, 1)

        self.history["g_loss"].append(avg_g_loss)
        self.history["d_loss"].append(avg_d_loss)

        return avg_g_loss, avg_d_loss


class VAETrainer:
    """Trainer for VAE-based models using TensorFlow."""

    def __init__(
        self,
        encoder: keras.Model,
        decoder: keras.Model,
        optimizer: keras.optimizers.Optimizer,
        beta: float = 1.0,
    ) -> None:
        _check_tensorflow()
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.beta = beta

        self.history: Dict[str, List[float]] = {
            "loss": [],
            "recon_loss": [],
            "kl_loss": [],
        }

    def reparameterize(
        self,
        mu: tf.Tensor,
        logvar: tf.Tensor,
    ) -> tf.Tensor:
        """Reparameterization trick for VAE."""
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(tf.shape(std))
        return mu + eps * std

    @tf.function
    def _train_step(
        self,
        batch: tf.Tensor,
        output_info: List[Tuple[int, str]],
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Execute a single training step."""
        with tf.GradientTape() as tape:
            # Encode
            mu, logvar = self.encoder(batch, training=True)

            # Reparameterize
            z = self.reparameterize(mu, logvar)

            # Decode
            recon = self.decoder(z, training=True)

            # Reconstruction loss
            recon_loss = self._compute_reconstruction_loss(batch, recon, output_info)

            # KL divergence
            kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar))

            # Total loss
            loss = recon_loss + self.beta * kl_loss

        # Get all trainable variables
        trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return loss, recon_loss, kl_loss

    def _compute_reconstruction_loss(
        self,
        original: tf.Tensor,
        reconstructed: tf.Tensor,
        output_info: List[Tuple[int, str]],
    ) -> tf.Tensor:
        """Compute mode-specific reconstruction loss."""
        loss = 0.0
        offset = 0

        for dim, activation in output_info:
            orig_slice = original[:, offset : offset + dim]
            recon_slice = reconstructed[:, offset : offset + dim]

            if activation == "tanh":
                loss += tf.reduce_sum(tf.square(tf.tanh(recon_slice) - orig_slice))
            elif activation == "softmax":
                loss += tf.reduce_sum(
                    tf.nn.softmax_cross_entropy_with_logits(labels=orig_slice, logits=recon_slice)
                )

            offset += dim

        return loss / tf.cast(tf.shape(original)[0], tf.float32)

    def train_epoch(
        self,
        dataset: tf.data.Dataset,
        output_info: List[Tuple[int, str]],
    ) -> Tuple[float, float, float]:
        """Train for one epoch."""
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        for batch in dataset:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            loss, recon, kl = self._train_step(batch, output_info)
            total_loss += float(loss)
            total_recon += float(recon)
            total_kl += float(kl)
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_recon = total_recon / max(n_batches, 1)
        avg_kl = total_kl / max(n_batches, 1)

        self.history["loss"].append(avg_loss)
        self.history["recon_loss"].append(avg_recon)
        self.history["kl_loss"].append(avg_kl)

        return avg_loss, avg_recon, avg_kl


def create_dataset(
    data: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    conditional: Optional[np.ndarray] = None,
) -> tf.data.Dataset:
    """Create a TensorFlow Dataset from numpy arrays."""
    _check_tensorflow()

    if conditional is not None:
        dataset = tf.data.Dataset.from_tensor_slices((data, conditional))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(data)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(data))

    dataset = dataset.batch(batch_size)
    return dataset.prefetch(tf.data.AUTOTUNE)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    _check_tensorflow()
    tf.random.set_seed(seed)
    np.random.seed(seed)
