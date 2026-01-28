"""TensorFlow neural network architectures for synthetic data generation."""

from typing import Tuple

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None
    layers = None


def _check_tensorflow():
    if not TENSORFLOW_AVAILABLE:
        raise ImportError(
            "TensorFlow is required. Install with: pip install genesis-synth[tensorflow]"
        )


class ResidualLayer(layers.Layer):
    """Residual layer for generator networks."""

    def __init__(self, units: int, **kwargs) -> None:
        _check_tensorflow()
        super().__init__(**kwargs)
        self.dense = layers.Dense(units)
        self.bn = layers.BatchNormalization()

    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        out = self.dense(x)
        out = self.bn(out, training=training)
        out = tf.nn.relu(out)
        return tf.concat([out, x], axis=1)


def create_generator(
    embedding_dim: int,
    generator_dim: Tuple[int, ...],
    data_dim: int,
) -> keras.Model:
    """Create generator model for CTGAN/TVAE."""
    _check_tensorflow()

    inputs = keras.Input(shape=(embedding_dim,))
    x = inputs

    for dim in generator_dim:
        dense = layers.Dense(dim)
        bn = layers.BatchNormalization()
        out = dense(x)
        out = bn(out)
        out = tf.nn.relu(out)
        x = tf.concat([out, x], axis=1)

    outputs = layers.Dense(data_dim)(x)

    return keras.Model(inputs, outputs, name="generator")


def create_discriminator(
    input_dim: int,
    discriminator_dim: Tuple[int, ...],
    pac: int = 10,
) -> keras.Model:
    """Create discriminator model for CTGAN."""
    _check_tensorflow()

    pacdim = input_dim * pac
    inputs = keras.Input(shape=(pacdim,))
    x = inputs

    for dim in discriminator_dim:
        x = layers.Dense(dim)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1)(x)

    return keras.Model(inputs, outputs, name="discriminator")


def create_encoder(
    data_dim: int,
    encoder_dim: Tuple[int, ...],
    embedding_dim: int,
) -> keras.Model:
    """Create encoder model for TVAE."""
    _check_tensorflow()

    inputs = keras.Input(shape=(data_dim,))
    x = inputs

    for dim in encoder_dim:
        x = layers.Dense(dim, activation="relu")(x)

    mu = layers.Dense(embedding_dim, name="mu")(x)
    logvar = layers.Dense(embedding_dim, name="logvar")(x)

    return keras.Model(inputs, [mu, logvar], name="encoder")


def create_decoder(
    embedding_dim: int,
    decoder_dim: Tuple[int, ...],
    data_dim: int,
) -> keras.Model:
    """Create decoder model for TVAE."""
    _check_tensorflow()

    inputs = keras.Input(shape=(embedding_dim,))
    x = inputs

    for dim in decoder_dim:
        x = layers.Dense(dim, activation="relu")(x)

    outputs = layers.Dense(data_dim)(x)

    return keras.Model(inputs, outputs, name="decoder")


def create_timeseries_generator(
    z_dim: int,
    hidden_dim: int,
    n_layers: int,
    n_features: int,
) -> keras.Model:
    """Create generator for time series."""
    _check_tensorflow()

    inputs = keras.Input(shape=(None, z_dim))
    x = inputs

    for _ in range(n_layers):
        x = layers.GRU(hidden_dim, return_sequences=True)(x)

    outputs = layers.Dense(n_features, activation="sigmoid")(x)

    return keras.Model(inputs, outputs, name="ts_generator")


def create_timeseries_discriminator(
    n_features: int,
    hidden_dim: int,
    n_layers: int,
) -> keras.Model:
    """Create discriminator for time series."""
    _check_tensorflow()

    inputs = keras.Input(shape=(None, n_features))
    x = inputs

    for i in range(n_layers):
        return_seq = i < n_layers - 1
        x = layers.GRU(hidden_dim, return_sequences=return_seq)(x)

    outputs = layers.Dense(1)(x)

    return keras.Model(inputs, outputs, name="ts_discriminator")


def create_embedding_network(
    n_features: int,
    hidden_dim: int,
    n_layers: int,
) -> keras.Model:
    """Create embedding network for TimeGAN."""
    _check_tensorflow()

    inputs = keras.Input(shape=(None, n_features))
    x = inputs

    for _ in range(n_layers):
        x = layers.GRU(hidden_dim, return_sequences=True)(x)

    outputs = layers.Dense(hidden_dim, activation="sigmoid")(x)

    return keras.Model(inputs, outputs, name="embedding")


def create_recovery_network(
    hidden_dim: int,
    n_features: int,
    n_layers: int,
) -> keras.Model:
    """Create recovery network for TimeGAN."""
    _check_tensorflow()

    inputs = keras.Input(shape=(None, hidden_dim))
    x = inputs

    for _ in range(n_layers):
        x = layers.GRU(hidden_dim, return_sequences=True)(x)

    outputs = layers.Dense(n_features, activation="sigmoid")(x)

    return keras.Model(inputs, outputs, name="recovery")
