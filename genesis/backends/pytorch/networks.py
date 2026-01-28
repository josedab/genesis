"""PyTorch neural network architectures for synthetic data generation."""

from typing import List, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


def _check_pytorch():
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch is required. Install with: pip install genesis-synth[pytorch]")


class Residual(nn.Module):
    """Residual layer for generator networks."""

    def __init__(self, in_features: int, out_features: int) -> None:
        _check_pytorch()
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        out = self.bn(out)
        out = F.relu(out)
        return torch.cat([out, x], dim=1)


class Generator(nn.Module):
    """Generator network for CTGAN/TVAE."""

    def __init__(
        self,
        embedding_dim: int,
        generator_dim: Tuple[int, ...],
        data_dim: int,
    ) -> None:
        _check_pytorch()
        super().__init__()

        dim = embedding_dim
        layers: List[nn.Module] = []

        for layer_dim in generator_dim:
            layers.append(Residual(dim, layer_dim))
            dim += layer_dim

        layers.append(nn.Linear(dim, data_dim))
        self.seq = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class Discriminator(nn.Module):
    """Discriminator network for CTGAN."""

    def __init__(
        self,
        input_dim: int,
        discriminator_dim: Tuple[int, ...],
        pac: int = 10,
    ) -> None:
        _check_pytorch()
        super().__init__()

        self.pac = pac
        self.pacdim = input_dim * pac

        layers: List[nn.Module] = []
        dim = self.pacdim

        for layer_dim in discriminator_dim:
            layers.append(nn.Linear(dim, layer_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.5))
            dim = layer_dim

        layers.append(nn.Linear(dim, 1))
        self.seq = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        if batch_size % self.pac != 0:
            pad_size = self.pac - (batch_size % self.pac)
            x = torch.cat([x, x[:pad_size]], dim=0)
            batch_size = x.size(0)

        x = x.view(batch_size // self.pac, self.pacdim)
        return self.seq(x)

    def calc_gradient_penalty(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        device: str = "cpu",
        lambda_gp: float = 10.0,
    ) -> torch.Tensor:
        """Calculate gradient penalty for WGAN-GP."""
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, device=device)
        alpha = alpha.expand_as(real_data)

        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates.requires_grad_(True)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()

        return gradient_penalty


class Encoder(nn.Module):
    """Encoder network for TVAE."""

    def __init__(
        self,
        data_dim: int,
        encoder_dim: Tuple[int, ...],
        embedding_dim: int,
    ) -> None:
        _check_pytorch()
        super().__init__()

        dim = data_dim
        layers: List[nn.Module] = []

        for layer_dim in encoder_dim:
            layers.append(nn.Linear(dim, layer_dim))
            layers.append(nn.ReLU())
            dim = layer_dim

        self.seq = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(dim, embedding_dim)
        self.fc_logvar = nn.Linear(dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.seq(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Decoder network for TVAE."""

    def __init__(
        self,
        embedding_dim: int,
        decoder_dim: Tuple[int, ...],
        data_dim: int,
    ) -> None:
        _check_pytorch()
        super().__init__()

        dim = embedding_dim
        layers: List[nn.Module] = []

        for layer_dim in decoder_dim:
            layers.append(nn.Linear(dim, layer_dim))
            layers.append(nn.ReLU())
            dim = layer_dim

        layers.append(nn.Linear(dim, data_dim))
        self.seq = nn.Sequential(*layers)
        self.sigma = nn.Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class TimeSeriesGenerator(nn.Module):
    """Generator for time series (TimeGAN)."""

    def __init__(
        self,
        z_dim: int,
        hidden_dim: int,
        n_layers: int,
        n_features: int,
    ) -> None:
        _check_pytorch()
        super().__init__()

        self.rnn = nn.GRU(
            input_size=z_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, n_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(z)
        return torch.sigmoid(self.fc(out))


class TimeSeriesDiscriminator(nn.Module):
    """Discriminator for time series (TimeGAN)."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        n_layers: int,
    ) -> None:
        _check_pytorch()
        super().__init__()

        self.rnn = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class EmbeddingNetwork(nn.Module):
    """Embedding network for TimeGAN."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        n_layers: int,
    ) -> None:
        _check_pytorch()
        super().__init__()

        self.rnn = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return torch.sigmoid(self.fc(out))


class RecoveryNetwork(nn.Module):
    """Recovery network for TimeGAN."""

    def __init__(
        self,
        hidden_dim: int,
        n_features: int,
        n_layers: int,
    ) -> None:
        _check_pytorch()
        super().__init__()

        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, n_features)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(h)
        return torch.sigmoid(self.fc(out))
