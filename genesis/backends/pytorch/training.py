"""PyTorch training utilities."""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None


def _check_pytorch():
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch is required. Install with: pip install genesis-synth[pytorch]")


class Trainer:
    """Generic trainer for PyTorch models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: str = "cpu",
        gradient_clip: Optional[float] = None,
    ) -> None:
        _check_pytorch()
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.gradient_clip = gradient_clip
        self.history: Dict[str, List[float]] = {"loss": []}

    def train_step(
        self,
        batch: torch.Tensor,
        loss_fn: Callable,
    ) -> float:
        """Execute a single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        batch = batch.to(self.device)
        loss = loss_fn(self.model, batch)

        loss.backward()

        if self.gradient_clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

        self.optimizer.step()

        return loss.item()

    def train_epoch(
        self,
        dataloader: DataLoader,
        loss_fn: Callable,
    ) -> float:
        """Train for one epoch."""
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            loss = self.train_step(batch, loss_fn)
            total_loss += loss
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        self.history["loss"].append(avg_loss)

        return avg_loss


class GANTrainer:
    """Trainer for GAN-based models."""

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        g_optimizer: optim.Optimizer,
        d_optimizer: optim.Optimizer,
        device: str = "cpu",
        n_critic: int = 1,
        lambda_gp: float = 10.0,
    ) -> None:
        _check_pytorch()
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.device = device
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp

        self.history: Dict[str, List[float]] = {
            "g_loss": [],
            "d_loss": [],
        }

    def train_discriminator_step(
        self,
        real_data: torch.Tensor,
        noise: torch.Tensor,
        conditional: Optional[torch.Tensor] = None,
    ) -> float:
        """Train discriminator for one step."""
        self.discriminator.train()
        self.d_optimizer.zero_grad()

        real_data = real_data.to(self.device)
        noise = noise.to(self.device)

        # Generate fake data
        if conditional is not None:
            conditional = conditional.to(self.device)
            gen_input = torch.cat([noise, conditional], dim=1)
        else:
            gen_input = noise

        with torch.no_grad():
            fake_data = self.generator(gen_input)

        # If conditional, concatenate condition
        if conditional is not None:
            real_input = torch.cat([real_data, conditional], dim=1)
            fake_input = torch.cat([fake_data, conditional], dim=1)
        else:
            real_input = real_data
            fake_input = fake_data

        # Discriminator outputs
        real_validity = self.discriminator(real_input)
        fake_validity = self.discriminator(fake_input)

        # WGAN loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)

        # Gradient penalty
        if hasattr(self.discriminator, "calc_gradient_penalty"):
            gp = self.discriminator.calc_gradient_penalty(
                real_input.data, fake_input.data, self.device, self.lambda_gp
            )
            d_loss += gp

        d_loss.backward()
        self.d_optimizer.step()

        return d_loss.item()

    def train_generator_step(
        self,
        noise: torch.Tensor,
        conditional: Optional[torch.Tensor] = None,
    ) -> float:
        """Train generator for one step."""
        self.generator.train()
        self.g_optimizer.zero_grad()

        noise = noise.to(self.device)

        if conditional is not None:
            conditional = conditional.to(self.device)
            gen_input = torch.cat([noise, conditional], dim=1)
        else:
            gen_input = noise

        fake_data = self.generator(gen_input)

        if conditional is not None:
            fake_input = torch.cat([fake_data, conditional], dim=1)
        else:
            fake_input = fake_data

        fake_validity = self.discriminator(fake_input)
        g_loss = -torch.mean(fake_validity)

        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.item()

    def train_epoch(
        self,
        dataloader: DataLoader,
        embedding_dim: int,
        cond_dim: int = 0,
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        total_g_loss = 0.0
        total_d_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                if len(batch) > 1:
                    real_data, conditional = batch[0], batch[1]
                else:
                    real_data = batch[0]
                    conditional = None
            else:
                real_data = batch
                conditional = None

            batch_size = real_data.size(0)

            # Train discriminator
            for _ in range(self.n_critic):
                noise = torch.randn(batch_size, embedding_dim, device=self.device)
                d_loss = self.train_discriminator_step(real_data, noise, conditional)

            # Train generator
            noise = torch.randn(batch_size, embedding_dim, device=self.device)
            g_loss = self.train_generator_step(noise, conditional)

            total_d_loss += d_loss
            total_g_loss += g_loss
            n_batches += 1

        avg_g_loss = total_g_loss / max(n_batches, 1)
        avg_d_loss = total_d_loss / max(n_batches, 1)

        self.history["g_loss"].append(avg_g_loss)
        self.history["d_loss"].append(avg_d_loss)

        return avg_g_loss, avg_d_loss


class VAETrainer:
    """Trainer for VAE-based models."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        optimizer: optim.Optimizer,
        device: str = "cpu",
        beta: float = 1.0,
    ) -> None:
        _check_pytorch()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.optimizer = optimizer
        self.device = device
        self.beta = beta

        self.history: Dict[str, List[float]] = {
            "loss": [],
            "recon_loss": [],
            "kl_loss": [],
        }

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def train_step(
        self,
        batch: torch.Tensor,
        output_info: List[Tuple[int, str]],
    ) -> Tuple[float, float, float]:
        """Execute a single training step."""
        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()

        batch = batch.to(self.device)

        # Encode
        mu, logvar = self.encoder(batch)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        recon = self.decoder(z)

        # Reconstruction loss (mode-specific)
        recon_loss = self._compute_reconstruction_loss(batch, recon, output_info)

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        loss = recon_loss + self.beta * kl_loss

        loss.backward()
        self.optimizer.step()

        return loss.item(), recon_loss.item(), kl_loss.item()

    def _compute_reconstruction_loss(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        output_info: List[Tuple[int, str]],
    ) -> torch.Tensor:
        """Compute mode-specific reconstruction loss."""
        loss = 0.0
        offset = 0

        for dim, activation in output_info:
            orig_slice = original[:, offset : offset + dim]
            recon_slice = reconstructed[:, offset : offset + dim]

            if activation == "tanh":
                # Continuous: MSE loss
                loss += nn.functional.mse_loss(torch.tanh(recon_slice), orig_slice, reduction="sum")
            elif activation == "softmax":
                # Categorical: Cross entropy
                loss += nn.functional.cross_entropy(
                    recon_slice, torch.argmax(orig_slice, dim=1), reduction="sum"
                )

            offset += dim

        return loss / original.size(0)

    def train_epoch(
        self,
        dataloader: DataLoader,
        output_info: List[Tuple[int, str]],
    ) -> Tuple[float, float, float]:
        """Train for one epoch."""
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            loss, recon, kl = self.train_step(batch, output_info)
            total_loss += loss
            total_recon += recon
            total_kl += kl
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_recon = total_recon / max(n_batches, 1)
        avg_kl = total_kl / max(n_batches, 1)

        self.history["loss"].append(avg_loss)
        self.history["recon_loss"].append(avg_recon)
        self.history["kl_loss"].append(avg_kl)

        return avg_loss, avg_recon, avg_kl


def create_dataloader(
    data: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    conditional: Optional[np.ndarray] = None,
) -> DataLoader:
    """Create a DataLoader from numpy arrays."""
    _check_pytorch()

    tensor_data = torch.FloatTensor(data)

    if conditional is not None:
        tensor_cond = torch.FloatTensor(conditional)
        dataset = TensorDataset(tensor_data, tensor_cond)
    else:
        dataset = TensorDataset(tensor_data)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    _check_pytorch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
