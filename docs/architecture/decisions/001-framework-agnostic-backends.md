# ADR 001: Framework-Agnostic Deep Learning Backends

## Status

Accepted

## Context

Genesis needs to support deep learning-based generators (CTGAN, TVAE, TimeGAN) that require neural network frameworks. The two dominant frameworks are PyTorch and TensorFlow, each with its own ecosystem and user base.

Key considerations:
- Users may already have one framework installed and prefer not to install another
- Different organizations have different framework preferences
- Framework APIs differ significantly
- We want to avoid forcing users to install heavy dependencies they don't need

## Decision

Implement a **backend abstraction layer** that allows users to choose between PyTorch and TensorFlow.

### Architecture

```
genesis/backends/
├── __init__.py      # Backend selection logic
├── pytorch/
│   ├── networks.py  # PyTorch network architectures
│   └── training.py  # PyTorch training utilities
└── tensorflow/
    ├── networks.py  # TensorFlow network architectures
    └── training.py  # TensorFlow training utilities
```

### Selection Logic

1. If user explicitly specifies `device="cuda"` or imports PyTorch, use PyTorch
2. If PyTorch is available, prefer PyTorch (better ecosystem, easier debugging)
3. If only TensorFlow is available, use TensorFlow
4. If neither is available, fall back to non-DL methods (Gaussian Copula)

### API Contract

Both backends implement the same interface:
- `Generator(input_dim, output_dim, hidden_dims)` - Generator network
- `Discriminator(input_dim, hidden_dims)` - Discriminator network
- `Encoder(input_dim, latent_dim, hidden_dims)` - VAE encoder
- `Decoder(latent_dim, output_dim, hidden_dims)` - VAE decoder

## Consequences

### Positive

- Users can use their preferred framework
- Neither framework is a hard dependency
- Easier onboarding for users with existing framework preferences
- Gaussian Copula works without any DL framework

### Negative

- Code duplication between backends
- Need to maintain two implementations
- Subtle differences in behavior between frameworks
- Testing complexity increases

### Neutral

- Auto-selection favors PyTorch, which may not match all user preferences
- Users need to install framework as optional dependency
