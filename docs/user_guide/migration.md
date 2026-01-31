# Migration Guide

This guide helps you upgrade between Genesis versions and migrate to new features.

## Migrating from Pickle to Safe Serialization

Genesis v1.3+ introduces a new safe serialization format that replaces pickle for model persistence. This change improves security by preventing arbitrary code execution when loading model files.

### Why Migrate?

**Security Risk with Pickle:**
- Pickle files can execute arbitrary code when loaded
- Loading untrusted `.pkl` model files is a security vulnerability
- The new format uses JSON + numpy arrays, which cannot execute code

### How to Migrate Existing Models

#### Option 1: Re-train and Save with New Format (Recommended)

```python
from genesis.generators.tabular import GaussianCopulaGenerator

# Train your generator
generator = GaussianCopulaGenerator()
generator.fit(training_data)

# Save with new safe format (default in v1.3+)
generator.save("model.genesis")
```

#### Option 2: Convert Existing Pickle Models

If you have existing pickle-based models, you can convert them:

```python
from genesis.generators.tabular import GaussianCopulaGenerator

# Load old pickle model (only from trusted sources!)
generator = GaussianCopulaGenerator.load(
    "old_model.pkl",
    use_safe_serialization=False  # Required for pickle files
)

# Save with new safe format
generator.save("new_model.genesis", use_safe_serialization=True)
```

#### Option 3: Maintain Backward Compatibility

If you need to continue using pickle format temporarily:

```python
# Save with pickle (not recommended for production)
generator.save("model.pkl", use_safe_serialization=False)

# Load pickle file
generator = GaussianCopulaGenerator.load(
    "model.pkl",
    use_safe_serialization=False
)
```

### File Format Comparison

| Aspect | Pickle (Legacy) | Safe Format (New) |
|--------|----------------|-------------------|
| Extension | `.pkl` | `.genesis` (recommended) |
| Security | ⚠️ Can execute code | ✅ Data only |
| Format | Binary | ZIP with JSON + numpy |
| Portability | Python version dependent | More portable |
| Inspection | Not human-readable | Manifest is readable |

### New File Structure

The new `.genesis` format is a ZIP archive containing:

```
model.genesis/
├── manifest.json      # Version info, metadata
├── state.json         # Model parameters (scalars, small data)
└── arrays/
    ├── array_0.npy    # Large numpy arrays
    ├── array_1.npy
    └── ...
```

### Custom Serialization

If your generator subclass has custom state, implement these methods:

```python
class MyGenerator(BaseGenerator):
    def get_serialization_state(self) -> Dict[str, Any]:
        """Return state dictionary for serialization."""
        state = super().get_serialization_state()
        state["my_custom_field"] = self.my_custom_field
        return state
    
    def set_serialization_state(self, state: Dict[str, Any]) -> None:
        """Restore state from serialization."""
        super().set_serialization_state(state)
        self.my_custom_field = state.get("my_custom_field")
```

### Troubleshooting

**"Cannot load pickle file with safe_serialization=True"**
- The file is a legacy pickle format
- Use `use_safe_serialization=False` to load it
- Convert to new format after loading

**"Unknown array format in state"**
- Ensure numpy arrays are standard dtypes
- Complex custom objects may need manual handling

**"Version mismatch warning"**
- Model was saved with a different Genesis version
- Usually safe to ignore, but re-save for best compatibility

### Security Best Practices

1. **Never load untrusted pickle files** - They can execute malicious code
2. **Use `.genesis` extension** for new format to distinguish from pickle
3. **Migrate production models** to the new format
4. **Audit existing models** before loading in secure environments

## Migrating to Plugin Registry

Genesis v1.3+ encourages using the plugin registry for generator access.

### Old Approach (Direct Import)

```python
from genesis.generators.tabular import CTGANGenerator

generator = CTGANGenerator()
```

### New Approach (Plugin Registry)

```python
from genesis.plugins import get_generator

GeneratorClass = get_generator("ctgan")
generator = GeneratorClass()
```

### Benefits of Plugin Registry

- **Extensibility**: Custom generators integrate seamlessly
- **Discovery**: List available generators with `list_generators()`
- **Decoupling**: Code doesn't depend on specific import paths

### Registering Custom Generators

```python
from genesis.plugins import register_generator
from genesis.core.base import SyntheticGenerator

@register_generator("my_generator", description="My custom generator")
class MyGenerator(SyntheticGenerator):
    ...
```

## Configuration Changes

### GenesisConfig (v1.3+)

The new unified configuration class simplifies setup:

```python
from genesis.core.config import GenesisConfig

# Old way - multiple config objects
from genesis.core.config import GeneratorConfig, PrivacyConfig
gen_config = GeneratorConfig(epochs=500)
privacy_config = PrivacyConfig(level="high")
generator = CTGANGenerator(config=gen_config, privacy=privacy_config)

# New way - unified config
config = GenesisConfig(
    training={"epochs": 500},
    privacy={"level": "high"}
)
generator = CTGANGenerator.from_config(config)
```

## Getting Help

If you encounter migration issues:

1. Check the [changelog](../changelog.md) for breaking changes
2. Search [GitHub issues](https://github.com/genesis/genesis/issues)
3. Open a new issue with the "migration" label
