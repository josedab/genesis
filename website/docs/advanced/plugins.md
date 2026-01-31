---
sidebar_position: 1
title: Plugins
---

# Plugin System

Extend Genesis with custom generators, transformers, and evaluators.

## Plugin Types

Genesis supports three types of plugins:

| Type | Purpose | Example |
|------|---------|---------|
| **Generator** | Custom generation methods | Industry-specific models |
| **Transformer** | Data preprocessing | Custom encoding schemes |
| **Evaluator** | Quality metrics | Domain-specific validation |

## Creating a Generator Plugin

### Basic Structure

```python
from genesis.plugins import GeneratorPlugin, register_plugin

@register_plugin('my_generator')
class MyGenerator(GeneratorPlugin):
    """Custom generator implementation."""
    
    name = 'my_generator'
    version = '1.0.0'
    
    def __init__(self, config=None):
        super().__init__(config)
        self.model = None
    
    def fit(self, data, discrete_columns=None, **kwargs):
        """Train the generator on data."""
        # Your training logic here
        self.model = train_model(data)
        return self
    
    def generate(self, n_samples, conditions=None):
        """Generate synthetic samples."""
        # Your generation logic here
        return self.model.sample(n_samples)
    
    def save(self, path):
        """Save the trained model."""
        save_model(self.model, path)
    
    @classmethod
    def load(cls, path):
        """Load a trained model."""
        instance = cls()
        instance.model = load_model(path)
        return instance
```

### Using Your Generator

```python
from genesis import SyntheticGenerator

# Use by name
generator = SyntheticGenerator(method='my_generator')
generator.fit(data)
synthetic = generator.generate(1000)
```

## Creating a Transformer Plugin

```python
from genesis.plugins import TransformerPlugin, register_plugin

@register_plugin('my_transformer')
class MyTransformer(TransformerPlugin):
    """Custom data transformer."""
    
    name = 'my_transformer'
    
    def fit(self, data):
        """Learn transformation parameters."""
        self.stats = compute_stats(data)
        return self
    
    def transform(self, data):
        """Apply transformation."""
        return apply_transform(data, self.stats)
    
    def inverse_transform(self, data):
        """Reverse transformation."""
        return reverse_transform(data, self.stats)
```

## Creating an Evaluator Plugin

```python
from genesis.plugins import EvaluatorPlugin, register_plugin

@register_plugin('my_evaluator')
class MyEvaluator(EvaluatorPlugin):
    """Custom quality evaluator."""
    
    name = 'my_evaluator'
    
    def evaluate(self, real_data, synthetic_data):
        """Compute quality metrics."""
        return {
            'my_metric': compute_metric(real_data, synthetic_data),
            'another_metric': compute_another(real_data, synthetic_data)
        }
```

## Plugin Discovery

### Automatic Discovery

Plugins in the `genesis_plugins` namespace are auto-discovered:

```
my_plugin_package/
├── setup.py
└── genesis_plugins/
    └── my_plugin/
        ├── __init__.py
        └── generator.py
```

```python
# setup.py
setup(
    name='genesis-my-plugin',
    packages=['genesis_plugins.my_plugin'],
    entry_points={
        'genesis.plugins': [
            'my_generator = genesis_plugins.my_plugin.generator:MyGenerator'
        ]
    }
)
```

### Manual Registration

```python
from genesis.plugins import register_plugin, PluginRegistry

# Register at import time
@register_plugin('custom_gen')
class CustomGenerator(GeneratorPlugin):
    ...

# Or register manually
PluginRegistry.register('custom_gen', CustomGenerator)
```

### List Available Plugins

```python
from genesis.plugins import PluginRegistry

# List all plugins
for name, plugin in PluginRegistry.list_all():
    print(f"{name}: {plugin.version}")

# List by type
generators = PluginRegistry.list_generators()
transformers = PluginRegistry.list_transformers()
evaluators = PluginRegistry.list_evaluators()
```

## Plugin Configuration

```python
@register_plugin('configurable_gen')
class ConfigurableGenerator(GeneratorPlugin):
    
    # Define configuration schema
    config_schema = {
        'epochs': {'type': 'int', 'default': 100, 'min': 1},
        'hidden_dim': {'type': 'int', 'default': 128},
        'learning_rate': {'type': 'float', 'default': 0.001}
    }
    
    def __init__(self, config=None):
        super().__init__(config)
        # Config is validated and defaults applied
        self.epochs = self.config['epochs']
        self.hidden_dim = self.config['hidden_dim']
```

## Plugin Dependencies

Declare dependencies for automatic checking:

```python
@register_plugin('torch_generator')
class TorchGenerator(GeneratorPlugin):
    
    dependencies = ['torch>=1.9.0', 'torchvision']
    
    def __init__(self, config=None):
        self.check_dependencies()  # Raises if missing
        super().__init__(config)
```

## Testing Plugins

```python
from genesis.plugins.testing import PluginTestCase

class TestMyGenerator(PluginTestCase):
    plugin_class = MyGenerator
    
    def test_fit_generate(self):
        """Test basic fit/generate cycle."""
        self.plugin.fit(self.sample_data)
        synthetic = self.plugin.generate(100)
        
        self.assertEqual(len(synthetic), 100)
        self.assertEqual(set(synthetic.columns), set(self.sample_data.columns))
    
    def test_save_load(self):
        """Test model persistence."""
        self.plugin.fit(self.sample_data)
        self.plugin.save('model.pkl')
        
        loaded = MyGenerator.load('model.pkl')
        synthetic = loaded.generate(100)
        
        self.assertEqual(len(synthetic), 100)
```

## Complete Example

```python
# my_industry_generator.py
import numpy as np
import pandas as pd
from genesis.plugins import GeneratorPlugin, register_plugin

@register_plugin('industry_specific')
class IndustrySpecificGenerator(GeneratorPlugin):
    """Generator optimized for financial transaction data."""
    
    name = 'industry_specific'
    version = '1.0.0'
    dependencies = ['scipy', 'scikit-learn']
    
    config_schema = {
        'n_components': {'type': 'int', 'default': 10},
        'random_state': {'type': 'int', 'default': None}
    }
    
    def __init__(self, config=None):
        self.check_dependencies()
        super().__init__(config)
        self.model = None
        self.columns = None
    
    def fit(self, data, discrete_columns=None, **kwargs):
        from sklearn.mixture import GaussianMixture
        
        self.columns = data.columns.tolist()
        self.discrete_columns = discrete_columns or []
        
        # Fit model (simplified example)
        numeric_data = data.select_dtypes(include=[np.number])
        self.model = GaussianMixture(
            n_components=self.config['n_components'],
            random_state=self.config['random_state']
        )
        self.model.fit(numeric_data)
        
        return self
    
    def generate(self, n_samples, conditions=None):
        samples, _ = self.model.sample(n_samples)
        return pd.DataFrame(samples, columns=self.columns)
    
    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'columns': self.columns,
                'config': self.config
            }, f)
    
    @classmethod
    def load(cls, path):
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(config=data['config'])
        instance.model = data['model']
        instance.columns = data['columns']
        return instance
```

Usage:

```python
from genesis import SyntheticGenerator

# Use the custom plugin
generator = SyntheticGenerator(
    method='industry_specific',
    config={'n_components': 20}
)

generator.fit(transaction_data)
synthetic = generator.generate(10000)
```

## Best Practices

1. **Follow the interface** - Implement all required methods
2. **Declare dependencies** - List all external packages
3. **Validate config** - Use schema for configuration
4. **Handle errors gracefully** - Raise appropriate exceptions
5. **Write tests** - Use PluginTestCase for testing
6. **Document** - Include docstrings and usage examples

## Next Steps

- **[Distributed Generation](/docs/advanced/distributed)** - Scale with plugins
- **[GPU Acceleration](/docs/advanced/gpu)** - GPU-enabled plugins
- **[API Reference](/docs/api/reference)** - Plugin base classes
