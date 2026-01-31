# ADR-0011: Plugin Architecture with Decorator-Based Registration

## Status

Accepted

## Context

As Genesis matured, users requested the ability to extend the platform with custom generators, transformers, evaluators, and constraints without modifying core code. We needed an extension mechanism that:

1. **Low barrier to entry**: Data scientists should add custom generators without understanding internals
2. **Runtime discovery**: Plugins should be loadable without restarting applications
3. **Type safety**: Registration should validate that plugins implement required interfaces
4. **Discoverability**: Users should be able to list available plugins programmatically

Several patterns were considered:

- **Entry points** (setuptools): Standard but requires package installation
- **Configuration files**: Explicit but verbose and error-prone
- **Decorator-based registration**: Pythonic and familiar to Flask/FastAPI users
- **Metaclass registration**: Automatic but magical and hard to debug

## Decision

We implement a **decorator-based plugin system** with a singleton registry:

```python
from genesis.plugins import register_generator, get_generator

@register_generator("my_generator", description="Custom generator for specific use case")
class MyGenerator(BaseGenerator):
    def _fit_impl(self, data, discrete_columns, progress_callback):
        # Custom fitting logic
        pass
    
    def _generate_impl(self, n_samples, conditions, progress_callback):
        # Custom generation logic
        return synthetic_data

# Use the registered generator
gen_class = get_generator("my_generator")
gen = gen_class()
```

Key design elements:

1. **Singleton `PluginRegistry`**: Global state ensures plugins registered anywhere are available everywhere
2. **Type-specific decorators**: `@register_generator`, `@register_transformer`, `@register_evaluator`, `@register_constraint`
3. **Metadata capture**: Name, description, version, author, and tags stored with each plugin
4. **Discovery function**: `discover_plugins(path)` loads all modules from a directory

```python
class PluginRegistry:
    _instance: Optional["PluginRegistry"] = None
    
    def __new__(cls) -> "PluginRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._plugins = {pt: {} for pt in PluginType}
        return cls._instance
```

## Consequences

### Positive

- **Familiar pattern**: Mirrors Flask's `@app.route()` and FastAPI's dependency injection
- **Self-documenting**: Plugin metadata (description, version) is captured at registration
- **Lazy loading**: Plugins are only instantiated when requested via `get_generator()`
- **Namespace isolation**: Plugin names are scoped by type (generator vs. transformer can share names)
- **Hot reload capable**: In development, plugins can be re-registered without restart

### Negative

- **Global state**: Singleton pattern makes testing harder; must reset registry between tests
- **Import side effects**: Importing a module with `@register_generator` modifies global state
- **Name collisions**: Two plugins with same name silently overwrite (logged as warning)
- **No dependency management**: Plugins cannot declare dependencies on other plugins

### Mitigations

- Registry provides `clear()` method for test isolation
- Warning logged when overwriting existing plugin registration
- `PluginInfo` dataclass captures all metadata for introspection
- Future: Consider adding plugin dependency graph

## Plugin Types

| Type | Decorator | Base Class | Purpose |
|------|-----------|------------|---------|
| Generator | `@register_generator` | `BaseGenerator` | Custom synthesis methods |
| Transformer | `@register_transformer` | `BaseTransformer` | Data preprocessing |
| Evaluator | `@register_evaluator` | `BaseEvaluator` | Quality metrics |
| Constraint | `@register_constraint` | `BaseConstraint` | Data validation rules |
| Callback | `@register_callback` | `BaseCallback` | Training hooks |

## Examples

```python
# List all registered generators
from genesis.plugins import list_generators

for plugin in list_generators():
    print(f"{plugin.name}: {plugin.description}")

# Discover plugins from directory
from genesis.plugins import discover_plugins

n_loaded = discover_plugins("./my_plugins", prefix="my_plugins")
print(f"Loaded {n_loaded} plugin modules")

# Use in SyntheticGenerator with 'auto' detection
from genesis import SyntheticGenerator

# If "my_generator" is registered, it becomes available
gen = SyntheticGenerator(method="my_generator")
```
