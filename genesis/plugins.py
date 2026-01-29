"""Plugin system for extending Genesis with custom components.

This module provides a decorator-based plugin system for registering:
- Custom generators
- Custom transformers
- Custom evaluators
- Custom constraints

Example:
    >>> from genesis.plugins import register_generator, get_generator
    >>>
    >>> @register_generator("my_generator")
    ... class MyGenerator(BaseGenerator):
    ...     def _fit_impl(self, data, discrete_columns, progress_callback):
    ...         # Custom fitting logic
    ...         pass
    ...
    ...     def _generate_impl(self, n_samples, conditions, progress_callback):
    ...         # Custom generation logic
    ...         pass
    >>>
    >>> # Use the registered generator
    >>> gen = get_generator("my_generator")()
"""

import importlib
import pkgutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from genesis.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class PluginType(Enum):
    """Types of plugins supported by Genesis."""

    GENERATOR = "generator"
    TRANSFORMER = "transformer"
    EVALUATOR = "evaluator"
    CONSTRAINT = "constraint"
    CALLBACK = "callback"


@dataclass
class PluginInfo:
    """Metadata about a registered plugin."""

    name: str
    plugin_type: PluginType
    cls: Type
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.plugin_type.value,
            "class": f"{self.cls.__module__}.{self.cls.__name__}",
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
        }


class PluginRegistry:
    """Central registry for all Genesis plugins.

    This is a singleton that maintains registrations across the application.
    """

    _instance: Optional["PluginRegistry"] = None

    def __new__(cls) -> "PluginRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._plugins: Dict[PluginType, Dict[str, PluginInfo]] = {
                plugin_type: {} for plugin_type in PluginType
            }
            cls._instance._initialized = False
        return cls._instance

    def register(
        self,
        name: str,
        plugin_type: PluginType,
        cls: Type,
        description: str = "",
        version: str = "1.0.0",
        author: str = "",
        tags: Optional[List[str]] = None,
    ) -> None:
        """Register a plugin.

        Args:
            name: Unique name for the plugin
            plugin_type: Type of plugin
            cls: The plugin class
            description: Human-readable description
            version: Plugin version
            author: Plugin author
            tags: Searchable tags
        """
        if name in self._plugins[plugin_type]:
            logger.warning(f"Overwriting existing plugin: {plugin_type.value}/{name}")

        info = PluginInfo(
            name=name,
            plugin_type=plugin_type,
            cls=cls,
            description=description or cls.__doc__ or "",
            version=version,
            author=author,
            tags=tags or [],
        )

        self._plugins[plugin_type][name] = info
        logger.debug(f"Registered plugin: {plugin_type.value}/{name}")

    def get(self, name: str, plugin_type: PluginType) -> Optional[Type]:
        """Get a plugin class by name and type.

        Args:
            name: Plugin name
            plugin_type: Plugin type

        Returns:
            Plugin class or None if not found
        """
        info = self._plugins[plugin_type].get(name)
        return info.cls if info else None

    def get_info(self, name: str, plugin_type: PluginType) -> Optional[PluginInfo]:
        """Get plugin info by name and type."""
        return self._plugins[plugin_type].get(name)

    def list_plugins(
        self,
        plugin_type: Optional[PluginType] = None,
        tags: Optional[List[str]] = None,
    ) -> List[PluginInfo]:
        """List registered plugins.

        Args:
            plugin_type: Filter by type (None for all)
            tags: Filter by tags (None for all)

        Returns:
            List of plugin info objects
        """
        plugins = []

        types_to_check = [plugin_type] if plugin_type else list(PluginType)

        for pt in types_to_check:
            for info in self._plugins[pt].values():
                if tags:
                    if any(tag in info.tags for tag in tags):
                        plugins.append(info)
                else:
                    plugins.append(info)

        return plugins

    def unregister(self, name: str, plugin_type: PluginType) -> bool:
        """Unregister a plugin.

        Args:
            name: Plugin name
            plugin_type: Plugin type

        Returns:
            True if plugin was removed, False if not found
        """
        if name in self._plugins[plugin_type]:
            del self._plugins[plugin_type][name]
            logger.debug(f"Unregistered plugin: {plugin_type.value}/{name}")
            return True
        return False

    def clear(self, plugin_type: Optional[PluginType] = None) -> None:
        """Clear all plugins of a type, or all plugins if type is None."""
        if plugin_type:
            self._plugins[plugin_type].clear()
        else:
            for pt in PluginType:
                self._plugins[pt].clear()


# Global registry instance
_registry = PluginRegistry()


def _make_register_decorator(
    plugin_type: PluginType,
    docstring_example: str,
) -> Callable[..., Callable[[Type[T]], Type[T]]]:
    """Factory function to create plugin registration decorators.

    This reduces code duplication across the five register_* functions.

    Args:
        plugin_type: The type of plugin this decorator registers
        docstring_example: Example code for the docstring

    Returns:
        A decorator factory function
    """

    def register_decorator(
        name: str,
        description: str = "",
        version: str = "1.0.0",
        author: str = "",
        tags: Optional[List[str]] = None,
    ) -> Callable[[Type[T]], Type[T]]:
        def decorator(cls: Type[T]) -> Type[T]:
            _registry.register(
                name=name,
                plugin_type=plugin_type,
                cls=cls,
                description=description,
                version=version,
                author=author,
                tags=tags,
            )
            return cls

        return decorator

    # Set docstring dynamically
    register_decorator.__doc__ = f"""Decorator to register a custom {plugin_type.value}.

    Args:
        name: Unique name for the plugin
        description: Human-readable description
        version: Plugin version string
        author: Plugin author
        tags: Searchable tags for discovery

    Returns:
        Decorator that registers the class

    Example:
        {docstring_example}
    """
    return register_decorator


# Create registration decorators using the factory
register_generator = _make_register_decorator(
    PluginType.GENERATOR,
    '''>>> @register_generator("my_gen", description="My custom generator")
        ... class MyGenerator(BaseGenerator):
        ...     pass''',
)

register_transformer = _make_register_decorator(
    PluginType.TRANSFORMER,
    '''>>> @register_transformer("my_transformer")
        ... class MyTransformer(BaseTransformer):
        ...     pass''',
)

register_evaluator = _make_register_decorator(
    PluginType.EVALUATOR,
    '''>>> @register_evaluator("my_evaluator")
        ... class MyEvaluator:
        ...     pass''',
)

register_constraint = _make_register_decorator(
    PluginType.CONSTRAINT,
    '''>>> @register_constraint("my_constraint")
        ... class MyConstraint(BaseConstraint):
        ...     pass''',
)

register_callback = _make_register_decorator(
    PluginType.CALLBACK,
    '''>>> @register_callback("my_callback")
        ... class MyCallback:
        ...     pass''',
)


def get_generator(name: str) -> Optional[Type]:
    """Get a registered generator by name."""
    return _registry.get(name, PluginType.GENERATOR)


def get_transformer(name: str) -> Optional[Type]:
    """Get a registered transformer by name."""
    return _registry.get(name, PluginType.TRANSFORMER)


def get_evaluator(name: str) -> Optional[Type]:
    """Get a registered evaluator by name."""
    return _registry.get(name, PluginType.EVALUATOR)


def get_constraint(name: str) -> Optional[Type]:
    """Get a registered constraint by name."""
    return _registry.get(name, PluginType.CONSTRAINT)


def get_callback(name: str) -> Optional[Type]:
    """Get a registered callback by name."""
    return _registry.get(name, PluginType.CALLBACK)


def list_generators() -> List[PluginInfo]:
    """List all registered generators."""
    return _registry.list_plugins(PluginType.GENERATOR)


def list_transformers() -> List[PluginInfo]:
    """List all registered transformers."""
    return _registry.list_plugins(PluginType.TRANSFORMER)


def list_evaluators() -> List[PluginInfo]:
    """List all registered evaluators."""
    return _registry.list_plugins(PluginType.EVALUATOR)


def list_constraints() -> List[PluginInfo]:
    """List all registered constraints."""
    return _registry.list_plugins(PluginType.CONSTRAINT)


def list_callbacks() -> List[PluginInfo]:
    """List all registered callbacks."""
    return _registry.list_plugins(PluginType.CALLBACK)


def list_all_plugins() -> Dict[str, List[PluginInfo]]:
    """List all registered plugins grouped by type."""
    return {pt.value: _registry.list_plugins(pt) for pt in PluginType}


def discover_plugins(
    package_path: Union[str, Path],
    prefix: str = "",
) -> int:
    """Discover and load plugins from a package directory.

    Args:
        package_path: Path to the plugins package
        prefix: Module prefix for imports

    Returns:
        Number of modules loaded
    """
    path = Path(package_path)
    if not path.exists():
        logger.warning(f"Plugin path does not exist: {path}")
        return 0

    count = 0
    for _importer, modname, _ispkg in pkgutil.iter_modules([str(path)]):
        try:
            full_name = f"{prefix}.{modname}" if prefix else modname
            importlib.import_module(full_name)
            count += 1
            logger.debug(f"Loaded plugin module: {full_name}")
        except Exception as e:
            logger.error(f"Failed to load plugin module {modname}: {e}")

    return count


def get_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    return _registry


__all__ = [
    # Types
    "PluginType",
    "PluginInfo",
    "PluginRegistry",
    # Decorators
    "register_generator",
    "register_transformer",
    "register_evaluator",
    "register_constraint",
    "register_callback",
    # Getters
    "get_generator",
    "get_transformer",
    "get_evaluator",
    "get_constraint",
    "get_callback",
    # Listers
    "list_generators",
    "list_transformers",
    "list_evaluators",
    "list_constraints",
    "list_callbacks",
    "list_all_plugins",
    # Utilities
    "discover_plugins",
    "get_registry",
]
