"""
Central registry for the deblend package.

Provides a generic factory pattern for registering and retrieving
components by category and name. All module-level registries delegate
to this single implementation.
"""
from typing import Dict, Type, Any, List, Callable


class Registry:
    """
    Generic component registry.

    Usage::

        @Registry.register('sampler', 'my_sampler')
        class MySampler: ...

        cls = Registry.get('sampler', 'my_sampler')
        instance = cls(**kwargs)
    """

    _registries: Dict[str, Dict[str, Type]] = {}

    @classmethod
    def register(cls, category: str, name: str) -> Callable:
        """Decorator that registers a class under *category*/*name*."""
        def decorator(target_cls: Type) -> Type:
            registry = cls._registries.setdefault(category, {})
            if name in registry:
                raise NameError(
                    f"{category} '{name}' is already registered!")
            registry[name] = target_cls
            return target_cls
        return decorator

    @classmethod
    def get(cls, category: str, name: str) -> Type:
        """Return the class registered under *category*/*name*."""
        registry = cls._registries.get(category, {})
        if name not in registry:
            raise NameError(
                f"{category} '{name}' is not registered. "
                f"Available: {list(registry.keys())}")
        return registry[name]

    @classmethod
    def list(cls, category: str = None) -> Any:
        """List registered names.  If *category* is given return a list,
        otherwise return a dict of all categories."""
        if category is not None:
            return list(cls._registries.get(category, {}).keys())
        return {k: list(v.keys()) for k, v in cls._registries.items()}
