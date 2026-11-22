import functools
import hashlib
import inspect
import json
import logging
import re
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Mapping, Optional, Union

from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

logger = logging.getLogger(__name__)

#: An explicit cache namespace: deliberately selected configuration values that
#: must partition cache entries.
CacheIdentity = Mapping[str, Any]

#: Either a cache identity, or a zero-argument callable returning one. The
#: callable form exists because callers wrap bound methods during ``__init__``,
#: before the attributes that make up the identity have been assigned.
CacheIdentityProvider = Union[CacheIdentity, Callable[[], CacheIdentity]]


class CacheInterface(ABC):
    """Abstract base class defining the interface for cache implementations.

    This class provides a standard interface that all cache implementations must follow.
    It supports basic cache operations like get, set and key checking.
    """

    @abstractmethod
    def get(self, key: str) -> Any:
        """Retrieve a value from the cache by key.

        Args:
            key: The key to look up in the cache.

        Returns:
            The cached value associated with the key.
        """
        pass

    @abstractmethod
    def set(self, key: str, value) -> None:
        """Store a value in the cache with the given key.

        Args:
            key: The key to store the value under.
            value: The value to cache.
        """
        pass

    @abstractmethod
    def has_key(self, key: str) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: The key to check for.

        Returns:
            True if the key exists in the cache, False otherwise.
        """
        pass

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """
        Define how Pydantic generates a schema for BaseRagasEmbeddings.
        """
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.is_instance_schema(cls),  # The validator function
        )


class DiskCacheBackend(CacheInterface):
    """A cache implementation that stores data on disk using the diskcache library.

    This cache backend persists data to disk, allowing it to survive between program runs.
    It implements the CacheInterface for use with Ragas caching functionality.

    Args:
        cache_dir (str, optional): Directory where cache files will be stored. Defaults to ".cache".
    """

    def __init__(self, cache_dir: str = ".cache"):
        try:
            from diskcache import Cache
        except ImportError:
            raise ImportError(
                "For using the diskcache backend, please install it with `pip install diskcache`."
            )

        self.cache = Cache(cache_dir)

    def get(self, key: str) -> Any:
        """Retrieve a value from the disk cache by key.

        Args:
            key: The key to look up in the cache.

        Returns:
            The cached value associated with the key, or None if not found.
        """
        return self.cache.get(key)

    def set(self, key: str, value) -> None:
        """Store a value in the disk cache with the given key.

        Args:
            key: The key to store the value under.
            value: The value to cache.
        """
        self.cache.set(key, value)

    def has_key(self, key: str) -> bool:
        """Check if a key exists in the disk cache.

        Args:
            key: The key to check for.

        Returns:
            True if the key exists in the cache, False otherwise.
        """
        return key in self.cache

    def __del__(self):
        """Cleanup method to properly close the cache when the object is destroyed."""
        if hasattr(self, "cache"):
            self.cache.close()

    def __repr__(self):
        """Return string representation of the cache object.

        Returns:
            String showing the cache directory location.
        """
        return f"DiskCacheBackend(cache_dir={self.cache.directory})"


def _make_hashable(o):
    if isinstance(o, (tuple, list)):
        return tuple(_make_hashable(e) for e in o)
    elif isinstance(o, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in o.items()))
    elif isinstance(o, set):
        return tuple(sorted(_make_hashable(e) for e in o))
    elif isinstance(o, BaseModel):
        return _make_hashable(o.model_dump())
    else:
        return o


EXCLUDE_KEYS = ["callbacks"]

# Matches the address in a default object repr, e.g. "<httpx.Client object at 0x7f...>".
_OBJ_REPR_RE = re.compile(r" object at 0x[0-9a-fA-F]+>")


def _stable_identity_value(o):
    """Normalise a cache identity value so it hashes the same in every process.

    Identity values reach the cache key through ``json.dumps(default=str)``. An
    object without a custom ``__repr__`` serialises to something like
    ``<httpx.Client object at 0x7f9c1a2b3c40>``, whose address differs on every
    run - so the key would change every process and the cache would never hit
    again. Such values are replaced by their stable type name instead.

    This is applied only to identities, never to call arguments, so the key
    format for callers that supply no identity is unaffected.
    """
    if isinstance(o, (str, int, float, bool, type(None))):
        return o
    if isinstance(o, Mapping):
        return {k: _stable_identity_value(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_stable_identity_value(e) for e in o]

    text = repr(o)
    if _OBJ_REPR_RE.search(text):
        stable = f"{type(o).__module__}.{type(o).__qualname__}"
        logger.debug(
            "Cache identity value with an unstable repr replaced by %s", stable
        )
        return stable
    return text


def _resolve_cache_identity(
    cache_identity: Optional[CacheIdentityProvider],
) -> Optional[Dict[str, Any]]:
    """Resolve a cache identity to a plain mapping, or None when absent or empty.

    The provider is resolved on *every* cached call, by design. Wrappers pass a
    bound method (e.g. ``self._cache_identity``) while still inside ``__init__``,
    before the attributes it reads have been assigned - resolving it eagerly at
    decoration time would raise ``AttributeError``. Do not "optimise" this into a
    single up-front call.

    An empty identity is treated as absent so that the generated key stays
    byte-identical to the format used before identities existed.
    """
    if cache_identity is None:
        return None

    identity = cache_identity() if callable(cache_identity) else cache_identity
    if not identity:
        return None

    return {k: _stable_identity_value(v) for k, v in identity.items()}


def _make_pydantic_picklable(obj: Any) -> Any:
    """Make Pydantic models returned by instructor library picklable.

    The instructor library dynamically creates new class objects during structured
    output generation. These modified classes have class identity issues that prevent
    pickling. This function detects such instances and recreates them using the
    original class from the module namespace.

    Args:
        obj: Object to make picklable (typically a Pydantic model instance).

    Returns:
        A picklable version of the object. For Pydantic models with class identity
        issues, returns a new instance created with the correct class. Otherwise,
        returns the original object unchanged.
    """
    if isinstance(obj, BaseModel):
        obj_class = obj.__class__
        module = sys.modules.get(obj_class.__module__)

        if module is not None:
            actual_class = getattr(module, obj_class.__name__, None)

            if actual_class is not None and actual_class is not obj_class:
                logger.debug(
                    f"Detected class identity mismatch for {obj_class.__name__}, "
                    f"recreating with actual class from module"
                )
                return actual_class(**obj.model_dump())

    return obj


def _generate_cache_key(func, args, kwargs, cache_identity=None):
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in EXCLUDE_KEYS}

    # This dict must stay byte-identical to the pre-identity format whenever no
    # identity is supplied: existing on-disk caches are keyed by its hash.
    key_data = {
        "function": func.__qualname__,
        "args": _make_hashable(args),
        "kwargs": _make_hashable(filtered_kwargs),
    }

    # Deliberately a conditional insert. Adding "cache_identity": None
    # unconditionally would, under sort_keys=True, put `"cache_identity": null`
    # into the serialised string and re-key every existing cache entry.
    identity = _resolve_cache_identity(cache_identity)
    if identity is not None:
        key_data["cache_identity"] = _make_hashable(identity)

    key_string = json.dumps(key_data, sort_keys=True, default=str)
    cache_key = hashlib.sha256(key_string.encode("utf-8")).hexdigest()
    return cache_key


def cacher(
    cache_backend: Optional[CacheInterface] = None,
    cache_identity: Optional[CacheIdentityProvider] = None,
):
    """Decorator that adds caching functionality to a function.

    This decorator can be applied to both synchronous and asynchronous functions to cache their results.
    If no cache backend is provided, the original function is returned unchanged.

    Args:
        cache_backend (Optional[CacheInterface]): The cache backend to use for storing results.
            If None, caching is disabled.
        cache_identity (Optional[CacheIdentityProvider]): An explicit cache namespace,
            given either as a mapping of deliberately selected configuration values or
            as a zero-argument callable returning one.

            This is needed when caching a *bound method*, because the bound instance
            never appears in ``args`` and so instance configuration cannot otherwise
            reach the cache key. Two objects of the same class that differ only in,
            say, which model they call would otherwise share cache entries.

            The callable form is resolved on every call, which makes it safe to pass a
            bound method (``self._cache_identity``) from inside ``__init__`` before the
            attributes it reads exist.

            When omitted, the generated key is identical to one produced without this
            argument, so existing caches keep working. Pass ``{}`` to opt out
            explicitly.

    Returns:
        Callable: A decorated function that implements caching behavior.

    Note:
        Exceptions raised while resolving ``cache_identity`` are propagated rather
        than swallowed. Falling back to a key without the identity would silently
        reintroduce cross-configuration cache collisions.
    """

    def decorator(func):
        if cache_backend is None:
            return func

        # hack to make pyright happy
        backend: CacheInterface = cache_backend

        is_async = inspect.iscoroutinefunction(func)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_key = _generate_cache_key(func, args, kwargs, cache_identity)

            if backend.has_key(cache_key):
                logger.debug(f"Cache hit for {cache_key}")
                return backend.get(cache_key)

            result = await func(*args, **kwargs)
            picklable_result = _make_pydantic_picklable(result)
            backend.set(cache_key, picklable_result)
            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache_key = _generate_cache_key(func, args, kwargs, cache_identity)

            if backend.has_key(cache_key):
                logger.debug(f"Cache hit for {cache_key}")
                return backend.get(cache_key)

            result = func(*args, **kwargs)
            picklable_result = _make_pydantic_picklable(result)
            backend.set(cache_key, picklable_result)
            return result

        return async_wrapper if is_async else sync_wrapper

    return decorator
