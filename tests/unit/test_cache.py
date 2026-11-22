import asyncio
import hashlib
import json

import pytest

from ragas import cacher
from ragas.cache import (
    EXCLUDE_KEYS,
    DiskCacheBackend,
    _generate_cache_key,
    _make_hashable,
)


@pytest.fixture(scope="function")
def temp_cache_dir(tmp_path):
    """Use a temporary directory for caching."""
    return str(tmp_path)


@pytest.fixture(scope="function")
def cache_backend(temp_cache_dir):
    """Provide a DiskCacheBackend instance with a temporary directory."""
    return DiskCacheBackend(cache_dir=temp_cache_dir)


def test_make_hashable():
    """Test that _make_hashable converts various objects into a hashable structure."""
    data = {"tuple": (1, 2), "list": [3, 4], "set": {5, 6}, "dict": {"a": 1, "b": 2}}
    result = _make_hashable(data)
    assert isinstance(result, tuple)
    assert len(result) == len(data)


def test_generate_cache_key():
    """Test that cache keys change when arguments or kwargs differ."""

    def sample_func(a, b):
        return a + b

    key1 = _generate_cache_key(sample_func, (1, 2), {})
    key2 = _generate_cache_key(sample_func, (2, 2), {})
    assert key1 != key2, "Cache keys should differ for different args"

    key3 = _generate_cache_key(sample_func, (1, 2), {"c": 3})
    assert key1 != key3, "Cache keys should differ if kwargs differ"


def test_generate_cache_key_bound_method():
    """Test that cache keys stay the same, when caching bound methods of different objects."""

    class Clazz:
        def __init__(self, irrelevant):
            self.irrelevant = irrelevant

        def sample_func(self, a, b):
            return a + b

    object = Clazz(irrelevant=1)
    object2 = Clazz(irrelevant=2)

    key1 = _generate_cache_key(object.sample_func, (1, 2), {})
    key2 = _generate_cache_key(object2.sample_func, (1, 2), {})
    assert key1 == key2, (
        "Cache keys should match even if the originating objects the methods are bound to are not the same, as long as the arguments match"
    )


def test_no_cache_backend():
    """Test that if no cache backend is provided, results are not cached."""
    call_count = {"count": 0}

    @cacher(cache_backend=None)
    def no_cache_func():
        call_count["count"] += 1
        return call_count["count"]

    # Each call should increment count since caching is disabled
    val1 = no_cache_func()
    val2 = no_cache_func()
    assert val2 == val1 + 1, "Without a cache backend, calls should not be cached."


def test_caching_with_cache_backend(cache_backend):
    """Test that providing a cache backend enables caching."""
    call_count = {"count": 0}

    @cacher(cache_backend=cache_backend)
    def expensive_function():
        call_count["count"] += 1
        return "expensive_result"

    # First call: should run the function
    result1 = expensive_function()
    assert result1 == "expensive_result"
    assert call_count["count"] == 1

    # Second call with same args: should return cached result, not increment call_count
    result2 = expensive_function()
    assert result2 == "expensive_result"
    assert call_count["count"] == 1, "Call count should not increase on cached result"


@pytest.mark.asyncio
async def test_async_caching_with_cache_backend(cache_backend):
    """Test that caching works for async functions when a backend is provided."""
    call_count = {"count": 0}

    @cacher(cache_backend=cache_backend)
    async def async_expensive_function(x):
        call_count["count"] += 1
        await asyncio.sleep(0.1)
        return x * 2

    # First call: should run the function
    result1 = await async_expensive_function(10)
    assert result1 == 20
    assert call_count["count"] == 1

    # Second call with same args: should return cached result
    result2 = await async_expensive_function(10)
    assert result2 == 20
    assert call_count["count"] == 1, "Should have come from cache"


@pytest.mark.filterwarnings("ignore:.*coroutine.*was never awaited:RuntimeWarning")
def test_caching_with_different_args(cache_backend):
    """Test that different arguments produce different cache entries."""
    call_count = {"count": 0}

    @cacher(cache_backend=cache_backend)
    def multiply(x, y):
        call_count["count"] += 1
        return x * y

    assert multiply(2, 3) == 6
    assert multiply(2, 3) == 6
    # Same arguments, should have cached
    assert call_count["count"] == 1

    # Different arguments, cache miss
    assert multiply(3, 3) == 9
    assert call_count["count"] == 2


# =============================================================================
# Cache identity (issue #2983)
#
# `cacher` is applied to already-bound methods at construction time, so `self`
# never appears in `args`. Without an explicit identity, two wrappers of the
# same class differing only in which model they call share cache entries, and
# the second model silently receives the first model's response.
# =============================================================================


def _legacy_cache_key(func, args, kwargs):
    """Recompute the key format used before cache identities existed.

    Deliberately duplicated here rather than imported, so that a change to the
    key format is caught by the backward-compatibility test below instead of
    silently tracking the implementation.
    """
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in EXCLUDE_KEYS}
    key_data = {
        "function": func.__qualname__,
        "args": _make_hashable(args),
        "kwargs": _make_hashable(filtered_kwargs),
    }
    key_string = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(key_string.encode("utf-8")).hexdigest()


class IdentifiedCallable:
    """Mirrors how the LLM/embedding wrappers wrap a bound method.

    The wrapping deliberately happens *before* the identity attributes are
    assigned, which is the ordering every real wrapper constructor uses (they
    call `super().__init__(cache=...)` first). This is why the identity must be a
    callable resolved per call rather than a mapping resolved at decoration time.
    """

    def __init__(self, name, cache, provider="p", model_args=None, system_prompt=None):
        self.calls = 0
        self.echo = cacher(cache_backend=cache, cache_identity=self._cache_identity)(
            self.echo
        )
        self.aecho = cacher(cache_backend=cache, cache_identity=self._cache_identity)(
            self.aecho
        )
        # assigned only after wrapping, on purpose
        self.name = name
        self.provider = provider
        self.model_args = model_args
        self.system_prompt = system_prompt

    def _cache_identity(self):
        return {
            "class": type(self).__name__,
            "provider": self.provider,
            "model": self.name,
            "model_args": self.model_args,
            "system_prompt": self.system_prompt,
        }

    def echo(self, prompt):
        self.calls += 1
        return f"{self.name}:{prompt}"

    async def aecho(self, prompt):
        self.calls += 1
        return f"{self.name}:{prompt}"


def test_generate_cache_key_without_identity_is_backward_compatible():
    """No identity must reproduce the pre-identity key exactly.

    Existing on-disk caches are keyed by that hash, so adding an unconditional
    `"cache_identity": null` entry would silently invalidate all of them.
    """

    def sample_func(a, b):
        return a + b

    expected = _legacy_cache_key(sample_func, (1, 2), {"c": 3})

    assert _generate_cache_key(sample_func, (1, 2), {"c": 3}) == expected
    assert _generate_cache_key(sample_func, (1, 2), {"c": 3}, None) == expected
    # an empty identity is treated as absent, so it stays byte-identical too
    assert _generate_cache_key(sample_func, (1, 2), {"c": 3}, {}) == expected
    assert _generate_cache_key(sample_func, (1, 2), {"c": 3}, lambda: {}) == expected


def test_generate_cache_key_excluded_kwargs_still_ignored_with_identity():
    """EXCLUDE_KEYS must keep being honoured when an identity is supplied."""

    def sample_func(a):
        return a

    identity = {"model": "a"}
    with_cb = _generate_cache_key(sample_func, (1,), {"callbacks": object()}, identity)
    without_cb = _generate_cache_key(sample_func, (1,), {}, identity)
    assert with_cb == without_cb


def test_generate_cache_key_identity_partitions_keys():
    """Different identities must produce different keys, equal ones must match."""

    def sample_func(a):
        return a

    key_a = _generate_cache_key(sample_func, (1,), {}, {"model": "a"})
    key_b = _generate_cache_key(sample_func, (1,), {}, {"model": "b"})
    key_none = _generate_cache_key(sample_func, (1,), {})

    assert key_a != key_b, "different models must not share a cache entry"
    assert key_a != key_none and key_b != key_none

    # insertion order of the identity is irrelevant
    assert _generate_cache_key(
        sample_func, (1,), {}, {"a": 1, "b": 2}
    ) == _generate_cache_key(sample_func, (1,), {}, {"b": 2, "a": 1})

    # the mapping form and the callable form agree
    assert _generate_cache_key(
        sample_func, (1,), {}, {"model": "a"}
    ) == _generate_cache_key(sample_func, (1,), {}, lambda: {"model": "a"})


def test_generate_cache_key_identity_value_repr_is_stable():
    """An object with a default repr must not embed its heap address.

    `json.dumps(default=str)` would otherwise put `<X object at 0x...>` into the
    key, changing it every process and silently disabling the cache for good.
    """

    def sample_func(a):
        return a

    class Client:
        pass

    key1 = _generate_cache_key(sample_func, (1,), {}, {"client": Client()})
    key2 = _generate_cache_key(sample_func, (1,), {}, {"client": Client()})
    assert key1 == key2

    # nested in a dict/list too, since model_args can hold such objects
    nested1 = _generate_cache_key(sample_func, (1,), {}, {"a": {"c": [Client()]}})
    nested2 = _generate_cache_key(sample_func, (1,), {}, {"a": {"c": [Client()]}})
    assert nested1 == nested2


def test_cacher_different_model_does_not_reuse_cached_response(cache_backend):
    """The bug from issue #2983: two models sharing a backend must not collide."""
    model_a = IdentifiedCallable("model-a", cache_backend)
    model_b = IdentifiedCallable("model-b", cache_backend)

    assert model_a.echo("same prompt") == "model-a:same prompt"
    assert model_b.echo("same prompt") == "model-b:same prompt"

    assert (model_a.calls, model_b.calls) == (
        1,
        1,
    ), "each model must execute rather than read the other model's entry"


def test_cacher_same_identity_hits_across_instances(cache_backend):
    """Two instances with identical configuration should still share entries."""
    first = IdentifiedCallable("model-a", cache_backend)
    second = IdentifiedCallable("model-a", cache_backend)

    assert first.echo("prompt") == "model-a:prompt"
    assert second.echo("prompt") == "model-a:prompt"

    assert first.calls == 1
    assert second.calls == 0, "equal configuration must reuse the cached response"


def test_cacher_repeat_calls_on_same_instance_hit(cache_backend):
    """The identity must hold no per-call state, or repeat hits break.

    `IdentifiedCallable.calls` increments on every execution and is deliberately
    absent from the identity; including it would change the key on every call.
    """
    llm = IdentifiedCallable("model-a", cache_backend)

    for _ in range(3):
        assert llm.echo("prompt") == "model-a:prompt"

    assert llm.calls == 1


@pytest.mark.asyncio
async def test_cacher_identity_applies_to_async_methods(cache_backend):
    """Async wrappers must partition and hit exactly like the sync ones."""
    model_a = IdentifiedCallable("model-a", cache_backend)
    model_b = IdentifiedCallable("model-b", cache_backend)

    assert await model_a.aecho("prompt") == "model-a:prompt"
    assert await model_b.aecho("prompt") == "model-b:prompt"
    assert (model_a.calls, model_b.calls) == (1, 1)

    # a repeat call on the same instance is a hit
    assert await model_a.aecho("prompt") == "model-a:prompt"
    assert model_a.calls == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("provider", "other-provider"),
        ("model_args", {"temperature": 0.9}),
        ("system_prompt", "you are terse"),
    ],
)
def test_cacher_each_identity_field_partitions(cache_backend, field, value):
    """Every field listed in an identity must actually reach the key."""
    baseline = IdentifiedCallable("model-a", cache_backend)
    changed = IdentifiedCallable("model-a", cache_backend, **{field: value})

    assert baseline.echo("prompt") == "model-a:prompt"
    changed.echo("prompt")

    assert changed.calls == 1, f"changing {field} must not hit the cached entry"


def test_cacher_identity_resolution_error_propagates(cache_backend):
    """A broken identity must fail loudly, never fall back to a colliding key."""

    def boom():
        raise RuntimeError("identity unavailable")

    @cacher(cache_backend=cache_backend, cache_identity=boom)
    def fn(x):
        return x

    with pytest.raises(RuntimeError, match="identity unavailable"):
        fn(1)
