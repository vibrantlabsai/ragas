from __future__ import annotations


def test_basic_legacy_imports():
    """Test that basic legacy imports work."""
    from ragas.embeddings import BaseRagasEmbeddings, embedding_factory

    assert BaseRagasEmbeddings is not None
    assert embedding_factory is not None


def test_debug_base_module():
    """Debug what's available in base module."""
    import ragas.embeddings.base as base_module

    # Check if BaseRagasEmbedding is in the module
    has_class = hasattr(base_module, "BaseRagasEmbedding")
    print(f"base_module has BaseRagasEmbedding: {has_class}")

    if has_class:
        cls = getattr(base_module, "BaseRagasEmbedding")
        print(f"BaseRagasEmbedding type: {type(cls)}")
        assert cls is not None
    else:
        # List what is available
        attrs = [attr for attr in dir(base_module) if not attr.startswith("_")]
        print(f"Available attributes: {attrs}")
        raise AssertionError("BaseRagasEmbedding not found in base module")


def test_direct_import_from_base():
    """Test direct import from base module."""
    try:
        from ragas.embeddings.base import BaseRagasEmbedding

        print(f"Successfully imported BaseRagasEmbedding: {BaseRagasEmbedding}")
        assert BaseRagasEmbedding is not None
    except ImportError as e:
        print(f"Import error: {e}")
        # Try to import the whole module first
        import ragas.embeddings.base

        print(f"Module imported successfully: {ragas.embeddings.base}")
        # Now try to get the class
        if hasattr(ragas.embeddings.base, "BaseRagasEmbedding"):
            cls = getattr(ragas.embeddings.base, "BaseRagasEmbedding")
            print(f"Found class via getattr: {cls}")
        else:
            print("Class not found via getattr either")
        raise


def test_main_module_import():
    """Test import from main embeddings module."""
    try:
        from ragas.embeddings import RagasBaseEmbedding

        print(f"Successfully imported from main module: {RagasBaseEmbedding}")
        assert RagasBaseEmbedding is not None
    except ImportError as e:
        print(f"Main module import error: {e}")
        # Check what's in the main module
        import ragas.embeddings

        attrs = [
            attr for attr in dir(ragas.embeddings) if "Ragas" in attr or "Base" in attr
        ]
        print(f"Ragas/Base related attributes in main module: {attrs}")
        raise


def test_backward_compatibility_alias():
    """Test that RagasBaseEmbedding works as an alias to BaseRagasEmbedding."""
    from ragas.embeddings import BaseRagasEmbedding, RagasBaseEmbedding

    # They should be the same class
    assert RagasBaseEmbedding is BaseRagasEmbedding
    print("Backward compatibility confirmed: RagasBaseEmbedding is BaseRagasEmbedding")


class _FakeEmbeddingModel:
    """Stand-in for a loaded model object (e.g. fastembed's TextEmbedding)."""

    model_name = "BAAI/bge-small-en-v1.5"


class _EmbeddingsWithObjectModel:
    """Mimics langchain_community's FastEmbedEmbeddings, whose `.model` holds a
    model *object* rather than a string."""

    def __init__(self):
        self.model = _FakeEmbeddingModel()

    def embed_query(self, text):
        return [0.0, 1.0]

    def embed_documents(self, texts):
        return [[0.0, 1.0] for _ in texts]

    async def aembed_query(self, text):
        return [0.0, 1.0]

    async def aembed_documents(self, texts):
        return [[0.0, 1.0] for _ in texts]


def test_get_model_name_coerces_non_string_model():
    """Regression for #2783: a non-string `.model` must not crash usage
    telemetry. `_get_model_name` should return a readable string."""
    from ragas.embeddings.base import _get_model_name

    # string model passes through unchanged
    assert _get_model_name(type("E", (), {"model": "text-embedding-3-small"})()) == (
        "text-embedding-3-small"
    )
    # missing model -> None
    assert _get_model_name(type("E", (), {})()) is None
    # object model -> readable name (prefers model_name), never raises
    assert _get_model_name(_EmbeddingsWithObjectModel()) == "BAAI/bge-small-en-v1.5"

    # object model without a name attribute -> class name, still a str
    class _Bare:
        pass

    result = _get_model_name(type("E", (), {"model": _Bare()})())
    assert isinstance(result, str) and result == "_Bare"


def test_langchain_wrapper_embed_with_object_model_does_not_raise():
    """Regression for #2783: embedding via LangchainEmbeddingsWrapper must not
    raise a telemetry ValidationError when the underlying `.model` is an
    object, which previously silently NaN'd embedding-based metrics."""
    from ragas.embeddings.base import LangchainEmbeddingsWrapper

    wrapper = LangchainEmbeddingsWrapper(_EmbeddingsWithObjectModel())

    # Before the fix these raised pydantic ValidationError from usage telemetry.
    assert wrapper.embed_query("hello") == [0.0, 1.0]
    assert wrapper.embed_documents(["a", "b"]) == [[0.0, 1.0], [0.0, 1.0]]
