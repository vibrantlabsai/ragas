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


def test_infer_embedding_provider_maps_abliteration_to_openai():
    """Test that Abliteration LLMs fall back to OpenAI embeddings."""
    from ragas.embeddings.base import _infer_embedding_provider_from_llm

    class MockLLM:
        provider = "abliteration"

    assert _infer_embedding_provider_from_llm(MockLLM()) == "openai"
