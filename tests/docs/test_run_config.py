"""Test script for run_config guide examples.

Tests the code examples from docs/howtos/customizations/run_config.md
"""

from dotenv import load_dotenv

load_dotenv()


def test_openai_client_configuration():
    """Test OpenAI client with timeout and retries."""
    from openai import AsyncOpenAI

    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    # Configure timeout and retries on the client
    client = AsyncOpenAI(
        timeout=60.0,  # 60 second timeout
        max_retries=5,  # Retry up to 5 times on failures
    )

    llm = llm_factory("gpt-4o-mini", client=client)

    # Use with metrics
    scorer = Faithfulness(llm=llm)
    result = scorer.score(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        retrieved_contexts=[
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
        ],
    )

    assert result.value is not None
    print(f"✓ Faithfulness Score: {result.value}")


def test_fine_grained_timeout_control():
    """Test httpx.Timeout for fine-grained control."""
    import httpx
    from openai import AsyncOpenAI

    from ragas.llms import llm_factory

    client = AsyncOpenAI(
        timeout=httpx.Timeout(
            60.0,  # Total timeout
            connect=5.0,  # Connection timeout
            read=30.0,  # Read timeout
            write=10.0,  # Write timeout
        ),
        max_retries=3,
    )

    llm = llm_factory("gpt-4o-mini", client=client)
    assert llm is not None
    print(f"✓ LLM with httpx timeout created: {llm}")


if __name__ == "__main__":
    print("Test 1: OpenAI Client Configuration")
    test_openai_client_configuration()

    print("\nTest 2: Fine-Grained Timeout Control")
    test_fine_grained_timeout_control()

    print("\n✅ All tests passed!")
