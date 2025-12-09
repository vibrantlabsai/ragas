"""Test code examples from modifying-prompts-metrics.md guide."""

import asyncio

from dotenv import load_dotenv

load_dotenv()


async def test_access_prompts():
    """Test accessing prompts on Faithfulness metric."""
    from openai import AsyncOpenAI

    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)

    scorer = Faithfulness(llm=llm)

    # Verify the prompts exist
    assert hasattr(scorer, "statement_generator_prompt")
    assert hasattr(scorer, "nli_statement_prompt")
    print("✓ Faithfulness has statement_generator_prompt and nli_statement_prompt")


async def test_generate_prompt_string():
    """Test generating prompt string from StatementGeneratorPrompt."""
    from openai import AsyncOpenAI

    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness
    from ragas.metrics.collections.faithfulness.util import StatementGeneratorInput

    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)

    scorer = Faithfulness(llm=llm)

    sample_input = StatementGeneratorInput(
        question="What is the Eiffel Tower?",
        answer="The Eiffel Tower is located in Paris.",
    )

    prompt_string = scorer.statement_generator_prompt.to_string(sample_input)
    assert "Eiffel Tower" in prompt_string
    assert "Paris" in prompt_string
    print("✓ Generated prompt string successfully")
    print(f"Prompt preview: {prompt_string[:200]}...")


async def test_custom_factual_correctness_prompt():
    """Test customizing FactualCorrectness prompt and running evaluation."""
    from openai import AsyncOpenAI

    from ragas.llms import llm_factory
    from ragas.metrics.collections import FactualCorrectness
    from ragas.metrics.collections.factual_correctness.util import (
        ClaimDecompositionPrompt,
    )

    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)

    class CustomClaimDecompositionPrompt(ClaimDecompositionPrompt):
        instruction = """You are an expert at breaking down complex statements into atomic claims.
Break down the input text into clear, verifiable claims.
Only output valid JSON with a "claims" array."""

    scorer = FactualCorrectness(llm=llm)
    scorer.prompt = CustomClaimDecompositionPrompt()

    # Run evaluation with custom prompt
    result = await scorer.ascore(
        response="The Eiffel Tower is in Paris and was built in 1889.",
        reference="The Eiffel Tower is located in Paris. It was completed in 1889.",
    )

    assert scorer.prompt.instruction.startswith("You are an expert")
    assert result.value is not None
    print(
        f"✓ Custom FactualCorrectness prompt evaluation completed. Score: {result.value}"
    )


async def test_custom_faithfulness_nli_prompt():
    """Test customizing Faithfulness NLI prompt examples and running evaluation."""
    from openai import AsyncOpenAI

    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness
    from ragas.metrics.collections.faithfulness.util import (
        NLIStatementInput,
        NLIStatementOutput,
        NLIStatementPrompt,
        StatementFaithfulnessAnswer,
    )

    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)

    class DomainSpecificNLIPrompt(NLIStatementPrompt):
        examples = [
            (
                NLIStatementInput(
                    context="Machine learning is a field within artificial intelligence that enables systems to learn from data.",
                    statements=[
                        "Machine learning is a subset of AI.",
                        "Machine learning uses statistical techniques.",
                    ],
                ),
                NLIStatementOutput(
                    statements=[
                        StatementFaithfulnessAnswer(
                            statement="Machine learning is a subset of AI.",
                            reason="The context states ML is 'a field within artificial intelligence', supporting this claim.",
                            verdict=1,
                        ),
                        StatementFaithfulnessAnswer(
                            statement="Machine learning uses statistical techniques.",
                            reason="The context doesn't mention statistical techniques.",
                            verdict=0,
                        ),
                    ]
                ),
            ),
        ]

    scorer = Faithfulness(llm=llm)
    scorer.nli_statement_prompt = DomainSpecificNLIPrompt()

    # Run evaluation with custom prompt
    result = await scorer.ascore(
        user_input="How do neural networks work?",
        response="Neural networks are inspired by biological neurons.",
        retrieved_contexts=[
            "Artificial neural networks are computing systems loosely inspired by biological neural networks."
        ],
    )

    assert len(scorer.nli_statement_prompt.examples) >= 1
    assert result.value is not None
    print(
        f"✓ Custom Faithfulness NLI prompt evaluation completed. Score: {result.value}"
    )


async def test_nli_prompt_to_string():
    """Test generating NLI prompt string."""
    from openai import AsyncOpenAI

    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness
    from ragas.metrics.collections.faithfulness.util import NLIStatementInput

    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)

    scorer = Faithfulness(llm=llm)

    sample_input = NLIStatementInput(
        context="Paris is the capital and most populous city of France.",
        statements=["The capital of France is Paris.", "Paris is in Germany."],
    )

    full_prompt = scorer.nli_statement_prompt.to_string(sample_input)
    assert "Paris" in full_prompt
    assert "France" in full_prompt
    print("✓ NLI prompt string generated successfully")


async def test_faithfulness_evaluation():
    """Test running actual Faithfulness evaluation."""
    from openai import AsyncOpenAI

    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)

    scorer = Faithfulness(llm=llm)

    result = await scorer.ascore(
        user_input="Where was Einstein born?",
        response="Einstein was born in Germany.",
        retrieved_contexts=[
            "Albert Einstein was born in Ulm, Germany on March 14, 1879."
        ],
    )

    assert result.value is not None
    print(f"✓ Faithfulness evaluation completed. Score: {result.value}")


async def test_language_adaptation():
    """Test adapting prompts to different languages."""
    from openai import AsyncOpenAI

    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)

    scorer = Faithfulness(llm=llm)

    # Adapt the statement generator prompt to Spanish
    adapted_prompt = await scorer.statement_generator_prompt.adapt(
        target_language="spanish",
        llm=llm,
        adapt_instruction=False,  # Keep instruction in English, only translate examples
    )

    # Replace the prompt with the adapted version
    scorer.statement_generator_prompt = adapted_prompt

    # Verify the adaptation worked
    assert scorer.statement_generator_prompt.language == "spanish"

    # Run evaluation with adapted Spanish prompt (as shown in guide)
    result = await scorer.ascore(
        user_input="¿Dónde nació Einstein?",
        response="Einstein nació en Alemania.",
        retrieved_contexts=["Albert Einstein nació en Alemania..."],
    )
    assert result.value is not None
    print(f"✓ Language adaptation completed. Score: {result.value}")


async def main():
    """Run all tests."""
    print("Testing modifying-prompts-metrics.md examples...\n")

    await test_access_prompts()
    await test_generate_prompt_string()
    await test_custom_factual_correctness_prompt()
    await test_custom_faithfulness_nli_prompt()
    await test_nli_prompt_to_string()
    await test_faithfulness_evaluation()
    await test_language_adaptation()

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
