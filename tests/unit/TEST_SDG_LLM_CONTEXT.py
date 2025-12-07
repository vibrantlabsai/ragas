#!/usr/bin/env python3
"""Test llm_context feature with calculation-based Pell Grant questions"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
#from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

load_dotenv()

def main():
    # Load PDFs
    path = "TESTS/UNIT/TEST_SDG_LLM_CONTEXT/"
    loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader ) # PyMuPDFLoader
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from Pell Grant PDF")

    # Setup models
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0.1))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    # Create personas for financial aid context
    personas = [
        Persona(
            name="Financial Aid Officer",
            role_description="A financial aid officer who needs to calculate Pell Grant awards accurately using specific formulas and numerical examples"
        ),
        Persona(
            name="Student Aid Counselor",
            role_description="A counselor helping students understand their Pell Grant eligibility with real numerical scenarios"
        )
    ]

    # LLM Context for generating calculation-based questions
    llm_context = """
Generate ONLY Calculation/Application Questions. 
These questions must require applying the Pell Grant formulas and rules from the document to a specific scenario in order to:
    • calculate a numerical outcome (e.g., award amount, disbursement, enrollment intensity)

Examples:
- "A student's calculated SAI is 1,004 and their Pell COA is $6,493. If the maximum Pell is $7,500 and the minimum Pell is $750, what would be the student's Scheduled Award?"
- "A student has a Scheduled Award of $6,200 and an enrollment intensity of 75%. What would be their actual Pell Grant disbursement?"
- "If a student's LEU is 450% and they receive a Pell Grant of $3,000 (representing 50% of their Scheduled Award), what is their remaining eligibility in percentage?"
- "A student is taking 6 semester hours at their home school and 4 quarter hours at a different school under a consortium agreement. What would be the total semester hours for determining enrollment intensity?"
- "A student has a Scheduled Award of $5,000 and a current LEU of 500%. If the school only disburses in whole dollars, what is the maximum Pell Grant amount the student is eligible to receive for the remaining eligibility?"
- "If a student withdraws after completing 40% of the payment period with a Scheduled Award of $4,800, what amount should be returned?"

Requirements:
- Don't combine multiple questions in one question.
- ALL questions MUST include specific numbers and amounts from the document when possible (e.g., SAI of 1,004; Pell COA of $6,493; max Pell of $7,500; min Pell of $750).
- Questions MUST require calculation or application of Pell Grant formulas.
- Use realistic SAI amounts ($0-$6,000), Pell amounts ($750-$7,500), and percentages.
- Avoid simple factual questions like "What is a Pell Grant?" or "What is SAI?"
- Focus on practical scenarios that students or financial aid officers would encounter.
- Extract actual numbers from examples in the document whenever possible.
- Never generate repetitive questions.

Answers should show the calculation steps and final numerical result.

"""

    print("\n🎯 Testing WITH llm_context (calculation-based questions)...")
    print("=" * 80)

    # Generator WITH llm_context
    generator_with_context = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        persona_list=personas,
        llm_context=llm_context  # 🆕 WITH CONTEXT for calculation questions!
    )

    # Minimal transforms (workaround for ragas headline bug)
    from ragas.testset.transforms import EmbeddingExtractor, CosineSimilarityBuilder, OverlapScoreBuilder
    from ragas.testset.transforms.extractors.llm_based import NERExtractor

    minimal_transforms = [
        EmbeddingExtractor(embedding_model=generator_embeddings),
        NERExtractor(llm=generator_llm),
        CosineSimilarityBuilder(),
        OverlapScoreBuilder(),
    ]

    # Use up to 10 docs for better coverage
    num_docs = min(10, len(docs))

    # Set run_config to limit concurrency
    run_config = RunConfig(max_workers=3, max_wait=60)

    dataset_with_context = generator_with_context.generate_with_langchain_docs(
        docs[:num_docs],
        testset_size=4,  # Generate 4 calculation-based questions
        transforms=minimal_transforms,
        run_config=run_config
    )

    print(f"\n✅ Generated {len(dataset_with_context)} queries WITH llm_context!")

    # Display samples
    print("\n" + "=" * 80)
    print("📊 SAMPLE QUESTIONS (WITH LLM CONTEXT):")
    print("=" * 80)

    for i, sample in enumerate(dataset_with_context.samples[:4], 1):
        eval_sample = sample.eval_sample
        print(f"\n[{i}] Synthesizer: {sample.synthesizer_name}")
        print(f"Question: {eval_sample.user_input}")
        print(f"Answer: {eval_sample.reference[:300]}...")
        print("-" * 80)

    # Save to CSV
    df = dataset_with_context.to_pandas()
    output_file = "SDG_with_LLM_CONTEXT.csv"
    df.to_csv(output_file, index=False)

    print(f"\n💾 Saved to: {output_file}")
    print(f"\n✨ Notice: Questions include specific numbers and require calculations!")

    # Compare: Generate WITHOUT llm_context for comparison
    print("\n" + "=" * 80)
    print("🧪 Testing WITHOUT llm_context (generic questions) for comparison...")
    print("=" * 80)

    generator_no_context = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        persona_list=personas
        # NO llm_context!
    )

    dataset_no_context = generator_no_context.generate_with_langchain_docs(
        docs[:num_docs],
        testset_size=4,
        transforms=minimal_transforms,
        run_config=run_config
    )

    print(f"\n✅ Generated {len(dataset_no_context)} queries WITHOUT llm_context!")

    # Display samples
    print("\n" + "=" * 80)
    print("📊 SAMPLE QUESTIONS (WITHOUT LLM CONTEXT):")
    print("=" * 80)

    for i, sample in enumerate(dataset_no_context.samples[:4], 1):
        eval_sample = sample.eval_sample
        print(f"\n[{i}] Synthesizer: {sample.synthesizer_name}")
        print(f"Question: {eval_sample.user_input}")
        print(f"Answer: {eval_sample.reference[:300]}...")
        print("-" * 80)

    # Save to CSV
    df_no_context = dataset_no_context.to_pandas()
    output_file_no_context = "SDG_without_LLM_CONTEXT.csv"
    df_no_context.to_csv(output_file_no_context, index=False)

    print(f"\n💾 Saved to: {output_file_no_context}")

    # Summary
    print("\n" + "=" * 80)
    print("✅ COMPARISON COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Output Files:")
    print(f"   WITH llm_context:    {output_file}")
    print(f"   WITHOUT llm_context: {output_file_no_context}")
    print(f"\n💡 Compare the two files to see how llm_context guides question generation!")

if __name__ == "__main__":
    main()
