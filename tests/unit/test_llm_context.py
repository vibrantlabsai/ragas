#!/usr/bin/env python3
"""Test llm_context feature with calculation-based Pell Grant questions"""

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona

load_dotenv()


def main():
    # Create documents from hardcoded text (no PDF needed!)
    pell_grant_text = """
    Federal Pell Grant Program Overview
    
    The Federal Pell Grant is a need-based grant for undergraduate students. The maximum Pell Grant for the 2023-2024 award year is $7,395. The minimum Pell Grant is $750.
    
    Scheduled Award Calculation:
    The Scheduled Award is calculated using the Student Aid Index (SAI) and Cost of Attendance (COA). 
    Formula: Scheduled Award = min(max_pell, Pell_COA - SAI)
    Where Pell_COA is the institution's cost of attendance for Pell purposes.
    
    Example 1: If a student's SAI is $1,004 and the Pell COA is $6,493, and the maximum Pell is $7,500:
    Scheduled Award = min($7,500, $6,493 - $1,004) = min($7,500, $5,489) = $5,489
    
    Enrollment Intensity:
    Full-time enrollment is typically 12 credit hours or more per semester. Part-time enrollment affects the actual disbursement amount.
    Formula: Actual Disbursement = Scheduled Award × Enrollment Intensity Percentage
    
    Example 2: If a student has a Scheduled Award of $6,200 and is enrolled at 75% intensity (9 credit hours):
    Actual Disbursement = $6,200 × 0.75 = $4,650
    
    Lifetime Eligibility Used (LEU):
    Students can receive Pell Grants for up to 600% of their Scheduled Award across their lifetime (equivalent to 6 years of full-time enrollment).
    Each semester's usage is calculated as: (Actual Disbursement / Scheduled Award) × 100%
    
    Example 3: If a student receives $3,000 from a Scheduled Award of $6,000:
    LEU used = ($3,000 / $6,000) × 100% = 50%
    If their previous LEU was 450%, remaining LEU = 600% - 450% - 50% = 100%
    
    Consortium Agreements:
    When students take courses at multiple institutions, credit hours are combined to determine enrollment intensity.
    Semester hours are the standard. Quarter hours are converted: Quarter Hours × 0.667 = Semester Hours
    
    Example 4: A student takes 6 semester hours at home school and 4 quarter hours at another school:
    Converted quarter hours = 4 × 0.667 = 2.67 semester hours
    Total = 6 + 2.67 = 8.67 semester hours
    
    Recalculation Upon Withdrawal:
    If a student withdraws, the Pell Grant may need to be recalculated based on the percentage of the payment period completed.
    Formula: Earned Amount = Scheduled Award × Percentage Completed
    Amount to Return = Disbursed Amount - Earned Amount
    
    Example 5: Student withdraws after completing 40% of term with $4,800 Scheduled Award:
    Earned = $4,800 × 0.40 = $1,920
    If $4,800 was disbursed: Return = $4,800 - $1,920 = $2,880
    
    Minimum Award Rule:
    The minimum Pell Grant award is $750. If calculations result in less than $750, the student receives $0.
    
    Rounding Rules:
    All Pell Grant disbursements must be rounded down to whole dollars. No cents are allowed in Pell payments.
    
    Example 6: If calculation results in $3,456.78, the disbursement is $3,456.
    """

    # Use single document to minimize async complexity
    docs = [
        Document(
            page_content=pell_grant_text,
            metadata={"source": "pell_grant_doc", "page": 1},
        )
    ]

    print(f"Created {len(docs)} document from Pell Grant text")

    # Setup models
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0.1))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    # Create minimal personas (only 1 to reduce concurrent API calls)
    personas = [
        Persona(
            name="Financial Aid Officer",
            role_description="A financial aid officer who needs to calculate Pell Grant awards accurately using specific formulas and numerical examples",
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
        llm_context=llm_context,  # 🆕 WITH CONTEXT for calculation questions!
    )

    # Minimal transforms (workaround for ragas headline bug)
    from ragas.testset.transforms import (
        CosineSimilarityBuilder,
        EmbeddingExtractor,
        OverlapScoreBuilder,
    )
    from ragas.testset.transforms.extractors.llm_based import NERExtractor

    minimal_transforms = [
        EmbeddingExtractor(embedding_model=generator_embeddings),
        NERExtractor(llm=generator_llm),
        CosineSimilarityBuilder(),
        OverlapScoreBuilder(),
    ]

    # Use all docs
    num_docs = len(docs)

    # IMPORTANT: Using minimal settings to avoid Python 3.11 async event loop bug
    # - 1 persona (not 2)
    # - 1 document (not 3)
    # - testset_size=1 (not 2)
    # - max_workers=1 (not 3)
    run_config = RunConfig(max_workers=1, max_wait=120)

    dataset_with_context = generator_with_context.generate_with_langchain_docs(
        docs[:num_docs],
        testset_size=1,  # Generate 1 calculation-based question (minimal to avoid async issues)
        transforms=minimal_transforms,
        run_config=run_config,
    )

    print(f"\n✅ Generated {len(dataset_with_context)} queries WITH llm_context!")

    # Convert to dataframe
    df_with_context = dataset_with_context.to_pandas()

    # Display samples
    print("\n" + "=" * 80)
    print("📊 QUESTIONS WITH LLM CONTEXT (calculation-based):")
    print("=" * 80)

    for i, sample in enumerate(dataset_with_context.samples, 1):
        eval_sample = sample.eval_sample
        print(f"\n[{i}] Synthesizer: {sample.synthesizer_name}")
        print(f"Question: {eval_sample.user_input}")
        print(f"Answer: {eval_sample.reference}")
        print("-" * 80)

    print("\n📊 DataFrame Columns:", df_with_context.columns.tolist())
    print(f"📊 DataFrame Shape: {df_with_context.shape}")

    # Compare: Generate WITHOUT llm_context for comparison
    print("\n" + "=" * 80)
    print("🧪 Testing WITHOUT llm_context (generic questions) for comparison...")
    print("=" * 80)

    generator_no_context = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        persona_list=personas,
        # NO llm_context!
    )

    dataset_no_context = generator_no_context.generate_with_langchain_docs(
        docs[:num_docs],
        testset_size=1,  # Generate 1 generic question (minimal to avoid async issues)
        transforms=minimal_transforms,
        run_config=run_config,
    )

    print(f"\n✅ Generated {len(dataset_no_context)} queries WITHOUT llm_context!")

    # Convert to dataframe
    df_no_context = dataset_no_context.to_pandas()

    # Display samples
    print("\n" + "=" * 80)
    print("📊 QUESTIONS WITHOUT LLM CONTEXT (generic):")
    print("=" * 80)

    for i, sample in enumerate(dataset_no_context.samples, 1):
        eval_sample = sample.eval_sample
        print(f"\n[{i}] Synthesizer: {sample.synthesizer_name}")
        print(f"Question: {eval_sample.user_input}")
        print(f"Answer: {eval_sample.reference}")
        print("-" * 80)

    print("\n📊 DataFrame Columns:", df_no_context.columns.tolist())
    print(f"📊 DataFrame Shape: {df_no_context.shape}")

    # Summary Comparison
    print("\n" + "=" * 80)
    print("✅ COMPARISON COMPLETE!")
    print("=" * 80)
    print("\n📊 Summary:")
    print(
        f"   WITH llm_context:    {len(df_with_context)} questions (calculation-based)"
    )
    print(f"   WITHOUT llm_context: {len(df_no_context)} questions (generic)")
    print(
        "\n💡 Notice how llm_context guides the LLM to generate calculation-based questions!"
    )
    print(
        "   Questions WITH context include specific numbers and require calculations."
    )
    print("   Questions WITHOUT context are more generic and factual.")


if __name__ == "__main__":
    main()
