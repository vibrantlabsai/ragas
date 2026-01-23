"""Example demonstrating testset generation using Knowledge Graphs with Ragas.

This example shows how to:
1. Load documents from local directories
2. Create or load a Knowledge Graph for efficient reuse
3. Apply transforms to extract relationships
4. Generate synthetic test datasets with custom personas
5. Save results in JSON format

Prerequisites:
1. Install required dependencies:
   pip install ragas langchain-community langchain-huggingface langchain-openai python-dotenv

2. Set up API credentials:
   - Create a .env file with your GOOGLE_API_KEY
   - Or set the environment variable directly

3. Prepare your data:
   - Organize documents in a directory structure
   - Supported formats: Markdown (.md), Text (.txt)

For more information, see: docs/howtos/testset-generation.md
"""

import os

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona
from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer
from ragas.testset.transforms import apply_transforms, default_transforms


def main():
    """Main function demonstrating Knowledge Graph-based testset generation."""
    # Load environment variables
    load_dotenv()

    # --- Configuration ---
    DIRECTORIES = ["Data/Books-MD/Chemistry/Chemistry-kech2-MD"]
    OUTPUT_FILE = "Ragas/Chemistry-kech2-MD.json"
    KG_OUTPUT_FILE = "Ragas/Chemistry-kech2-MD-knowledge_graph.json"

    # --- Setup LLM and Embeddings ---
    print("\n🚀 Setting up Gemini LLM and HuggingFace embeddings...")

    openai_compatible_llm = ChatOpenAI(
        model="gemini-2.5-flash-lite",
        openai_api_key=os.environ.get("GOOGLE_API_KEY"),
        openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    generator_llm = LangchainLLMWrapper(openai_compatible_llm)

    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    # --- Knowledge Graph Logic ---
    if os.path.exists(KG_OUTPUT_FILE):
        print(f"\n✅ Found existing Knowledge Graph at {KG_OUTPUT_FILE}. Loading...")
        kg = KnowledgeGraph.load(KG_OUTPUT_FILE)
        print(f"Loaded {len(kg.nodes)} nodes from file.")
    else:
        print("\n⚠️  No existing graph found. Creating new Knowledge Graph...")

        # Load Documents
        print("📄 Loading documents...")
        documents = []
        for directory in DIRECTORIES:
            if os.path.exists(directory):
                print(f"Loading from {directory}...")
                loader = DirectoryLoader(
                    directory,
                    glob="**/*.md",
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"},
                )
                documents.extend(loader.load())
            else:
                print(f"Warning: Directory {directory} not found.")

        processed_docs = []
        for doc in documents:
            doc.metadata["source"] = doc.metadata.get("source", "unknown")
            processed_docs.append(doc)

        # Create KG Nodes
        kg = KnowledgeGraph()
        for doc in processed_docs:
            kg.nodes.append(
                Node(
                    type=NodeType.DOCUMENT,
                    properties={
                        "page_content": doc.page_content,
                        "document_metadata": doc.metadata,
                    },
                )
            )

        # Apply Transforms
        print("🔄 Applying transforms...")
        transforms = default_transforms(
            documents=processed_docs, llm=generator_llm, embedding_model=embeddings
        )
        apply_transforms(kg, transforms=transforms)

        # Save KG
        print(f"💾 Saving Knowledge Graph to {KG_OUTPUT_FILE}...")
        kg.save(KG_OUTPUT_FILE)
        print("✅ Knowledge graph saved successfully!")

    # --- Setup Persona and Generator ---
    jee_persona = Persona(
        name="JEE_Chemistry_Professor",
        role_description=(
            "You are a distinguished Chemistry Professor and Chief Examiner for the JEE Advanced "
            "Entrance Exam in India. Your task is to create a challenging test dataset for top-tier "
            "engineering aspirants.\n\n"
            "GUIDELINES:\n"
            "1. **Complexity:** Generate multi-step questions that require deep conceptual understanding "
            "of Physical, Organic, and Inorganic Chemistry.\n"
            "2. **Formatting:** Use LaTeX formatting (e.g., $\\text{H}_2\\text{SO}_4$) for formulas "
            "only when necessary. For purely theoretical questions, clear text is preferred.\n"
            "3. **Topics:** Focus on reaction mechanisms, stoichiometry, and periodic trends.\n"
            "4. **Tone:** Academic, rigorous, and precise."
        ),
    )

    print("\n🔍 Generating testset...")
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0),
    ]

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=embeddings,
        knowledge_graph=kg,
        persona_list=[jee_persona],
    )

    testset = generator.generate(
        testset_size=50,
        query_distribution=query_distribution,
    )

    # --- Save Results ---
    print(f"\n💾 Saving testset to {OUTPUT_FILE}...")
    df = testset.to_pandas()

    df = df.rename(
        columns={
            "user_input": "question",
            "reference": "answer",
            "reference_contexts": "contexts",
        }
    )

    df = df[["question", "answer", "contexts"]]
    df.to_json(OUTPUT_FILE, orient="records", indent=4)

    print(f"\n✅ Successfully generated {len(df)} Q&A pairs!")
    print(f"📊 Dataset saved to: {OUTPUT_FILE}")
    print(f"🗺️  Knowledge Graph saved to: {KG_OUTPUT_FILE}")

    return testset


if __name__ == "__main__":
    print("🧪 Knowledge Graph-based Testset Generation Example")
    print("=" * 60)

    # Check if API key is configured
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  GOOGLE_API_KEY not set.")
        print("Set your API key in .env file or as environment variable:")
        print("  export GOOGLE_API_KEY='your-api-key-here'")
        print()
        exit(1)

    # Run main function
    try:
        testset = main()
        print("\n🎉 Example completed successfully!")
        print("For more information, see: docs/howtos/testset-generation.md")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure to:")
        print("1. Install required dependencies")
        print("2. Set up your GOOGLE_API_KEY")
        print("3. Update DIRECTORIES to point to your data")
        print("4. Ensure the output directory exists")
