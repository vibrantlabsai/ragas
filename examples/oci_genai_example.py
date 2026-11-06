#!/usr/bin/env python3
"""
Example script demonstrating OCI Gen AI integration with Ragas.

This script shows how to use Oracle Cloud Infrastructure Generative AI
models for RAG evaluation with Ragas.

Prerequisites:
1. Install ragas with OCI support: pip install ragas[oci]
2. Configure OCI authentication (see docs/howtos/integrations/oci_genai.md)
3. Have access to OCI Gen AI models in your compartment
"""

import os
from datasets import Dataset
from ragas import evaluate
from ragas.llms import oci_genai_factory
from ragas.metrics import faithfulness, answer_relevancy, context_precision


def main():
    """Main function demonstrating OCI Gen AI integration."""
    
    # Configuration - Update these values for your environment
    MODEL_ID = os.getenv("OCI_MODEL_ID", "cohere.command-a-03-2025")
    COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    ENDPOINT_ID = os.getenv("OCI_ENDPOINT_ID", None)  # Optional
    
    print("🚀 Initializing OCI Gen AI LLM...")
    
    # Initialize OCI Gen AI LLM
    try:
        llm = oci_genai_factory(
            model_id=MODEL_ID,
            compartment_id=COMPARTMENT_ID,
            endpoint_id=ENDPOINT_ID
        )
        print(f"✅ Successfully initialized OCI Gen AI with model: {MODEL_ID}")
    except Exception as e:
        print(f"❌ Failed to initialize OCI Gen AI: {e}")
        print("Please check your OCI configuration and credentials.")
        return
    
    # Create sample dataset for evaluation
    print("\n📊 Creating sample dataset...")
    dataset = Dataset.from_dict({
        "question": [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "What is the largest planet in our solar system?",
        ],
        "answer": [
            "Paris is the capital of France.",
            "William Shakespeare wrote Romeo and Juliet.",
            "Jupiter is the largest planet in our solar system.",
        ],
        "contexts": [
            ["France is a country in Europe. Its capital is Paris. France is known for its culture and cuisine."],
            ["Romeo and Juliet is a famous play written by William Shakespeare. It's a tragic love story."],
            ["Jupiter is the largest planet in our solar system. It's a gas giant with many moons."],
        ],
        "ground_truth": [
            "Paris",
            "William Shakespeare", 
            "Jupiter"
        ]
    })
    
    print(f"✅ Created dataset with {len(dataset)} examples")
    
    # Run evaluation
    print("\n🔍 Running RAG evaluation with OCI Gen AI...")
    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=llm
        )
        
        print("✅ Evaluation completed successfully!")
        print("\n📈 Results:")
        print(result)
        
        # Print individual metric scores
        print("\n📊 Detailed Scores:")
        for metric_name, score in result.items():
            print(f"  {metric_name}: {score:.4f}")
            
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        print("Please check your OCI configuration and model access.")


def test_llm_connection():
    """Test basic LLM connection and generation."""
    print("🧪 Testing OCI Gen AI connection...")
    
    MODEL_ID = os.getenv("OCI_MODEL_ID", "cohere.command-a-03-2025")
    COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..example")
    
    try:
        llm = oci_genai_factory(
            model_id=MODEL_ID,
            compartment_id=COMPARTMENT_ID
        )
        
        # Test simple generation
        from langchain_core.prompt_values import StringPromptValue
        prompt = StringPromptValue(text="Hello, how are you?")
        
        result = llm.generate_text(prompt, n=1, temperature=0.1)
        
        print("✅ Connection test successful!")
        print(f"Generated response: {result.generations[0][0].text}")
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        print("Please check your OCI configuration.")


if __name__ == "__main__":
    print("🔧 OCI Gen AI Integration Example")
    print("=" * 50)
    
    # Check if OCI configuration is available
    if not os.getenv("OCI_COMPARTMENT_ID"):
        print("⚠️  OCI_COMPARTMENT_ID not set. Using example value.")
        print("Set environment variables for your OCI configuration:")
        print("  export OCI_MODEL_ID='cohere.command-a-03-2025'")
        print("  export OCI_COMPARTMENT_ID='ocid1.compartment.oc1..your-compartment'")
        print("  export OCI_ENDPOINT_ID='ocid1.endpoint.oc1..your-endpoint'  # Optional")
        print()
    
    # Test connection first
    test_llm_connection()
    
    print("\n" + "=" * 50)
    
    # Run main evaluation
    main()
    
    print("\n🎉 Example completed!")
    print("For more information, see: docs/howtos/integrations/oci_genai.md")
