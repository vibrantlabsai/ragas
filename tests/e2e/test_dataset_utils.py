"""Utilities for creating test datasets in e2e tests."""

import logging

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

# Sample data structure matching the amnesty_qa dataset
SAMPLE_AMNESTY_DATA = [
    {
        "user_input": "What are the global implications of the USA Supreme Court ruling on abortion?",
        "reference": "The global implications of the USA Supreme Court ruling on abortion are significant. The ruling has led to limited or no access to abortion for one in three women and girls of reproductive age in states where abortion access is restricted. These states also have weaker maternal health support, higher maternal death rates, and higher child poverty rates. Additionally, the ruling has had an impact beyond national borders due to the USA's geopolitical and cultural influence globally.",
        "response": "The global implications of the USA Supreme Court ruling on abortion can be significant, as it sets a precedent for other countries and influences the global discourse on reproductive rights. The Supreme Court's ruling can serve as a reference point for other countries grappling with their own abortion laws.",
        "retrieved_contexts": [
            "In 2022, the USA Supreme Court handed down a decision ruling that overturned 50 years of jurisprudence recognizing a constitutional right to abortion.",
            "This decision has had a massive impact: one in three women and girls of reproductive age now live in states where abortion access is either totally or near-totally inaccessible.",
            "The USA Supreme Court ruling has also had impacts beyond national borders due to the geopolitical and cultural influence wielded by the USA globally.",
        ],
    },
    {
        "user_input": "How does climate change affect human rights?",
        "reference": "Climate change poses significant threats to human rights by affecting access to water, food security, health, and adequate housing. It disproportionately impacts vulnerable populations and can lead to displacement and migration.",
        "response": "Climate change impacts human rights through multiple pathways including threats to life, health, food, water, and adequate standard of living. The effects are often most severe for marginalized communities.",
        "retrieved_contexts": [
            "Climate change threatens the effective enjoyment of human rights including life, water and sanitation, food, health, housing, and livelihoods.",
            "The impacts of climate change will be felt most acutely by those segments of the population who are already in vulnerable situations.",
            "Climate change is already displacing people and will continue to do so in the future.",
        ],
    },
]

# Sample data structure matching the fiqa dataset
SAMPLE_FIQA_DATA = [
    {
        "user_input": "How to deposit a cheque issued to an associate in my business account?",
        "reference": "Have the check reissued to the proper payee. Just have the associate sign the back and then deposit it. It's called a third party cheque and is perfectly legal. I wouldn't be surprised if it has a longer hold period and, as always, you don't get the money if the cheque doesn't clear.",
        "response": "The best way to deposit a cheque issued to an associate in your business account is to have the associate sign the back of the cheque and deposit it as a third party cheque.",
        "retrieved_contexts": [
            "Just have the associate sign the back and then deposit it. It's called a third party cheque and is perfectly legal.",
            "I wouldn't be surprised if it has a longer hold period and, as always, you don't get the money if the cheque doesn't clear.",
        ],
    },
    {
        "user_input": "What is the difference between a mutual fund and an ETF?",
        "reference": "Mutual funds are actively managed investment vehicles that pool money from multiple investors. ETFs are passively managed and trade on exchanges like stocks. ETFs typically have lower fees and can be bought and sold throughout the trading day.",
        "response": "A mutual fund pools money from investors and is actively managed, while an ETF trades like a stock and typically tracks an index with lower fees.",
        "retrieved_contexts": [
            "Mutual funds pool money from multiple investors and are actively managed by professional fund managers.",
            "ETFs trade on exchanges like stocks and can be bought and sold throughout the trading day.",
            "ETFs typically have lower expense ratios compared to mutual funds.",
        ],
    },
    {
        "user_input": "Should I pay off my mortgage early or invest the money?",
        "reference": "It depends on your mortgage interest rate and expected investment returns. If your mortgage rate is low and you expect higher returns from investments, investing may be better. Consider your risk tolerance and financial goals.",
        "response": "The decision depends on comparing your mortgage interest rate to expected investment returns, along with your risk tolerance and financial security needs.",
        "retrieved_contexts": [
            "Compare your mortgage interest rate to expected investment returns to make an informed decision.",
            "Consider your risk tolerance and overall financial situation before making this decision.",
            "Having no mortgage provides peace of mind and guaranteed savings equal to the interest rate.",
        ],
    },
]


def load_amnesty_dataset_safe(config: str = "english_v3"):
    """
    Safely load the amnesty_qa dataset, falling back to local data if remote fails.

    Args:
        config: Dataset configuration name (e.g., "english_v3", "english_v2")

    Returns:
        Dataset: The loaded dataset
    """
    try:
        logger.info(f"Attempting to load amnesty_qa dataset with config '{config}'")
        dataset = load_dataset("vibrantlabsai/amnesty_qa", config)["eval"]
        logger.info(f"Successfully loaded dataset with {len(dataset)} samples")
        return dataset
    except Exception as e:
        logger.warning(f"Failed to load remote dataset: {e}")
        logger.info("Using local sample data as fallback")

        # Create a local dataset from sample data
        local_dataset = Dataset.from_list(SAMPLE_AMNESTY_DATA)
        logger.info(f"Created local dataset with {len(local_dataset)} samples")
        return local_dataset


def load_fiqa_dataset_safe(config: str = "ragas_eval_v3"):
    """
    Safely load the fiqa dataset, falling back to local data if remote fails.

    Args:
        config: Dataset configuration name (default: "ragas_eval_v3" - recommended)

    Returns:
        Dataset: The loaded dataset
    """
    try:
        logger.info(f"Attempting to load fiqa dataset with config '{config}'")
        dataset = load_dataset("vibrantlabsai/fiqa", config)["baseline"]
        logger.info(f"Successfully loaded dataset with {len(dataset)} samples")
        return dataset
    except Exception as e:
        logger.warning(f"Failed to load remote dataset: {e}")
        logger.info("Using local sample data as fallback")

        # Create a local dataset from sample data
        local_dataset = Dataset.from_list(SAMPLE_FIQA_DATA)
        logger.info(f"Created local dataset with {len(local_dataset)} samples")
        return local_dataset
