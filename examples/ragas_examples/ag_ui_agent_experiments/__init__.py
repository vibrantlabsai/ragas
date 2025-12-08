"""
AG-UI Agent Evaluation Examples

This package demonstrates how to evaluate agents built with the AG-UI protocol
using Ragas metrics.

## What is AG-UI?

AG-UI (Agent-to-UI) is a protocol for streaming agent events from backend to frontend.
It defines a standardized event format for agent-to-UI communication.

## Getting Started

Before running these examples, you'll need to have an AG-UI compatible agent running.
Follow the AG-UI quickstart guide to set up your agent:

https://docs.ag-ui.com/quickstart/applications

Popular agent frameworks that support AG-UI include:
- Google ADK (Agent Development Kit)
- Pydantic AI
- And more...

## Running the Examples

Once you have your AG-UI agent endpoint running (typically at
http://localhost:8000/chat or http://localhost:8000/agentic_chat), you can run
the evaluation examples:

```bash
# From the examples directory
cd ragas_examples/ag_ui_agent_evals
uv run python evals.py --endpoint-url http://localhost:8000/agentic_chat
```

## Evaluation Scenarios

This package includes two evaluation scenarios:

1. **Scientist Biographies** - Uses the modern collections metrics
   (`FactualCorrectness`, `ContextPrecisionWithReference`, `ContextRecall`,
   `ResponseGroundedness`) with `SingleTurnSample` datasets to score factuality
   and grounding in one pass.

2. **Weather Tool Usage** - Tests tool calling accuracy using the `ToolCallF1`
   metric with `MultiTurnSample` datasets.

## Results

Evaluation results are saved as CSV files with timestamps for tracking performance
over time.
"""

__version__ = "0.1.0"
