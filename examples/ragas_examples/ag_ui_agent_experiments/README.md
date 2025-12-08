# AG-UI Agent Evaluation Examples

This example demonstrates how to evaluate agents built with the **AG-UI protocol** using Ragas metrics.

## What is AG-UI?

AG-UI (Agent-User Interaction) is a protocol for streaming agent events from backend to frontend. It defines a standardized event format for agent-to-UI communication, enabling real-time streaming of agent actions, tool calls, and responses.

## Prerequisites

Before running these examples, you need to have an AG-UI compatible agent running. Follow the [AG-UI Quickstart Guide](https://docs.ag-ui.com/quickstart/applications) to set up your agent.

### Popular AG-UI Compatible Frameworks

- **Google ADK (Agent Development Kit)** - Google's framework for building AI agents
- **Pydantic AI** - Type-safe agent framework using Pydantic
- **Mastra** - Modular, TypeScript-based agentic AI framework
- **Crew.ai** - Python framework for orchestrating collaborative, specialized AI agent teams
- And more...

### Example Setup

Here's a quick overview of setting up an AG-UI agent (refer to the [official documentation](https://docs.ag-ui.com/quickstart/applications) for detailed instructions):u

1. Choose your agent framework (e.g., Google ADK, Pydantic AI)
2. Implement your agent with the required tools
3. Start the AG-UI server (typically runs at `http://localhost:8000/chat` or `http://localhost:8000/agentic_chat`)
4. Verify the endpoint is accessible

## Installation

Install the required dependencies:

```bash
# From the ragas repository root
uv pip install -e ".[dev]"

# Or install specific dependencies
pip install ragas openai
```

## Evaluation Scenarios

This example includes two evaluation scenarios:

### 1. Scientist Biographies (Factuality & Grounding)

Tests the agent's ability to provide factually correct information about famous scientists and keep responses concise. The evaluation uses the modern collections portfolio plus a discrete conciseness check implemented with `DiscreteMetric`.

- **Metrics**: Collections metrics — `FactualCorrectness` (mode `f1`, atomicity `high`, coverage `high`), `AnswerRelevancy` (strictness `2`), and a custom `conciseness` metric (DiscreteMetric)
- **Dataset**: `test_data/scientist_biographies.csv` - 5 questions about scientists (Einstein, Fleming, Newton, etc.)
- **Sample Type**: `SingleTurnSample` - Simple question-answer pairs

### 2. Weather Tool Usage (Tool Call F1)

Tests the agent's ability to correctly invoke the weather tool when appropriate.

- **Metric**: `ToolCallF1` - F1 score measuring precision and recall of tool invocations
- **Dataset**: `test_data/weather_tool_calls.csv` - 5 queries requiring weather tool calls
- **Sample Type**: `MultiTurnSample` - Multi-turn conversations with tool call expectations

## Usage

### Basic Usage

Run both evaluation scenarios:

```bash
cd examples/ragas_examples/ag_ui_agent_evals
python evals.py --endpoint-url http://localhost:8000/agentic_chat
```

### Command Line Options

```bash
# Specify a different endpoint
python evals.py --endpoint-url http://localhost:8010/chat

# Use a different evaluator model
python evals.py --evaluator-model gpt-4o

# Skip the factual correctness evaluation
python evals.py --skip-factual

# Skip the tool call evaluation
python evals.py --skip-tool-eval

# Specify output directory for results
python evals.py --output-dir ./results

# Combine options
python evals.py \
    --endpoint-url http://localhost:8000/agentic_chat \
    --evaluator-model gpt-4o-mini \
    --output-dir ./my_results
```

### Using uv (Recommended)

```bash
# Run with uv from the examples directory
cd examples
uv run python ragas_examples/ag_ui_agent_evals/evals.py --endpoint-url http://localhost:8000/agentic_chat
```

### Environment variables

The script loads `.env` from the repository root, so configure your evaluator credentials there:

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

## Expected Output

### Console Output

The script will print detailed evaluation results:

```
================================================================================
Starting Scientist Biographies Evaluation
================================================================================
Loading scientist biographies dataset from .../test_data/scientist_biographies.csv
Loaded 5 scientist biography samples
Evaluating against endpoint: http://localhost:8000/agentic_chat

================================================================================
Scientist Biographies Evaluation Results
================================================================================
                                          user_input  ... conciseness
0  Who originated the theory of relativity...     ...    concise
1  Who discovered penicillin and when...           ...    verbose
...

Average Factual Correctness: 0.7160
Average Answer Relevancy: 0.8120
Concise responses: 60.00%
Perfect factual scores (1.0): 2/5

Results saved to: .../scientist_biographies_results_20250101_143022.csv

================================================================================
Starting Weather Tool Usage Evaluation
================================================================================
...
Average Tool Call F1: 1.0000
Perfect scores (F1=1.0): 5/5
Failed scores (F1=0.0): 0/5

Results saved to: .../weather_tool_calls_results_20250101_143045.csv

================================================================================
All evaluations completed successfully!
================================================================================
```

### CSV Output Files

Results are saved as timestamped CSV files:

- `scientist_biographies_results_YYYYMMDD_HHMMSS.csv`
- `weather_tool_calls_results_YYYYMMDD_HHMMSS.csv`

Example CSV structure:

```csv
user_input,response,reference,factual_correctness(mode=f1),answer_relevancy,conciseness
"Who originated the theory of relativity...","Albert Einstein...","Albert Einstein originated...",0.75,0.82,concise
```

## Customizing the Evaluation

### Adding New Test Cases

#### For Factual Correctness

Edit `test_data/scientist_biographies.csv`:

```csv
user_input,reference
"Your question here","Your reference answer here"
```

#### For Tool Call Evaluation

Edit `test_data/weather_tool_calls.csv`:

```csv
user_input,reference_tool_calls
"What's the weather in Paris?","[{\"name\": \"weatherTool\", \"args\": {\"location\": \"Paris\"}}]"
```

### Using Different Metrics

Modify `evals.py` to include additional collections metrics:

```python
from ragas.metrics.collections import AnswerRelevancy, ContextPrecisionWithoutReference

# In evaluate_scientist_biographies function:
metrics = [
    AnswerRelevancy(llm=evaluator_llm),
    ContextPrecisionWithoutReference(llm=evaluator_llm),
    ResponseGroundedness(llm=evaluator_llm),
]
```

### Evaluating Your Own Agent

1. **Ensure your agent supports AG-UI protocol**
   - Agent must expose an endpoint that accepts AG-UI messages
   - Agent must return Server-Sent Events (SSE) with AG-UI event format

2. **Update the endpoint URL**
   ```bash
   python evals.py --endpoint-url http://your-agent:port/your-endpoint
   ```

3. **Customize test data**
   - Create new CSV files with your test cases
   - Update the loader functions in `evals.py` if needed

## Troubleshooting

### Connection Errors

```
Error: Connection refused at http://localhost:8000/agentic_chat
```

**Solution**: Ensure your AG-UI agent is running and accessible at the specified endpoint.

### Import Errors

```
ImportError: No module named 'ragas'
```

**Solution**: Install ragas and its dependencies:
```bash
pip install ragas langchain-openai
```

### API Key Errors

```
Error: OpenAI API key not found
```

**Solution**: Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

### Agent Timeout

```
Error: Request timeout after 60.0 seconds
```

**Solution**: Your agent may be slow to respond. You can increase the timeout in the code or optimize your agent's performance.

## Understanding the Results

### Factual Correctness Metric

- **Range**: 0.0 to 1.0
- **1.0**: Perfect match between response and reference
- **0.5-0.9**: Partially correct with some missing or incorrect information
- **<0.5**: Significant discrepancies with the reference

### Answer Relevancy Metric

- **Range**: 0.0 to 1.0
- **1.0**: All generated follow-up questions align tightly with the original user input
- **0.5-0.9**: Mostly relevant answers with minor drift or non-committal language
- **<0.5**: Response is largely unrelated or evasive compared to the user query

### Conciseness Metric

- **Values**: `concise` or `verbose`
- **concise**: The evaluator judged the answer as efficient and to the point
- **verbose**: The answer included unnecessary repetition or tangents

### Tool Call F1 Metric

- **Range**: 0.0 to 1.0
- **1.0**: Perfect tool call accuracy (correct tools with correct arguments)
- **0.5-0.9**: Some correct tools but missing some or calling extra tools
- **0.0**: Incorrect tool usage or no tool calls when expected

## Integration with Your Workflow

### CI/CD Integration

You can integrate these evaluations into your CI/CD pipeline:

```bash
# In your CI script
python evals.py \
    --endpoint-url http://staging-agent:8000/chat \
    --output-dir ./test-results \
    || exit 1
```

### Tracking Performance Over Time

Save results with timestamps to track improvements:

```bash
# Run evaluations regularly
python evals.py --output-dir ./historical-results/$(date +%Y%m%d)
```

### Automated Testing

Create a simple test harness:

```python
import subprocess
import sys

result = subprocess.run(
    ["python", "evals.py", "--endpoint-url", "http://localhost:8000/chat"],
    capture_output=True
)

if result.returncode != 0:
    print("Evaluation failed!")
    sys.exit(1)
```

## Additional Resources

- [AG-UI Documentation](https://docs.ag-ui.com)
- [AG-UI Quickstart](https://docs.ag-ui.com/quickstart/applications)
- [Ragas Documentation](https://docs.ragas.io)
- [Ragas AG-UI Integration Guide](https://docs.ragas.io/integrations/ag-ui)
