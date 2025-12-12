"""
Integrations module for Ragas evaluation framework.

This module provides integrations with various platforms, frameworks, and tools
to enhance the Ragas evaluation experience.

Available integrations:
- Tracing: Langfuse, MLflow for observability and tracking
- Frameworks: LangChain, LlamaIndex, Griptape, LangGraph
- Observability: Helicone, Langsmith, Opik
- Platforms: Amazon Bedrock, R2R
- AI Systems: Swarm for multi-agent evaluation
- Protocols: AG-UI for event-based agent communication

Import tracing integrations:
```python
from ragas.integrations.tracing import observe, LangfuseTrace, MLflowTrace
```
"""

# Tracing integrations are available as a submodule
# Import them explicitly when needed to handle optional dependencies gracefully
