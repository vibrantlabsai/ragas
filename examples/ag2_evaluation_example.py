"""Example: Evaluating AG2 multi-agent conversations with Ragas.

This example demonstrates how to convert AG2 agent conversation history
into Ragas format and evaluate it using built-in metrics.

AG2 (https://ag2.ai/) is an open-source multi-agent framework with
500K+ monthly PyPI downloads. This integration enables systematic
evaluation of AG2 agent interactions.

Requirements:
    pip install ragas ag2[openai]
"""

from ragas.integrations.ag2 import convert_to_ragas_messages

# Simulated AG2 multi-agent conversation
# In practice, you'd get this from agent.chat_messages after running
# user_proxy.run(assistant, message="...").process()
ag2_conversation = [
    {
        "role": "user",
        "content": "Research the latest developments in quantum computing "
        "and summarize the key breakthroughs.",
    },
    {
        "role": "assistant",
        "content": "",
        "name": "researcher",
        "tool_calls": [
            {
                "id": "call_research_1",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": '{"query": "quantum computing breakthroughs 2026"}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "content": '{"results": ["IBM unveiled 1000+ qubit processor", '
        '"Google achieved quantum error correction milestone", '
        '"Microsoft announced topological qubit progress"]}',
        "tool_call_id": "call_research_1",
        "name": "web_search",
    },
    {
        "role": "assistant",
        "content": "Based on my research, here are the key quantum computing "
        "breakthroughs in 2026:\n\n"
        "1. **IBM** unveiled a processor exceeding 1,000 qubits\n"
        "2. **Google** achieved a major quantum error correction milestone\n"
        "3. **Microsoft** announced significant progress on topological qubits\n\n"
        "These developments represent meaningful steps toward practical "
        "quantum computing applications.",
        "name": "researcher",
    },
    {
        "role": "assistant",
        "content": "The researcher's summary is accurate and well-structured. "
        "The three breakthroughs cover the major players (IBM, Google, Microsoft) "
        "and highlight distinct technical achievements. The summary is suitable "
        "for a general audience.",
        "name": "reviewer",
    },
]

# Convert AG2 messages to Ragas format
ragas_messages = convert_to_ragas_messages(ag2_conversation)

# Display converted messages
print("=== Converted Ragas Messages ===\n")
for i, msg in enumerate(ragas_messages):
    print(f"[{i}] {type(msg).__name__}: {msg.pretty_repr()}")
    print()

# Show message types summary
print("=== Summary ===")
print(f"Total messages: {len(ragas_messages)}")
type_counts = {}
for msg in ragas_messages:
    name = type(msg).__name__
    type_counts[name] = type_counts.get(name, 0) + 1
for name, count in type_counts.items():
    print(f"  {name}: {count}")

# With metadata enabled — captures agent names
print("\n=== With Metadata ===\n")
ragas_messages_meta = convert_to_ragas_messages(ag2_conversation, metadata=True)
for msg in ragas_messages_meta:
    if hasattr(msg, "metadata") and msg.metadata:
        print(f"{type(msg).__name__} metadata: {msg.metadata}")