"""AgentGoalAccuracy metrics - Modern collections implementation."""

from ragas.metrics.collections.agent_goal_accuracy.metric import (
    AgentGoalAccuracy,
    AgentGoalAccuracyWithoutReference,
    AgentGoalAccuracyWithReference,
)

__all__ = [
    "AgentGoalAccuracy",
    "AgentGoalAccuracyWithReference",
    "AgentGoalAccuracyWithoutReference",
]
