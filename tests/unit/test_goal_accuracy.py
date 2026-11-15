import json
import typing as t
from dataclasses import dataclass

import pytest

from ragas.config import InstructionConfig
from ragas.losses import BinaryMetricLoss, Loss
from ragas.metrics import (
    AgentGoalAccuracyWithoutReference,
    AgentGoalAccuracyWithReference,
)
from ragas.metrics.base import MetricOutputType
from ragas.optimizers import Optimizer


@dataclass
class RecordingOptimizer(Optimizer):
    """Optimizer stub that records the loss instead of searching for prompts."""

    recorded_loss: t.Optional[Loss] = None

    def optimize(self, dataset, loss, config, **kwargs):
        self.recorded_loss = loss
        return {}


@pytest.mark.parametrize(
    "metric",
    [AgentGoalAccuracyWithReference, AgentGoalAccuracyWithoutReference],
)
def test_goal_accuracy_metrics_declare_output_type(metric):
    assert metric().output_type == MetricOutputType.BINARY


def test_instruction_training_derives_loss_from_output_type(tmp_path, fake_llm):
    metric = AgentGoalAccuracyWithoutReference(llm=fake_llm)
    annotation_path = tmp_path / "agent_goal_accuracy.json"
    annotation_path.write_text(
        json.dumps(
            {
                "agent_goal_accuracy": [
                    {
                        "metric_input": {"user_input": "book a table for 8:00pm"},
                        "metric_output": 1.0,
                        "prompts": {},
                        "is_accepted": True,
                    }
                ]
            }
        )
    )
    optimizer = RecordingOptimizer()

    metric.train(
        str(annotation_path),
        instruction_config=InstructionConfig(llm=fake_llm, optimizer=optimizer),
    )

    assert isinstance(optimizer.recorded_loss, BinaryMetricLoss)
