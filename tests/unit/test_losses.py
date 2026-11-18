import pytest

from ragas.config import InstructionConfig
from ragas.losses import BinaryMetricLoss, MSELoss


class TestMSELoss:
    def test_default_reduction_is_mean(self):
        loss = MSELoss()
        assert loss.reduction == "mean"
        assert loss([1.0, 3.0], [0.0, 0.0]) == pytest.approx(5.0)

    def test_reduction_can_be_set_in_constructor(self):
        loss = MSELoss(reduction="sum")
        assert loss.reduction == "sum"
        assert loss([1.0, 3.0], [0.0, 0.0]) == pytest.approx(10.0)


class TestBinaryMetricLoss:
    def test_default_metric_is_accuracy(self):
        loss = BinaryMetricLoss()
        assert loss.metric == "accuracy"
        assert loss([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.75)

    def test_metric_can_be_set_in_constructor(self):
        loss = BinaryMetricLoss(metric="f1_score")
        assert loss.metric == "f1_score"
        # tp=1, fp=1, fn=0 -> precision=0.5, recall=1.0 -> f1=2/3
        assert loss([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "loss_cls, options",
    [
        (MSELoss, {}),
        (MSELoss, {"reduction": "sum"}),
        (BinaryMetricLoss, {}),
        (BinaryMetricLoss, {"metric": "f1_score"}),
    ],
)
def test_instruction_config_accepts_loss_instance(fake_llm, loss_cls, options):
    loss = loss_cls(**options)
    config = InstructionConfig(llm=fake_llm, loss=loss)
    assert config.loss is loss
