import pytest
from pydantic import ValidationError

from ragas.metrics.collections.factual_correctness.util import (
    NLIStatementOutput,
)
from ragas.metrics.collections.noise_sensitivity.util import StatementFaithfulnessOutput


@pytest.mark.parametrize(
    "schema",
    [NLIStatementOutput, StatementFaithfulnessOutput],
)
@pytest.mark.parametrize("verdict", [-1, 2])
def test_judge_output_rejects_nonbinary_statement_verdicts(schema, verdict):
    fields = {"statement": "An example statement", "reason": "Relevant context"}
    for valid in (0, 1):
        result = schema(statements=[{"verdict": valid, **fields}])
        assert result.statements[0].verdict == valid
    with pytest.raises(ValidationError, match="verdict"):
        schema(statements=[{"verdict": verdict, **fields}])
