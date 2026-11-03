import pytest

from ragas.testset.synthesizers.generate import TestsetGenerator


class _NeverUsed:
    pass


@pytest.mark.parametrize("size", [0, -1, None])
def test_a_testset_size_that_cannot_be_generated_is_named(size):
    generator = TestsetGenerator.__new__(TestsetGenerator)

    with pytest.raises(ValueError, match="testset_size must be a positive integer"):
        generator.generate(testset_size=size)
