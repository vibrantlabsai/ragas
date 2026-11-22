from ragas.metrics._faithfulness import FaithfulnesswithHHEM as LegacyFaithfulnesswithHHEM
from ragas.metrics.collections import FaithfulnesswithHHEM


def test_faithfulness_with_hhem_is_exported_from_collections():
    assert FaithfulnesswithHHEM is LegacyFaithfulnesswithHHEM
