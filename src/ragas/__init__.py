from ragas import backends
from ragas.cache import CacheInterface, DiskCacheBackend, cacher
from ragas.dataset import Dataset, DataTable
from ragas.dataset_schema import EvaluationDataset, MultiTurnSample, SingleTurnSample
from ragas.evaluation import aevaluate, evaluate
from ragas.experiment import Experiment, experiment, version_experiment
from ragas.run_config import RunConfig
from ragas.tokenizers import (
    BaseTokenizer,
    HuggingFaceTokenizer,
    TiktokenWrapper,
    get_tokenizer,
)

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown version"


__all__ = [
    "evaluate",
    "aevaluate",
    "RunConfig",
    "__version__",
    "SingleTurnSample",
    "MultiTurnSample",
    "EvaluationDataset",
    "DataTable",
    "Dataset",
    "cacher",
    "CacheInterface",
    "DiskCacheBackend",
    "backends",
    "Experiment",
    "experiment",
    "version_experiment",
    "BaseTokenizer",
    "TiktokenWrapper",
    "HuggingFaceTokenizer",
    "get_tokenizer",
]


def __getattr__(name):
    if name == "experimental":
        try:
            import ragas_experimental as experimental  # type: ignore

            return experimental
        except ImportError:
            raise ImportError(
                "ragas.experimental requires installation: "
                "pip install ragas[experimental]"
            )
    raise AttributeError(f"module 'ragas' has no attribute '{name}'")
