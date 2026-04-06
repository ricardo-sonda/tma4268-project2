"""Registry of available model classes."""

from .logistic_diff_only import LogisticDiffOnlyModel
from .logistic_educational import LogisticEducationalModel
from .logistic_ground_zero import LogisticGroundZeroModel
from .logistic_scratch_diff_only import LogisticScratchDiffOnlyModel
from .naive import NaiveModel

MODEL_CLASSES = {
    # LogisticDiffOnlyModel.name: LogisticDiffOnlyModel,
    LogisticGroundZeroModel.name: LogisticGroundZeroModel,
    # LogisticScratchDiffOnlyModel.name: LogisticScratchDiffOnlyModel,
    LogisticEducationalModel.name: LogisticEducationalModel,
    NaiveModel.name: NaiveModel,
}
