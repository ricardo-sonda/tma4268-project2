"""Registry of available model classes."""

from src.modeling.logistic_educational import LogiticEducationalModel
from src.modeling.naive import NaiveModel
from .logistic_diff_only import LogisticDiffOnlyModel
from .logistic_ground_zero import LogisticGroundZeroModel
from .logistic_scratch_diff_only import LogisticScratchDiffOnlyModel

MODEL_CLASSES = {
    # LogisticDiffOnlyModel.name: LogisticDiffOnlyModel,
    LogisticGroundZeroModel.name: LogisticGroundZeroModel,
    # LogisticScratchDiffOnlyModel.name: LogisticScratchDiffOnlyModel,
    LogiticEducationalModel.name: LogiticEducationalModel,
    NaiveModel.name: NaiveModel,
}
