"""Registry of available model classes."""

from src.modeling.baseline import BaselineModel
from .elasticnet_diff import ElasticNetDiffModel
from .elasticnet_nested_cv_diff import ElasticNetNestedCVDiffModel
from .logistic_diff import LogisticDiffModel
from .logistic_diff_complete import LogisticDiffCompleteModel
from .logistic_educational import LogisticEducationalModel
from .logistic_ground_zero import LogisticGroundZeroModel
from .logistic_ground_zero_complete import LogisticGroundZeroCompleteModel
from .logistic_scratch_diff_complete import LogisticScratchDiffCompleteModel
from .logistic_statsmodels_interpretable import LogisticStatsmodelsInterpretableModel
from .logistic_weightclass_interaction import LogisticWeightclassInteractionModel
from .logistic_weightclass_interaction_complete import (
    LogisticWeightclassInteractionCompleteModel,
)
from .logistic_weightclass_main_effect import LogisticWeightclassMainEffectModel
from .logistic_weightclass_main_effect_complete import (
    LogisticWeightclassMainEffectCompleteModel,
)
from .naive import NaiveModel
from .random_forest_diff import RandomForestDiffModel
from .xgb_diff import XGBDiffModel

MODEL_CLASSES = {
    LogisticDiffModel.name: LogisticDiffModel,
    LogisticDiffCompleteModel.name: LogisticDiffCompleteModel,
    LogisticGroundZeroModel.name: LogisticGroundZeroModel,
    LogisticGroundZeroCompleteModel.name: LogisticGroundZeroCompleteModel,
    LogisticScratchDiffCompleteModel.name: LogisticScratchDiffCompleteModel,
    LogisticEducationalModel.name: LogisticEducationalModel,
    NaiveModel.name: NaiveModel,
    BaselineModel.name: BaselineModel,
    XGBDiffModel.name: XGBDiffModel,
    ElasticNetDiffModel.name: ElasticNetDiffModel,
    ElasticNetNestedCVDiffModel.name: ElasticNetNestedCVDiffModel,
    LogisticStatsmodelsInterpretableModel.name: LogisticStatsmodelsInterpretableModel,
    LogisticWeightclassMainEffectModel.name: LogisticWeightclassMainEffectModel,
    LogisticWeightclassMainEffectCompleteModel.name: (
        LogisticWeightclassMainEffectCompleteModel
    ),
    LogisticWeightclassInteractionModel.name: LogisticWeightclassInteractionModel,
    LogisticWeightclassInteractionCompleteModel.name: (
        LogisticWeightclassInteractionCompleteModel
    ),
    RandomForestDiffModel.name: RandomForestDiffModel,
}
