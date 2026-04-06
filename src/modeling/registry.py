"""Registry of available model classes."""

from src.modeling.baseline import BaselineModel
from .elasticnet_diff_only import ElasticNetDiffOnlyModel
from .logistic_diff_only import LogisticDiffOnlyModel
from .logistic_educational import LogisticEducationalModel
from .logistic_ground_zero import LogisticGroundZeroModel
from .logistic_raw_imputed import LogisticRawImputedModel
from .logistic_scratch_diff_only import LogisticScratchDiffOnlyModel
from .naive import NaiveModel
from .random_forest_diff_imputed import RandomForestDiffImputedModel
from .xgb_diff_only import XGBDiffOnlyModel

MODEL_CLASSES = {
    # LogisticDiffOnlyModel.name: LogisticDiffOnlyModel,
    LogisticGroundZeroModel.name: LogisticGroundZeroModel,
    # LogisticScratchDiffOnlyModel.name: LogisticScratchDiffOnlyModel,
    LogisticEducationalModel.name: LogisticEducationalModel,
    NaiveModel.name: NaiveModel,
    BaselineModel.name: BaselineModel,
    XGBDiffOnlyModel.name: XGBDiffOnlyModel,
    ElasticNetDiffOnlyModel.name: ElasticNetDiffOnlyModel,
    LogisticRawImputedModel.name: LogisticRawImputedModel,
    RandomForestDiffImputedModel.name: RandomForestDiffImputedModel,
}
