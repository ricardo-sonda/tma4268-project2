import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.features import diff_only_feature_set
from src.modeling.base import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class LogiticEducationalModel(BaseModel):
    name = "logistic_educational"

    def __init__(self) -> None:
        self.feature_builder = diff_only_feature_set
        self.active_feature_names = [
            "AgeDiff",
            "CurrentLoseStreakDiff",
            "CurrentWinStreakDiff",
            "HeightCmsDiff",
            "ReachCmsDiff",
            "WeightLbsDiff",
        ]
        self.pipeline = Pipeline(
            [
                ("scale", MinMaxScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        )

    def fit(self, X, y) -> None:
        X = X[self.active_feature_names]
        self.pipeline.fit(X, y)

    def predict_proba(self, X) -> np.ndarray:
        X = X[self.active_feature_names]
        return self.pipeline.predict_proba(X)[:, 1]


if __name__ == "__main__":
    model = LogiticEducationalModel()
    print(model.active_feature_names)
