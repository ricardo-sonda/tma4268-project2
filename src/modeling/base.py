"""Abstract model contract."""

from abc import ABC, abstractmethod

import numpy as np

from ..features import FeatureBuilder


class BaseModel(ABC):
    """Thin interface for a model used by the shared evaluator."""

    name: str
    feature_builder: FeatureBuilder

    @abstractmethod
    def fit(self, X, y) -> None:
        """Fit the model on the provided design matrix."""

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """Return P(Red wins) for each row."""

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
