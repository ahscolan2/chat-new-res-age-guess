"""Age prediction model wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Protocol

from .image import SimpleImage


@dataclass(slots=True)
class AgePrediction:
    value: float
    uncertainty: float


class AgeModel(Protocol):
    """Protocol for models capable of predicting age."""

    model_name: str
    model_version: str

    def predict(self, image: SimpleImage) -> AgePrediction:
        ...


class CalibratedLinearModel:
    """Reference implementation with a deterministic calibrated output.

    The model computes the mean pixel intensity and scales it to an age using a
    simple calibration. Uncertainty captures model residual error.
    """

    model_name = "calibrated-linear"
    model_version = "1.0.0"

    def __init__(self, *, slope: float, intercept: float, residual_std: float) -> None:
        self.slope = slope
        self.intercept = intercept
        self.residual_std = residual_std

    def predict(self, image: SimpleImage) -> AgePrediction:
        mean_intensity = image.mean_intensity()
        value = self.slope * (mean_intensity / 255.0) + self.intercept
        return AgePrediction(value=value, uncertainty=self.residual_std)


def confidence_interval(prediction: AgePrediction, level: float) -> tuple[float, float]:
    """Compute a symmetric confidence interval for ``prediction``."""

    level = max(min(level, 0.999), 0.0)
    if level <= 0:
        return prediction.value, prediction.value

    dist = NormalDist(mu=prediction.value, sigma=prediction.uncertainty)
    tail = (1 - level) / 2
    lower = dist.inv_cdf(tail)
    upper = dist.inv_cdf(1 - tail)
    return lower, upper


__all__ = [
    "AgePrediction",
    "AgeModel",
    "CalibratedLinearModel",
    "confidence_interval",
]
