"""Configuration helpers for the age estimation service.

Configuration is persisted on disk in ``config/age_service.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import json

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "age_service.json"


@dataclass(slots=True)
class ModelMetadata:
    name: str
    version: str
    mean_absolute_error: float
    calibration_date: str
    fairness_warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AgeServiceConfig:
    min_image_edge: int
    detector_name: str
    model_metadata: ModelMetadata
    default_confidence_level: float
    return_face_bbox_default: bool

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "AgeServiceConfig":
        metadata_raw = mapping.get("model_metadata", {})
        metadata = ModelMetadata(
            name=str(metadata_raw.get("name", "unknown")),
            version=str(metadata_raw.get("version", "unknown")),
            mean_absolute_error=float(metadata_raw.get("mean_absolute_error", 0.0)),
            calibration_date=str(metadata_raw.get("calibration_date", "unknown")),
            fairness_warnings=list(metadata_raw.get("fairness_warnings", [])),
            limitations=list(metadata_raw.get("limitations", [])),
        )
        return cls(
            min_image_edge=int(mapping.get("min_image_edge", 128)),
            detector_name=str(mapping.get("detector_name", "simple")),
            model_metadata=metadata,
            default_confidence_level=float(mapping.get("default_confidence_level", 0.9)),
            return_face_bbox_default=bool(mapping.get("return_face_bbox_default", False)),
        )


def load_config(path: str | Path | None = None) -> AgeServiceConfig:
    """Load configuration from ``path`` or the default location."""

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return AgeServiceConfig.from_mapping(payload)


__all__ = ["AgeServiceConfig", "ModelMetadata", "load_config", "DEFAULT_CONFIG_PATH"]
