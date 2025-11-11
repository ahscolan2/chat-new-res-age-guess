"""Input validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import AgeServiceConfig
from .detector import FaceBoundingBox
from .exceptions import ValidationError


@dataclass(slots=True)
class ValidatedRequest:
    image_base64: str
    consent: bool
    confidence_level: float
    return_face_bbox: bool
    detector_hints: list[FaceBoundingBox]


def _coerce_confidence_level(value: Any, *, field: str) -> float:
    try:
        level = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("confidence level must be a number", field=field) from exc
    if not 0 < level < 1:
        raise ValidationError("confidence level must be in (0, 1)", field=field)
    return level


def _parse_detector_hints(value: Any, *, field: str) -> list[FaceBoundingBox]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValidationError("detector_hints must be a list", field=field)

    hints: list[FaceBoundingBox] = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ValidationError("detector_hints entries must be objects", field=field)
        try:
            x = int(entry["x"])
            y = int(entry["y"])
            width = int(entry["width"])
            height = int(entry["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("invalid detector hint", field=field) from exc
        hints.append(FaceBoundingBox(x=x, y=y, width=width, height=height))
    return hints


def validate_payload(payload: Dict[str, Any], config: AgeServiceConfig) -> ValidatedRequest:
    if not isinstance(payload, dict):
        raise ValidationError("payload must be an object")

    try:
        consent = bool(payload["consent"])
    except KeyError as exc:
        raise ValidationError("consent is required", field="consent") from exc
    if not consent:
        raise ValidationError("consent must be granted", field="consent")

    try:
        image_base64 = payload["image_base64"]
    except KeyError as exc:
        raise ValidationError("image_base64 is required", field="image_base64") from exc
    if not isinstance(image_base64, str) or not image_base64:
        raise ValidationError("image_base64 must be a non-empty string", field="image_base64")

    confidence_level_raw = payload.get("confidence_level", config.default_confidence_level)
    confidence_level = _coerce_confidence_level(confidence_level_raw, field="confidence_level")

    return_face_bbox_raw = payload.get("return_face_bbox", config.return_face_bbox_default)
    return_face_bbox = bool(return_face_bbox_raw)

    hints_raw = payload.get("detector_hints")
    detector_hints = _parse_detector_hints(hints_raw, field="detector_hints")

    return ValidatedRequest(
        image_base64=image_base64,
        consent=consent,
        confidence_level=confidence_level,
        return_face_bbox=return_face_bbox,
        detector_hints=detector_hints,
    )


__all__ = ["ValidatedRequest", "validate_payload"]
