"""Inference entry point for the age estimation service."""

from __future__ import annotations

from typing import Dict

from .config import AgeServiceConfig, load_config
from .detector import BaseFaceDetector, SimpleFaceDetector, crop_to_face
from .exceptions import InferenceError, ValidationError
from .image import load_base64_image, normalise_image
from .model import AgeModel, CalibratedLinearModel, confidence_interval
from .validation import validate_payload


def _resolve_detector(config: AgeServiceConfig, provided: BaseFaceDetector | None) -> BaseFaceDetector:
    if provided is not None:
        return provided
    if config.detector_name == SimpleFaceDetector.name:
        return SimpleFaceDetector()
    return SimpleFaceDetector()


def _resolve_model(config: AgeServiceConfig, provided: AgeModel | None) -> AgeModel:
    if provided is not None:
        return provided
    return CalibratedLinearModel(slope=60.0, intercept=5.0, residual_std=7.5)


def _validate_image_size(image, config: AgeServiceConfig) -> None:
    width, height = image.size
    if min(width, height) < config.min_image_edge:
        raise ValidationError(
            f"image must be at least {config.min_image_edge}px on each edge",
            field="image_base64",
        )


def run_inference(
    payload: Dict[str, object],
    *,
    config_path: str | None = None,
    detector: BaseFaceDetector | None = None,
    model: AgeModel | None = None,
) -> Dict[str, object]:
    """Run inference given a JSON payload."""

    config = load_config(config_path)
    request = validate_payload(payload, config)

    try:
        loaded = load_base64_image(request.image_base64)
    except ValueError as exc:
        raise ValidationError(str(exc), field="image_base64") from exc

    _validate_image_size(loaded.image, config)

    detector_impl = _resolve_detector(config, detector)
    detections = detector_impl.detect_faces(loaded.image, hints=request.detector_hints)

    if not detections:
        raise InferenceError("no faces detected", reason="no_faces")
    if len(detections) > 1:
        raise InferenceError("multiple faces detected", reason="multiple_faces")

    detection = detections[0]
    if getattr(detection, "occluded", False):
        raise InferenceError("primary face appears occluded", reason="occluded")

    cropped = crop_to_face(loaded.image, detection)
    normalised = normalise_image(cropped)

    model_impl = _resolve_model(config, model)
    prediction = model_impl.predict(normalised)

    lower, upper = confidence_interval(prediction, request.confidence_level)

    response: Dict[str, object] = {
        "status": "success",
        "age": {
            "value": prediction.value,
            "confidence_interval": {
                "level": request.confidence_level,
                "lower": lower,
                "upper": upper,
            },
        },
        "metadata": {
            "detector": detector_impl.name,
            "model": {
                "name": config.model_metadata.name,
                "version": config.model_metadata.version,
                "mean_absolute_error": config.model_metadata.mean_absolute_error,
                "calibration_date": config.model_metadata.calibration_date,
            },
            "fairness_warnings": list(config.model_metadata.fairness_warnings),
            "limitations": list(config.model_metadata.limitations),
        },
    }

    if request.return_face_bbox:
        bbox = detection.bounding_box.as_tuple()
        response["face_bbox"] = {
            "x": bbox[0],
            "y": bbox[1],
            "width": bbox[2],
            "height": bbox[3],
        }

    return response


__all__ = ["run_inference"]
