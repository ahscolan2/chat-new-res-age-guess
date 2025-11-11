import base64

import pytest

from age_service.config import load_config
from age_service.detector import BaseFaceDetector, FaceBoundingBox, FaceDetection
from age_service.exceptions import InferenceError, ValidationError
from age_service.model import AgePrediction
from age_service.validation import validate_payload
from api import run_inference


CONFIG_PATH = "config/age_service.json"


def _encode_image(size=(256, 256), intensity=180) -> str:
    width, height = size
    payload = f"{width},{height},{float(intensity)}"
    return base64.b64encode(payload.encode("ascii")).decode("ascii")


class DetectorStub(BaseFaceDetector):
    name = "stub"

    def __init__(self, detections):
        self._detections = detections

    def detect_faces(self, image, *, hints=None):  # type: ignore[override]
        return list(self._detections)


class ModelStub:
    model_name = "stub-model"
    model_version = "0.0.1"

    def __init__(self, value: float, uncertainty: float) -> None:
        self._prediction = AgePrediction(value=value, uncertainty=uncertainty)

    def predict(self, image):  # type: ignore[override]
        return self._prediction


def test_single_face_success():
    bbox = FaceBoundingBox(x=10, y=10, width=100, height=120)
    detection = FaceDetection(bounding_box=bbox, confidence=0.95, occluded=False)
    detector = DetectorStub([detection])
    model = ModelStub(value=32.5, uncertainty=3.0)

    payload = {
        "image_base64": _encode_image(),
        "consent": True,
        "confidence_level": 0.9,
        "return_face_bbox": True,
    }

    response = run_inference(payload, config_path=CONFIG_PATH, detector=detector, model=model)

    assert response["status"] == "success"
    assert pytest.approx(response["age"]["value"], rel=1e-6) == 32.5
    assert response["age"]["confidence_interval"]["level"] == 0.9
    assert response["face_bbox"] == {
        "x": bbox.x,
        "y": bbox.y,
        "width": bbox.width,
        "height": bbox.height,
    }
    metadata = response["metadata"]
    assert metadata["model"]["name"] == "CalibratedLinearNet"
    assert metadata["fairness_warnings"]
    assert metadata["limitations"]


def test_reject_no_faces():
    detector = DetectorStub([])
    model = ModelStub(value=30.0, uncertainty=2.0)
    payload = {
        "image_base64": _encode_image(),
        "consent": True,
    }

    with pytest.raises(InferenceError) as exc:
        run_inference(payload, config_path=CONFIG_PATH, detector=detector, model=model)
    assert exc.value.reason == "no_faces"


def test_reject_multiple_faces():
    bbox = FaceBoundingBox(x=10, y=10, width=90, height=90)
    detections = [
        FaceDetection(bounding_box=bbox, confidence=0.9, occluded=False),
        FaceDetection(bounding_box=bbox, confidence=0.8, occluded=False),
    ]
    detector = DetectorStub(detections)
    model = ModelStub(value=30.0, uncertainty=2.0)
    payload = {
        "image_base64": _encode_image(),
        "consent": True,
    }

    with pytest.raises(InferenceError) as exc:
        run_inference(payload, config_path=CONFIG_PATH, detector=detector, model=model)
    assert exc.value.reason == "multiple_faces"


def test_reject_occluded_face():
    bbox = FaceBoundingBox(x=5, y=5, width=100, height=100)
    detections = [FaceDetection(bounding_box=bbox, confidence=0.9, occluded=True)]
    detector = DetectorStub(detections)
    model = ModelStub(value=30.0, uncertainty=2.0)
    payload = {
        "image_base64": _encode_image(),
        "consent": True,
    }

    with pytest.raises(InferenceError) as exc:
        run_inference(payload, config_path=CONFIG_PATH, detector=detector, model=model)
    assert exc.value.reason == "occluded"


def test_reject_low_resolution_image():
    detector = DetectorStub([])
    model = ModelStub(value=30.0, uncertainty=2.0)
    payload = {
        "image_base64": _encode_image(size=(64, 64)),
        "consent": True,
    }

    with pytest.raises(ValidationError) as exc:
        run_inference(payload, config_path=CONFIG_PATH, detector=detector, model=model)
    assert exc.value.args[0].startswith("image must be at least")


def test_validation_schema_enforcement():
    config = load_config(CONFIG_PATH)

    with pytest.raises(ValidationError):
        validate_payload({}, config)

    with pytest.raises(ValidationError):
        validate_payload({"image_base64": "", "consent": False}, config)

    with pytest.raises(ValidationError):
        validate_payload({"image_base64": "abc", "consent": True, "confidence_level": 1.5}, config)
