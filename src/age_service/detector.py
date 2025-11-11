"""Face detection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .image import SimpleImage


@dataclass(slots=True)
class FaceBoundingBox:
    x: int
    y: int
    width: int
    height: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


@dataclass(slots=True)
class FaceDetection:
    bounding_box: FaceBoundingBox
    confidence: float
    occluded: bool = False


class BaseFaceDetector:
    """Abstract base class for face detectors."""

    name = "base"

    def detect_faces(
        self,
        image: SimpleImage,
        *,
        hints: Sequence[FaceBoundingBox] | None = None,
    ) -> List[FaceDetection]:
        raise NotImplementedError


class SimpleFaceDetector(BaseFaceDetector):
    """Detector used for tests that honours externally provided hints."""

    name = "simple"

    def detect_faces(
        self,
        image: SimpleImage,
        *,
        hints: Sequence[FaceBoundingBox] | None = None,
    ) -> List[FaceDetection]:
        if hints:
            return [FaceDetection(bounding_box=hint, confidence=0.9) for hint in hints]

        width, height = image.size
        box = FaceBoundingBox(
            x=int(width * 0.1),
            y=int(height * 0.1),
            width=int(width * 0.8),
            height=int(height * 0.8),
        )
        return [FaceDetection(bounding_box=box, confidence=0.5)]


def crop_to_face(image: SimpleImage, detection: FaceDetection) -> SimpleImage:
    """Crop the image to the supplied detection bounding box."""

    x, y, w, h = detection.bounding_box.as_tuple()
    return image.crop_box(x, y, w, h)


__all__ = [
    "FaceBoundingBox",
    "FaceDetection",
    "BaseFaceDetector",
    "SimpleFaceDetector",
    "crop_to_face",
]
