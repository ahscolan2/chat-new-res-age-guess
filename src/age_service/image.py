"""Image loading and normalisation helpers.

The reference implementation uses a lightweight internal ``SimpleImage`` class
rather than relying on an external imaging library. The base64 payload is
expected to encode a comma separated string ``width,height,intensity`` where the
intensity represents a uniform greyscale value between 0 and 255.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Tuple


@dataclass(slots=True)
class SimpleImage:
    width: int
    height: int
    intensity: float

    @property
    def size(self) -> Tuple[int, int]:
        return self.width, self.height

    def crop_box(self, x: int, y: int, width: int, height: int) -> "SimpleImage":
        new_width = max(1, min(width, self.width - x))
        new_height = max(1, min(height, self.height - y))
        return SimpleImage(width=new_width, height=new_height, intensity=self.intensity)

    def resize(self, size: Tuple[int, int]) -> "SimpleImage":
        width, height = size
        return SimpleImage(width=width, height=height, intensity=self.intensity)

    def mean_intensity(self) -> float:
        return self.intensity


@dataclass(slots=True)
class LoadedImage:
    image: SimpleImage
    raw_bytes: bytes

    @property
    def size(self) -> Tuple[int, int]:
        return self.image.size


def _parse_payload(payload: str) -> SimpleImage:
    try:
        width_str, height_str, intensity_str = payload.split(",")
        width = int(width_str)
        height = int(height_str)
        intensity = float(intensity_str)
    except (ValueError, TypeError) as exc:
        raise ValueError("image payload must be width,height,intensity") from exc
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0 <= intensity <= 255:
        raise ValueError("intensity must be between 0 and 255")
    return SimpleImage(width=width, height=height, intensity=intensity)


def load_base64_image(data: str) -> LoadedImage:
    """Decode a base64 encoded image into a :class:`SimpleImage`."""

    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid base64 encoded image") from exc

    try:
        payload = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Image payload must be ASCII") from exc

    image = _parse_payload(payload)
    return LoadedImage(image=image, raw_bytes=raw)


def normalise_image(image: SimpleImage) -> SimpleImage:
    """Normalise the image into model input space.

    The reference implementation simply resizes the image to 224x224 pixels.
    """

    return image.resize((224, 224))


__all__ = ["SimpleImage", "LoadedImage", "load_base64_image", "normalise_image"]
