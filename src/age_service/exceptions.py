"""Custom exceptions for the age estimation service."""

from __future__ import annotations


class AgeServiceError(RuntimeError):
    """Base class for domain specific errors."""


class ValidationError(AgeServiceError):
    """Raised when input payload validation fails."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class InferenceError(AgeServiceError):
    """Raised when inference cannot be completed."""

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason
