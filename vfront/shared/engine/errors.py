"""Shared engine-facing error types."""

from __future__ import annotations


class EngineInputError(Exception):
    """Backend rejected request input during preprocessing/tokenization."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)
