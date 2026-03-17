"""Translate OpenAI embedding params to engine-agnostic EncodeParams."""

from __future__ import annotations

from typing import TypeGuard

from vfront.frontend.middleware.exceptions import InvalidRequestError
from vfront.protocol.embedding import EmbeddingRequest
from vfront.shared.engine.types import EncodeParams


def embedding_request_to_encode_params(
    request: EmbeddingRequest,
) -> EncodeParams:
    return EncodeParams(dimensions=request.dimensions)


def normalize_embedding_inputs(
    input_value: str | list[str] | list[int] | list[list[int]],
) -> list[str | list[int]]:
    """Normalize embedding inputs to a list of per-item payloads."""
    if isinstance(input_value, str):
        return [input_value]
    if isinstance(input_value, list):
        if not input_value:
            raise InvalidRequestError("Input list must not be empty.")
        if _is_token_id_list(input_value):
            return [input_value]
        if _is_embedding_input_sequence(input_value):
            return input_value
        return [str(input_value)]
    return [str(input_value)]


def resolve_embedding_encoding_format(request: EmbeddingRequest) -> str:
    """Validate and resolve the public embedding encoding format."""
    encoding_format = request.encoding_format or "float"
    if encoding_format not in ("float", "base64"):
        raise InvalidRequestError(
            f"Invalid encoding_format '{encoding_format}'. Must be 'float' or 'base64'."
        )
    return encoding_format


def _is_token_id_list(value: object) -> TypeGuard[list[int]]:
    return isinstance(value, list) and all(isinstance(item, int) for item in value)


def _is_embedding_input_sequence(value: object) -> TypeGuard[list[str | list[int]]]:
    return isinstance(value, list) and all(
        isinstance(item, str) or _is_token_id_list(item) for item in value
    )
