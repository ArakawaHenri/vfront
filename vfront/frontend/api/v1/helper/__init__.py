"""Shared helpers for API v1 route implementations."""

from vfront.frontend.api.v1.helper.completion_execution import (
    finalize_completion_generation,
    prepare_completion_generation,
    validate_completion_best_of,
)
from vfront.frontend.api.v1.helper.completion_infill import (
    params_for_completion_prompt,
    prepare_completion_prompts,
    transform_completion_output_stream,
)
from vfront.frontend.api.v1.helper.indexed_store import (
    build_created_at_index_key,
    ensure_created_at_index,
    list_created_at_models,
    store_created_at_indexed_model,
    upsert_created_at_index,
)
from vfront.frontend.api.v1.helper.response_execution import (
    build_response_from_generation,
    execute_response_generation,
    prepare_response_generation,
    tokenize_response_messages,
)

__all__ = [
    "build_created_at_index_key",
    "build_response_from_generation",
    "ensure_created_at_index",
    "execute_response_generation",
    "finalize_completion_generation",
    "list_created_at_models",
    "params_for_completion_prompt",
    "prepare_completion_generation",
    "prepare_completion_prompts",
    "prepare_response_generation",
    "store_created_at_indexed_model",
    "tokenize_response_messages",
    "transform_completion_output_stream",
    "upsert_created_at_index",
    "validate_completion_best_of",
]
