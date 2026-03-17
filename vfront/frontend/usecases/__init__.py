"""Frontend use-case layer."""

from vfront.frontend.usecases.batches import (
    build_batch_job_spec,
    build_batch_request_targets,
    create_batch_use_case,
    process_batch_use_case,
)
from vfront.frontend.usecases.chat_completion_agentic import (
    PreparedChatAgenticRequest,
    detect_tool_calls_from_completion,
    run_non_streaming_agentic_chat_completion,
    stream_agentic_chat_completion,
)
from vfront.frontend.usecases.chat_completion_creation import (
    PreparedChatCompletionRequest,
    create_chat_completion_use_case,
    prepare_chat_completion_request,
)
from vfront.frontend.usecases.responses_admin import (
    cancel_response_use_case,
    delete_response_use_case,
    list_response_input_items_use_case,
)
from vfront.frontend.usecases.responses_background import (
    background_generate_response_use_case,
)
from vfront.frontend.usecases.responses_creation import (
    PreparedResponseCreation,
    create_background_response_use_case,
    create_non_streaming_response_use_case,
)
from vfront.frontend.usecases.responses_request_preparation import (
    PreparedResponsesRequest,
    prepare_responses_request,
)
from vfront.frontend.usecases.responses_streaming_agentic import (
    PreparedStreamingAgenticResponse,
    StreamingAgenticCallbacks,
    stream_agentic_response,
)
from vfront.frontend.usecases.responses_streaming_coordinator import (
    PersistedStreamingCallbacks,
    PreparedPersistedStreamingResponse,
    stream_response_with_persistence,
)
from vfront.frontend.usecases.responses_streaming_direct import (
    PreparedDirectStreamingResponse,
    StreamingDirectCallbacks,
    stream_direct_response,
)

__all__ = [
    "PersistedStreamingCallbacks",
    "PreparedChatAgenticRequest",
    "PreparedChatCompletionRequest",
    "PreparedDirectStreamingResponse",
    "PreparedPersistedStreamingResponse",
    "PreparedResponseCreation",
    "PreparedResponsesRequest",
    "PreparedStreamingAgenticResponse",
    "StreamingAgenticCallbacks",
    "StreamingDirectCallbacks",
    "background_generate_response_use_case",
    "build_batch_job_spec",
    "build_batch_request_targets",
    "cancel_response_use_case",
    "create_background_response_use_case",
    "create_batch_use_case",
    "create_chat_completion_use_case",
    "create_non_streaming_response_use_case",
    "delete_response_use_case",
    "detect_tool_calls_from_completion",
    "list_response_input_items_use_case",
    "prepare_chat_completion_request",
    "prepare_responses_request",
    "process_batch_use_case",
    "run_non_streaming_agentic_chat_completion",
    "stream_agentic_chat_completion",
    "stream_agentic_response",
    "stream_direct_response",
    "stream_response_with_persistence",
]
