"""Request-preparation use case for the Responses API."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vfront.adapter.response import (
    RESPONSE_CONTEXT_NAMESPACE,
    input_to_messages,
    merge_response_tools,
    truncate_messages_for_context,
)
from vfront.frontend.api.v1.helper import prepare_response_generation
from vfront.frontend.api.v1.helper.response_execution import PreparedResponseGeneration
from vfront.frontend.middleware.exceptions import InvalidRequestError, NotFoundError
from vfront.frontend.service.engine.client_initializer import (
    InitializedEngineClient,
    resolve_engine_client,
)
from vfront.frontend.service.engine.router import EngineRouter
from vfront.frontend.service.mcp.client import MCPClientManager
from vfront.frontend.service.mcp.executor import requires_explicit_tool_parser_for_tools
from vfront.frontend.service.store.lmdb import StoreService
from vfront.frontend.service.tool_parsing.service import ToolParsingService
from vfront.protocol.response import (
    FunctionToolDefinition,
    ResponseCreateRequest,
    ResponseObject,
)
from vfront.shared.engine.reasoning import extract_reasoning_effort

SettingsGetter = Callable[[str], Any]
PreviousResponseValidator = Callable[[str, ResponseObject], None]
InputItemsBuilder = Callable[[str, list[dict[str, Any]]], list[dict[str, Any]]]
PersistResponseDecider = Callable[[ResponseCreateRequest], bool]


@dataclass(slots=True)
class PreparedResponsesRequest:
    """Prepared request state shared by the response creation paths."""

    response_id: str
    created_at: float
    effective_request: ResponseCreateRequest
    all_tools: list[FunctionToolDefinition] | None
    conversation_messages: list[dict[str, Any]]
    stored_input_items: list[dict[str, Any]]
    prepared_generation: PreparedResponseGeneration
    initialized_client: InitializedEngineClient
    use_agentic_tools: bool
    persist_response: bool


async def prepare_responses_request(
    *,
    request: ResponseCreateRequest,
    engine_router: EngineRouter,
    store: StoreService,
    mcp_manager: MCPClientManager | None,
    tool_parsing_service: ToolParsingService,
    store_namespace: str,
    get_settings: SettingsGetter,
    validate_previous_response: PreviousResponseValidator,
    build_input_items: InputItemsBuilder,
    should_persist_response: PersistResponseDecider,
    resolve_engine_client_fn: Callable[..., Any] | None = None,
) -> PreparedResponsesRequest:
    """Prepare a Responses request before background, streaming, or sync execution."""
    mcp_settings = get_settings("frontend.mcp")
    mcp_tool_defs = (
        mcp_manager.get_openai_tools()
        if mcp_settings.enabled and mcp_manager is not None
        else []
    )
    all_tools = merge_response_tools(request.tools, mcp_tool_defs)
    effective_request = request.model_copy(update={"tools": all_tools or None})

    require_tool_parser = requires_explicit_tool_parser_for_tools(
        all_tools,
        mcp_manager=mcp_manager if mcp_settings.enabled and mcp_manager is not None else None,
        name_getter=lambda tool: tool.name,
    )
    if resolve_engine_client_fn is None:
        initialized = await resolve_engine_client(
            model=request.model,
            engine_router=engine_router,
            tool_parsing_service=tool_parsing_service,
            require_tools=require_tool_parser,
        )
    else:
        initialized = await resolve_engine_client_fn(
            model=request.model,
            engine_router=engine_router,
            tool_parsing_service=tool_parsing_service,
            require_tools=require_tool_parser,
        )

    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    created_at = time.time()

    previous_response: ResponseObject | None = None
    previous_input_messages: list[dict[str, Any]] | None = None
    if request.previous_response_id:
        stored = await store.get(namespace=store_namespace, key=request.previous_response_id)
        if stored is None:
            raise NotFoundError(f"Response '{request.previous_response_id}' not found.")
        previous_response = ResponseObject.model_validate(stored)
        validate_previous_response(request.previous_response_id, previous_response)
        stored_context = await store.get(
            namespace=RESPONSE_CONTEXT_NAMESPACE,
            key=request.previous_response_id,
        )
        if isinstance(stored_context, list):
            previous_input_messages = stored_context

    conversation_messages = input_to_messages(
        request,
        previous_response,
        previous_input_messages,
    )

    runtime = initialized.client
    if request.truncation == "auto" and runtime.max_model_len > 0:
        output_reserve = request.max_output_tokens or max(1024, runtime.max_model_len // 4)
        available = runtime.max_model_len - output_reserve
        try:
            conversation_messages = await truncate_messages_for_context(
                conversation_messages,
                available,
                runtime,
                all_tools,
                reasoning_effort=extract_reasoning_effort(effective_request.reasoning),
            )
        except ValueError as exc:
            raise InvalidRequestError(str(exc), param="reasoning.effort") from exc

    stored_input_items = build_input_items(response_id, conversation_messages)
    prepared_generation = prepare_response_generation(
        request=effective_request,
        conversation_messages=conversation_messages,
        tools=all_tools,
        lora_name=initialized.adapter,
        tool_parser=initialized.target.tool_parser,
    )
    use_agentic_tools = mcp_settings.enabled and mcp_manager is not None and any(
        mcp_manager.is_mcp_tool(tool.name) for tool in (all_tools or [])
    )

    return PreparedResponsesRequest(
        response_id=response_id,
        created_at=created_at,
        effective_request=effective_request,
        all_tools=all_tools,
        conversation_messages=conversation_messages,
        stored_input_items=stored_input_items,
        prepared_generation=prepared_generation,
        initialized_client=initialized,
        use_agentic_tools=use_agentic_tools,
        persist_response=should_persist_response(request),
    )
