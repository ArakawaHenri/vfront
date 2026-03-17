"""Background response-generation use case."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from vfront.frontend.service.mcp.executor import requires_explicit_tool_parser_for_tools
from vfront.protocol.response import ResponseObject

if TYPE_CHECKING:
    from vfront.frontend.service.mcp.client import MCPClientManager
    from vfront.frontend.service.store.lmdb import StoreService
    from vfront.frontend.service.tool_parsing.service import ToolParsingService
    from vfront.protocol.job import ResponseJobSpec

logger = logging.getLogger(__name__)

SettingsGetter = Callable[[str], Any]
PersistResponseDecider = Callable[[Any], bool]
InputItemsBuilder = Callable[[str, list[dict[str, Any]]], list[dict[str, Any]]]
ResolveClientForTarget = Callable[[Any], Any]
InitializeEngineClient = Callable[..., Awaitable[Any]]
PrepareResponseGeneration = Callable[..., Any]
ExecuteResponseGeneration = Callable[..., Awaitable[Any]]
BuildResponseFromGeneration = Callable[..., Any]
PersistResponseState = Callable[..., Awaitable[None]]
BuildFailedResponse = Callable[..., ResponseObject]


async def background_generate_response_use_case(
    *,
    spec: ResponseJobSpec,
    store: StoreService,
    mcp_manager: MCPClientManager,
    tool_parsing_service: ToolParsingService,
    resolve_client_for_target: ResolveClientForTarget,
    get_settings: SettingsGetter,
    should_persist_response: PersistResponseDecider,
    build_input_items: InputItemsBuilder,
    initialize_engine_client: InitializeEngineClient,
    prepare_response_generation: PrepareResponseGeneration,
    execute_response_generation: ExecuteResponseGeneration,
    build_response_from_generation: BuildResponseFromGeneration,
    persist_response_state: PersistResponseState,
    build_failed_response: BuildFailedResponse,
    store_namespace: str,
) -> None:
    """Run response generation in the background, persisting state transitions when enabled."""
    request = spec.request
    response_id = spec.response_id
    created_at = spec.created_at
    conversation_messages = spec.conversation_messages
    stored_input_items = build_input_items(response_id, conversation_messages)
    initialized = await initialize_engine_client(
        resolved=resolve_client_for_target(spec.target),
        tool_parsing_service=tool_parsing_service,
        require_tools=requires_explicit_tool_parser_for_tools(
            request.tools,
            mcp_manager=mcp_manager if get_settings("frontend.mcp").enabled else None,
            name_getter=lambda tool: tool.name,
        ),
    )
    runtime = initialized.client
    persist_response = should_persist_response(request)
    use_agentic_tools = any(mcp_manager.is_mcp_tool(tool.name) for tool in (request.tools or []))
    prepared = prepare_response_generation(
        request=request,
        conversation_messages=conversation_messages,
        tools=request.tools,
        lora_name=spec.target.adapter,
        tool_parser=spec.target.tool_parser,
    )
    try:
        if persist_response:
            stored = await store.get(namespace=store_namespace, key=response_id)
            if stored and isinstance(stored, dict):
                response = ResponseObject.model_validate(stored).model_copy(
                    update={"status": "in_progress"}
                )
                await persist_response_state(
                    store,
                    response,
                    input_items=stored_input_items,
                )

        mcp_settings = get_settings("frontend.mcp")
        generation_result = await execute_response_generation(
            conversation_messages=conversation_messages,
            generation_messages=prepared.generation_messages,
            tools=request.tools,
            parallel_tool_calls=request.parallel_tool_calls,
            max_tool_calls=request.max_tool_calls,
            runtime=runtime,
            params=prepared.params,
            response_id=response_id,
            use_agentic_tools=use_agentic_tools,
            mcp_manager=mcp_manager,
            max_tool_rounds=mcp_settings.max_tool_rounds,
            timeout=mcp_settings.timeout_seconds,
        )
        response = build_response_from_generation(
            engine_output=generation_result.final_output,
            request=request,
            response_id=response_id,
            created_at=created_at,
            tool_executions=generation_result.tool_executions,
            pending_client_calls=generation_result.pending_client_calls,
        )

        if persist_response:
            await persist_response_state(
                store=store,
                response=response,
                context_messages=generation_result.final_messages,
                input_items=stored_input_items,
            )
    except asyncio.CancelledError:
        logger.info("Background response cancelled: %s", response_id)
        raise
    except Exception:
        logger.exception("Background response failed: %s", response_id)
        if persist_response:
            await persist_response_state(
                store,
                build_failed_response(
                    response_id=response_id,
                    created_at=created_at,
                    request=request,
                    code="server_error",
                    message="Internal error",
                ),
                input_items=stored_input_items,
            )
        raise
