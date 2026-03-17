"""Create Chat Completions use case."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from starlette.responses import StreamingResponse

from vfront.adapter.output import generate_request_id
from vfront.adapter.sampling import chat_request_to_generate_params, extract_multimodal_data
from vfront.adapter.streaming import stream_chat_completion
from vfront.frontend.middleware.exceptions import InvalidRequestError
from vfront.frontend.service.engine.client_initializer import resolve_engine_client
from vfront.frontend.service.mcp.executor import requires_explicit_tool_parser_for_tools
from vfront.frontend.usecases.chat_completion_agentic import (
    PreparedChatAgenticRequest,
    run_non_streaming_agentic_chat_completion,
    stream_agentic_chat_completion,
)

SettingsGetter = Callable[[str], Any]
ExecuteMcpToolCall = Callable[..., Awaitable[Any]]
StoreCompletion = Callable[..., Awaitable[None]]
RequestOutputToChatResponse = Callable[..., Any]
CollectFinalStreamOutput = Callable[..., Awaitable[Any]]
TokenizeChatMessages = Callable[..., Awaitable[list[int]]]
ResolveEngineClient = Callable[..., Awaitable[Any]]


def _tool_name_getter(tool: dict[str, Any]) -> str | None:
    function_data = tool.get("function")
    if isinstance(function_data, dict):
        name = function_data.get("name")
        return name if isinstance(name, str) else None
    return None


@dataclass(slots=True)
class PreparedChatCompletionRequest:
    request_id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    params: Any
    runtime: Any
    adapter: str | None
    system_fingerprint: str | None
    prepared_agentic_request: PreparedChatAgenticRequest
    use_agentic_tools: bool
    include_usage: bool


async def prepare_chat_completion_request(
    *,
    request: Any,
    engine_router: Any,
    mcp_manager: Any | None,
    tool_parsing_service: Any,
    get_settings: SettingsGetter,
    tokenize_chat_messages: TokenizeChatMessages,
    resolve_engine_client_fn: ResolveEngineClient | None = None,
) -> PreparedChatCompletionRequest:
    request_id = generate_request_id()

    messages = [msg.model_dump(exclude_none=True) for msg in request.messages]
    messages = [
        {**message, "role": "system"} if message.get("role") == "developer" else message
        for message in messages
    ]
    tools = [tool.model_dump() for tool in request.tools] if request.tools else None

    mcp_settings = get_settings("frontend.mcp")
    mcp_tools = (
        mcp_manager.get_openai_tools()
        if mcp_settings.enabled and mcp_manager is not None
        else []
    )
    if mcp_tools:
        tools = (tools or []) + mcp_tools
    require_tool_parser = requires_explicit_tool_parser_for_tools(
        tools,
        mcp_manager=mcp_manager if mcp_settings.enabled and mcp_manager is not None else None,
        name_getter=_tool_name_getter,
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
    resolved = initialized.resolved
    runtime = initialized.client

    try:
        prompt_token_ids = await tokenize_chat_messages(
            runtime,
            messages,
            tools=tools,
            reasoning_effort=request.reasoning_effort,
        )
    except ValueError as exc:
        raise InvalidRequestError(str(exc), param="reasoning_effort") from exc
    max_len = runtime.max_model_len
    if max_len > 0 and len(prompt_token_ids) > max_len:
        raise InvalidRequestError(
            f"Prompt length {len(prompt_token_ids)} tokens exceeds model context window {max_len}.",
            code="context_length_exceeded",
        )

    params = chat_request_to_generate_params(request)
    params.lora_name = resolved.adapter
    params.multimodal_data = extract_multimodal_data(messages)
    params.tools = tools
    params.tool_choice = request.tool_choice
    params.tool_parser = resolved.target.tool_parser

    system_fingerprint = initialized.system_fingerprint
    prepared_agentic_request = PreparedChatAgenticRequest(
        request_id=request_id,
        model=request.model,
        n=request.n or 1,
        reasoning_effort=request.reasoning_effort,
        prompt_token_ids=prompt_token_ids,
        messages=messages,
        tools=tools,
        params=params,
        runtime=runtime,
        adapter=resolved.adapter,
        system_fingerprint=system_fingerprint,
        mcp_manager=mcp_manager,
        mcp_settings=mcp_settings,
    )

    return PreparedChatCompletionRequest(
        request_id=request_id,
        messages=messages,
        tools=tools,
        params=params,
        runtime=runtime,
        adapter=resolved.adapter,
        system_fingerprint=system_fingerprint,
        prepared_agentic_request=prepared_agentic_request,
        use_agentic_tools=bool(mcp_settings.enabled and mcp_tools),
        include_usage=bool(
            request.stream_options is not None and request.stream_options.include_usage
        ),
    )


async def create_chat_completion_use_case(
    *,
    request: Any,
    prepared: PreparedChatCompletionRequest,
    store: Any,
    execute_mcp_tool_call_fn: ExecuteMcpToolCall,
    request_output_to_chat_response_fn: RequestOutputToChatResponse,
    collect_final_stream_output_fn: CollectFinalStreamOutput,
    store_completion_fn: StoreCompletion,
) -> Any:
    if request.stream:
        if prepared.use_agentic_tools:
            return StreamingResponse(
                stream_agentic_chat_completion(
                    prepared=prepared.prepared_agentic_request,
                    execute_mcp_tool_call_fn=execute_mcp_tool_call_fn,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        generator = prepared.runtime.generate(
            prepared.prepared_agentic_request.prompt_token_ids,
            prepared.params,
            prepared.request_id,
        )
        return StreamingResponse(
            stream_chat_completion(
                request_id=prepared.request_id,
                model=request.model,
                output_generator=generator,
                include_usage=prepared.include_usage,
                n=request.n or 1,
                has_tool_definitions=bool(prepared.tools),
                semantic_tool_parsing=bool(request.tools),
                lora_name=prepared.adapter,
                system_fingerprint=prepared.system_fingerprint,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    final_output = await collect_final_stream_output_fn(
        prepared.runtime.generate(
            prepared.prepared_agentic_request.prompt_token_ids,
            prepared.params,
            prepared.request_id,
        ),
        empty_error_message="Engine returned no output",
    )

    if prepared.use_agentic_tools and final_output.outputs:
        final_output = await run_non_streaming_agentic_chat_completion(
            prepared=prepared.prepared_agentic_request,
            initial_output=final_output,
            execute_mcp_tool_call_fn=execute_mcp_tool_call_fn,
        )

    response = request_output_to_chat_response_fn(
        final_output,
        model=request.model,
        request_id=prepared.request_id,
        has_tool_definitions=bool(prepared.tools),
        semantic_tool_parsing=bool(request.tools),
        lora_name=prepared.adapter,
        system_fingerprint=prepared.system_fingerprint,
    )

    if request.store:
        assistant_turn = response.choices[0].message.model_dump(exclude_none=True)
        await store_completion_fn(store, request, response, [*prepared.messages, assistant_turn])

    return response
