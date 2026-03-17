"""POST /v1/completions — Legacy Completions endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapiex.di import Inject
from starlette.responses import StreamingResponse

from vfront.adapter.streaming import stream_completion
from vfront.frontend.api.v1.helper import (
    params_for_completion_prompt,
    transform_completion_output_stream,
)
from vfront.frontend.service.engine.client_initializer import resolve_engine_client
from vfront.frontend.service.engine.router import EngineRouter
from vfront.frontend.service.tool_parsing.service import ToolParsingService
from vfront.frontend.usecases.completions import (
    execute_non_streaming_completion,
    prepare_completion_use_case,
)
from vfront.protocol.completion import CompletionRequest, CompletionResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/completions", response_model=CompletionResponse)
async def create_completion(
    request: CompletionRequest,
    engine_router: EngineRouter = Inject("engine_router"),
    tool_parsing_service: ToolParsingService = Inject("tool_parsing_service"),
):
    initialized = await resolve_engine_client(
        model=request.model,
        engine_router=engine_router,
        tool_parsing_service=tool_parsing_service,
    )
    prepared = await prepare_completion_use_case(
        request=request,
        runtime=initialized.client,
        adapter=initialized.adapter,
        system_fingerprint=initialized.system_fingerprint,
    )

    if request.stream:
        prompt = prepared.prepared_prompts[0]
        generator = transform_completion_output_stream(
            prepared.runtime.generate(
                prompt.prompt_token_ids,
                params_for_completion_prompt(base=prepared.params, prompt=prompt),
                prepared.request_id,
            ),
            prompt=prompt,
            suffix=request.suffix,
        )
        include_usage = request.stream_options is not None and request.stream_options.include_usage
        return StreamingResponse(
            stream_completion(
                request_id=prepared.request_id,
                model=request.model,
                output_generator=generator,
                include_usage=include_usage,
                n=request.n or 1,
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

    return await execute_non_streaming_completion(
        request=request,
        prepared=prepared,
    )
