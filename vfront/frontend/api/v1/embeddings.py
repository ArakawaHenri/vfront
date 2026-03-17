"""POST /v1/embeddings — Embeddings endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapiex.di import Inject

from vfront.adapter.output import embedding_outputs_to_response
from vfront.adapter.pooling import (
    embedding_request_to_encode_params,
    normalize_embedding_inputs,
    resolve_embedding_encoding_format,
)
from vfront.frontend.service.engine.router import EngineRouter
from vfront.protocol.embedding import EmbeddingRequest, EmbeddingResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embedding(
    request: EmbeddingRequest,
    engine_router: EngineRouter = Inject("engine_router"),
):
    runtime = engine_router.resolve_client(request.model).client
    encode_params = embedding_request_to_encode_params(request)
    inputs = normalize_embedding_inputs(request.input)

    outputs = []
    for i, inp in enumerate(inputs):
        if isinstance(inp, str):
            prompt_token_ids = await runtime.tokenize_text(inp)
        else:
            prompt_token_ids = inp
        request_id = f"embd-{i}"
        result = await runtime.encode(prompt_token_ids, encode_params, request_id)
        outputs.append(result)

    encoding_format = resolve_embedding_encoding_format(request)
    return embedding_outputs_to_response(
        outputs,
        model=request.model,
        encoding_format=encoding_format,
    )
