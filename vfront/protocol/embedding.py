"""Embedding protocol — strictly matches OpenAI spec."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from vfront.protocol.common import UsageInfo


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str] | list[int] | list[list[int]]
    encoding_format: str | None = None
    dimensions: int | None = None
    user: str | None = None


class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float] | str
    index: int


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: UsageInfo
