"""Model protocol — strictly matches OpenAI spec."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "system"
    permission: list[dict] = []
    context_window: int | None = None
    capabilities: list[str] = Field(default_factory=list)


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]


class ModelDeleted(BaseModel):
    id: str
    object: Literal["model"] = "model"
    deleted: bool = True
