"""MCP integration settings."""

from __future__ import annotations

from fastapiex.settings import BaseSettings, Settings
from pydantic import Field


class MCPServerConfig(BaseSettings):
    name: str
    transport: str = "stdio"
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)


@Settings("frontend.mcp")
class MCPSettings(BaseSettings):
    enabled: bool = False
    servers: list[MCPServerConfig] = Field(default_factory=list)
    max_tool_rounds: int = 10
    timeout_seconds: int = 30
