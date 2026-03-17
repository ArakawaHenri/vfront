"""MCP client manager — manages connections to MCP tool servers."""

from __future__ import annotations

import os
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from fastapiex.di import BaseService, Service
from fastapiex.settings import GetSettings
from loguru import logger
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.types import Implementation, TextContent, Tool

if TYPE_CHECKING:
    from vfront.frontend.service.mcp.settings import MCPServerConfig


def _log_mcp_event(level: str, message: str, *args: object, event: str, **fields: object) -> None:
    getattr(logger.bind(event=event, **fields), level)(message, *args)


class MCPConnection:
    """A live connection to a single MCP server."""

    __slots__ = ("name", "session", "_tools")

    def __init__(self, name: str, session: ClientSession) -> None:
        self.name = name
        self.session = session
        self._tools: list[Tool] = []

    async def discover_tools(self) -> list[Tool]:
        """Fetch available tools from this server."""
        result = await self.session.list_tools()
        self._tools = result.tools
        _log_mcp_event(
            "info",
            "MCP server '{}': discovered {} tools: {}",
            self.name,
            len(self._tools),
            [t.name for t in self._tools],
            event="mcp.server.tools_discovered",
            server_name=self.name,
            tool_count=len(self._tools),
            tool_names=[t.name for t in self._tools],
        )
        return self._tools

    @property
    def tools(self) -> list[Tool]:
        return self._tools

    async def call_tool(self, name: str, arguments: dict[str, Any], timeout: int = 30) -> str:
        """Call a tool and return text result."""
        result = await self.session.call_tool(
            name=name,
            arguments=arguments,
            read_timeout_seconds=timedelta(seconds=timeout),
        )
        if result.isError:
            texts = [block.text for block in result.content if isinstance(block, TextContent)]
            error_msg = "\n".join(texts) if texts else "Tool execution failed"
            raise RuntimeError(f"MCP tool '{name}' error: {error_msg}")

        texts = [block.text for block in result.content if isinstance(block, TextContent)]
        return "\n".join(texts) if texts else ""


class MCPToolDefinition:
    """A tool definition combining MCP server info with the tool schema."""

    __slots__ = ("server_name", "tool_name", "description", "input_schema")

    def __init__(self, server_name: str, tool: Tool) -> None:
        self.server_name = server_name
        self.tool_name = tool.name
        self.description = tool.description or ""
        self.input_schema = tool.inputSchema

    @property
    def qualified_name(self) -> str:
        """Server-qualified tool name for disambiguation."""
        return f"mcp_{self.server_name}_{self.tool_name}"

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert to OpenAI function tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.qualified_name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@Service("mcp_client_manager", eager=True)
class MCPClientManager(BaseService):
    """Manages connections to all configured MCP servers.

    Lifecycle: create() → use → destroy()
    """

    def __init__(self) -> None:
        self._connections: dict[str, MCPConnection] = {}
        self._tool_map: dict[str, MCPToolDefinition] = {}  # qualified_name → def
        self._exit_stack: AsyncExitStack | None = None

    @classmethod
    async def create(cls) -> MCPClientManager:
        """DI factory — called by install_di during startup."""
        instance = cls()
        settings = GetSettings("frontend.mcp")
        if settings.enabled and settings.servers:
            await instance.start(settings.servers)
        else:
            _log_mcp_event(
                "info",
                "MCP disabled or no servers configured",
                event="mcp.lifecycle.disabled",
            )
        return instance

    @classmethod
    async def destroy(cls, instance: MCPClientManager) -> None:
        """DI teardown — called by install_di during shutdown."""
        await instance.shutdown()

    async def start(self, server_configs: list[MCPServerConfig]) -> None:
        """Connect to all configured MCP servers and discover tools."""
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        for config in server_configs:
            try:
                await self._connect_server(config)
            except Exception:
                _log_mcp_event(
                    "exception",
                    "Failed to connect to MCP server '{}'",
                    config.name,
                    event="mcp.server.connect_failed",
                    server_name=config.name,
                    transport=config.transport,
                )

    async def _connect_server(self, config: MCPServerConfig) -> None:
        """Connect to a single MCP server."""
        assert self._exit_stack is not None

        if config.transport == "stdio":
            env = {**os.environ, **config.env} if config.env else None
            server_params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=env,
            )
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
        elif config.transport == "streamable-http":
            if not config.url:
                raise ValueError(
                    f"MCP server '{config.name}': url is required for streamable-http transport"
                )
            http_client = create_mcp_http_client(
                headers=config.headers or None,
            )
            read_stream, write_stream, _get_session_id = await self._exit_stack.enter_async_context(
                streamable_http_client(
                    url=config.url,
                    http_client=http_client,
                    terminate_on_close=True,
                )
            )
        else:
            raise ValueError(f"Unsupported MCP transport: {config.transport}")

        session = ClientSession(
            read_stream,
            write_stream,
            client_info=Implementation(name="vfront", version="0.1.0"),
        )
        await session.initialize()

        conn = MCPConnection(name=config.name, session=session)
        await conn.discover_tools()

        self._connections[config.name] = conn

        # Register tools in the map
        for tool in conn.tools:
            tool_def = MCPToolDefinition(server_name=config.name, tool=tool)
            self._tool_map[tool_def.qualified_name] = tool_def

        _log_mcp_event(
            "info",
            "Connected to MCP server '{}'",
            config.name,
            event="mcp.server.connected",
            server_name=config.name,
            transport=config.transport,
            tool_count=len(conn.tools),
        )

    async def shutdown(self) -> None:
        """Close all MCP server connections."""
        exit_stack = self._exit_stack
        self._exit_stack = None
        closed_server_count = len(self._connections)
        cleared_tool_count = len(self._tool_map)
        close_error: Exception | None = None
        try:
            if exit_stack is not None:
                await exit_stack.aclose()
        except Exception as exc:
            close_error = exc
        finally:
            self._connections.clear()
            self._tool_map.clear()
        if close_error is not None:
            _log_mcp_event(
                "error",
                "Failed to close MCP connections cleanly",
                event="mcp.lifecycle.shutdown.failed",
                server_count=closed_server_count,
                tool_count=cleared_tool_count,
            )
            raise close_error
        _log_mcp_event(
            "info",
            "All MCP connections closed",
            event="mcp.lifecycle.shutdown.complete",
            server_count=closed_server_count,
            tool_count=cleared_tool_count,
        )

    @property
    def tool_definitions(self) -> list[MCPToolDefinition]:
        """All discovered MCP tool definitions."""
        return list(self._tool_map.values())

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Return all MCP tools in OpenAI function tool format."""
        return [td.to_openai_tool() for td in self._tool_map.values()]

    def is_mcp_tool(self, function_name: str) -> bool:
        """Check if a function name belongs to an MCP tool."""
        return function_name in self._tool_map

    async def execute_tool(
        self, function_name: str, arguments: dict[str, Any], timeout: int = 30
    ) -> str:
        """Execute an MCP tool by its qualified name."""
        tool_def = self._tool_map.get(function_name)
        if tool_def is None:
            raise ValueError(f"Unknown MCP tool: {function_name}")

        conn = self._connections.get(tool_def.server_name)
        if conn is None:
            raise RuntimeError(f"MCP server '{tool_def.server_name}' not connected")

        return await conn.call_tool(tool_def.tool_name, arguments, timeout)

    @property
    def connected_servers(self) -> list[str]:
        """Names of all connected MCP servers."""
        return list(self._connections.keys())
