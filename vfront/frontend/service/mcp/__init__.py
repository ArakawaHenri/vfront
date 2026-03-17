"""Frontend MCP service package.

Import settings at package load so bare GetSettings("frontend.mcp") resolves for MCP
service consumers.
"""

from vfront.frontend.service.mcp import settings as _settings  # noqa: F401
