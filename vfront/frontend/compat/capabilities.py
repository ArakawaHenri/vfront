"""Server capability flags — single source of truth for what this server supports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ServerCapabilities:
    # Chat Completions
    supports_chat_audio: bool = False
    supports_chat_web_search: bool = False
    supports_chat_modalities_non_text: bool = False
    """Whether the chat endpoint accepts modalities other than ["text"]."""

    # Responses API
    supports_response_conversation: bool = False
    """Whether the Responses API handles the 'conversation' continuation field."""
    supports_response_include_selectors: bool = False
    """Whether the Responses API handles the 'include' selectors field."""
    supports_response_file_input: bool = False
    supports_response_builtin_tools: bool = False

    # Legacy Completions
    supports_completion_multi_prompt: bool = True
    """Whether /v1/completions accepts prompt as a list with len > 1."""
    supports_completion_stream_multi_prompt: bool = False
    """Whether /v1/completions accepts streaming with multiple prompts."""


DEFAULT_CAPABILITIES = ServerCapabilities()
