"""Engine-side semantic reasoning parsing backed directly by vLLM parsers."""

from __future__ import annotations

from vllm.reasoning import ReasoningParserManager


def list_available_reasoning_parsers() -> list[str]:
    """Return parser names exposed by the installed vLLM package."""
    return sorted(ReasoningParserManager.list_registered())


def reasoning_parser_exists(name: str) -> bool:
    return name in ReasoningParserManager.list_registered()


def resolve_effective_reasoning_parser(
    configured_reasoning_parser: str | None,
    *,
    backend_model: str,
    available_reasoning_parsers: list[str] | None = None,
) -> str | None:
    """Resolve the effective reasoning parser using explicit configuration."""
    available = set(available_reasoning_parsers or list_available_reasoning_parsers())

    if configured_reasoning_parser is None:
        return None

    if configured_reasoning_parser == "auto":
        raise ValueError(
            "The 'auto' reasoning parser resolution is no longer supported. "
            f"Please explicitly configure a specific reasoning parser for model "
            f"'{backend_model}'. Available parsers: {sorted(available)}"
        )

    if configured_reasoning_parser not in available:
        raise ValueError(
            f"Configured reasoning parser '{configured_reasoning_parser}' is not available. "
            f"Available parsers: {sorted(available)}"
        )

    return configured_reasoning_parser
