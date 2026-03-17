"""Shared callable signature inspection helpers."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast


@dataclass(frozen=True)
class KeywordArgumentSupport:
    accepts: bool
    explicit: bool
    via_var_keyword: bool


def inspect_keyword_argument_support(
    callable_obj: object,
    name: str,
) -> KeywordArgumentSupport:
    """Describe whether a callable accepts a keyword argument."""
    if not callable(callable_obj):
        return KeywordArgumentSupport(
            accepts=False,
            explicit=False,
            via_var_keyword=False,
        )
    try:
        signature = inspect.signature(cast("Callable[..., object]", callable_obj))
    except (TypeError, ValueError):
        return KeywordArgumentSupport(
            accepts=False,
            explicit=False,
            via_var_keyword=False,
        )

    via_var_keyword = False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            via_var_keyword = True
            continue
        if parameter.name == name and parameter.kind != inspect.Parameter.POSITIONAL_ONLY:
            return KeywordArgumentSupport(
                accepts=True,
                explicit=True,
                via_var_keyword=via_var_keyword,
            )

    return KeywordArgumentSupport(
        accepts=via_var_keyword,
        explicit=False,
        via_var_keyword=via_var_keyword,
    )


def callable_accepts_keyword_argument(callable_obj: object, name: str) -> bool:
    """Return whether a callable accepts a keyword argument directly or via ``**kwargs``."""
    return inspect_keyword_argument_support(callable_obj, name).accepts
