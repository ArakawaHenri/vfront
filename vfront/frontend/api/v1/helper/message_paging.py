"""Cursor helpers for stored chat completion message pagination."""

from __future__ import annotations

from vfront.frontend.middleware.exceptions import InvalidRequestError

_MESSAGE_CURSOR_PREFIX = "msg_"


def _invalid_after_cursor() -> InvalidRequestError:
    return InvalidRequestError(
        "Invalid 'after' cursor for stored chat completion messages.",
        param="after",
    )


def encode_message_cursor(completion_id: str, index: int) -> str:
    return f"{completion_id}-{index}"


def decode_message_cursor(completion_id: str, cursor: str) -> int:
    if cursor.startswith(_MESSAGE_CURSOR_PREFIX):
        suffix = cursor[len(_MESSAGE_CURSOR_PREFIX) :]
    else:
        prefix = f"{completion_id}-"
        if not cursor.startswith(prefix):
            raise _invalid_after_cursor()
        suffix = cursor[len(prefix) :]

    try:
        index = int(suffix)
    except ValueError as exc:
        raise _invalid_after_cursor() from exc
    if index < 0:
        raise _invalid_after_cursor()
    return index


def page_message_items(messages: list[dict], *, completion_id: str, start_index: int) -> list[dict]:
    items: list[dict] = []
    for index, message in enumerate(messages, start=start_index):
        item = dict(message)
        item.setdefault("id", encode_message_cursor(completion_id, index))
        items.append(item)
    return items
