"""LoRA adapter management for local engine runtimes."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.lora.request import LoRARequest

from vfront.shared.engine.types import LoRAAdapterInfo

if TYPE_CHECKING:
    from vfront.engine.service.local import LocalEngineRuntimeState


GetRuntimeState = Callable[[str], "LocalEngineRuntimeState"]
GetPrimaryRuntimeModel = Callable[[], str]
GetPrimaryLoRARequests = Callable[[], dict[str, LoRARequest]]
SetPrimaryLoRARequests = Callable[[dict[str, LoRARequest]], None]


class LocalEngineAdapterManager:
    """Manage LoRA adapters across local runtime models."""

    def __init__(
        self,
        *,
        get_runtime_state: GetRuntimeState,
        get_primary_runtime_model: GetPrimaryRuntimeModel,
        get_primary_lora_requests: GetPrimaryLoRARequests,
        set_primary_lora_requests: SetPrimaryLoRARequests,
    ) -> None:
        self._get_runtime_state = get_runtime_state
        self._get_primary_runtime_model = get_primary_runtime_model
        self._get_primary_lora_requests = get_primary_lora_requests
        self._set_primary_lora_requests = set_primary_lora_requests

    def names(self) -> list[str]:
        return list(self._get_primary_lora_requests().keys())

    def resolve_lora(self, lora_name: str) -> LoRARequest:
        req = self._get_primary_lora_requests().get(lora_name)
        if req is None:
            raise ValueError(
                f"LoRA adapter '{lora_name}' not found. "
                f"Available: {list(self._get_primary_lora_requests().keys())}"
            )
        return req

    def resolve_lora_for_runtime(
        self,
        runtime: LocalEngineRuntimeState,
        lora_name: str | None,
    ) -> LoRARequest | None:
        if lora_name is None:
            return None
        req = runtime.lora_requests.get(lora_name)
        if req is None:
            raise ValueError(
                f"LoRA adapter '{lora_name}' not found for base model "
                f"'{runtime.runtime_model}'. Available: {list(runtime.lora_requests.keys())}"
            )
        return req

    def list_loras(self, runtime_model: str) -> list[LoRAAdapterInfo]:
        runtime = self._get_runtime_state(runtime_model)
        return [
            LoRAAdapterInfo(
                name=req.lora_name,
                path=req.lora_path,
                id=req.lora_int_id,
            )
            for req in runtime.lora_requests.values()
        ]

    async def load_lora(
        self,
        runtime_model: str,
        *,
        name: str,
        path: str,
    ) -> None:
        runtime = self._get_runtime_state(runtime_model)
        existing = runtime.lora_requests.get(name)
        if existing is not None:
            if existing.lora_path != path:
                raise ValueError(
                    f"LoRA adapter '{name}' is already loaded from '{existing.lora_path}'."
                )
            return

        lora_request = LoRARequest(
            lora_name=name,
            lora_int_id=self.next_lora_int_id(runtime),
            lora_path=path,
        )
        loaded = await runtime.engine.add_lora(lora_request)
        if not loaded:
            raise RuntimeError(f"Engine rejected LoRA adapter '{name}'.")

        runtime.lora_requests[name] = lora_request
        if runtime.runtime_model == self._get_primary_runtime_model():
            self._set_primary_lora_requests(runtime.lora_requests)

    async def unload_lora(self, runtime_model: str, *, name: str) -> None:
        runtime = self._get_runtime_state(runtime_model)
        req = runtime.lora_requests.get(name)
        if req is None:
            raise ValueError(
                f"LoRA adapter '{name}' not found for runtime model '{runtime.runtime_model}'."
            )

        removed = await runtime.engine.remove_lora(req.lora_int_id)
        if not removed:
            raise RuntimeError(f"Engine rejected LoRA adapter removal '{name}'.")

        runtime.lora_requests.pop(name, None)
        if runtime.runtime_model == self._get_primary_runtime_model():
            self._set_primary_lora_requests(runtime.lora_requests)

    @staticmethod
    def next_lora_int_id(runtime: LocalEngineRuntimeState) -> int:
        if not runtime.lora_requests:
            return 1
        return max(req.lora_int_id for req in runtime.lora_requests.values()) + 1
