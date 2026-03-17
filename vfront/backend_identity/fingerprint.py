"""Serving-configuration fingerprint for ``system_fingerprint``.

The fingerprint reflects only the resolved serving identity:
- runtime model weights/configuration
- optional resolved LoRA adapter identity

It must not vary merely because a different public model alias pointed to the
same backend.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from importlib.metadata import PackageNotFoundError, version

from fastapiex.settings import GetSettings

from vfront.shared.engine.types import EngineHealth


@dataclass(frozen=True)
class FingerprintInputs:
    """All serving-configuration fields that affect output determinism."""

    runtime_model: str
    weights: str
    tokenizer_id: str | None
    quantization: str | None
    dtype: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    engine_version: str | None
    effective_reasoning_parser: str | None = None
    configured_reasoning_effort_map: (
        dict[str, dict[str, bool | str | int | float | None] | None] | None
    ) = None
    # Extra parameters that vary per request, e.g., LoRA adapter name
    extra: dict[str, str | int | float | bool | None] | None = None


def compute_fingerprint(inputs: FingerprintInputs) -> str | None:
    """Return a stable fingerprint string, or *None* if inputs are insufficient.

    The hash is the first 16 hex characters of the SHA-256 digest of the
    JSON-serialised inputs (sorted keys, no whitespace).
    """
    if not inputs.runtime_model or not inputs.weights:
        return None

    payload = json.dumps(
        asdict(inputs),
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"fp-{digest[:16]}"


def inputs_from_health(health: EngineHealth) -> FingerprintInputs | None:
    """Build fingerprint inputs from a concrete backend health snapshot."""
    if (
        not health.weights
        or not health.dtype
        or health.tensor_parallel_size is None
        or health.pipeline_parallel_size is None
    ):
        return None

    return FingerprintInputs(
        runtime_model=health.runtime_model,
        weights=health.weights,
        tokenizer_id=health.tokenizer_id,
        quantization=health.quantization,
        dtype=health.dtype,
        tensor_parallel_size=health.tensor_parallel_size,
        pipeline_parallel_size=health.pipeline_parallel_size,
        engine_version=health.engine_version,
        effective_reasoning_parser=health.effective_reasoning_parser,
        configured_reasoning_effort_map=health.configured_reasoning_effort_map,
    )


def fingerprint_from_health(
    health: EngineHealth,
    *,
    lora_name: str | None = None,
) -> str | None:
    """Return a fingerprint for the resolved backend described by ``health``."""
    inputs = inputs_from_health(health)
    if inputs is None:
        return None
    if lora_name is None:
        return compute_fingerprint(inputs)
    return compute_fingerprint(replace(inputs, extra={"lora_name": lora_name}))


class FingerprintProvider:
    """Lazily computes and caches base serving identity inputs."""

    def __init__(self) -> None:
        self._base_inputs: FingerprintInputs | None = None

    def get_base_inputs(self) -> FingerprintInputs:
        if self._base_inputs is None:
            self._base_inputs = self._build_base()
        return self._base_inputs

    def get(self, *, lora_name: str | None = None) -> str | None:
        """Compute fingerprint for the base runtime plus resolved adapter identity."""
        base = self.get_base_inputs()
        if lora_name is None:
            return compute_fingerprint(base)
        return compute_fingerprint(replace(base, extra={"lora_name": lora_name}))

    def invalidate(self) -> None:
        self._base_inputs = None

    def _build_base(self) -> FingerprintInputs:
        settings = GetSettings("engine.runtime")
        primary_runtime = settings.primary_runtime_model
        try:
            engine_version = version("vllm")
        except PackageNotFoundError:
            engine_version = None

        return FingerprintInputs(
            runtime_model=primary_runtime.id,
            weights=primary_runtime.weights,
            tokenizer_id=primary_runtime.tokenizer,
            quantization=primary_runtime.quantization,
            dtype=primary_runtime.dtype,
            tensor_parallel_size=primary_runtime.tensor_parallel_size,
            pipeline_parallel_size=primary_runtime.pipeline_parallel_size,
            engine_version=engine_version,
            effective_reasoning_parser=primary_runtime.reasoning_parser,
        )


_provider = FingerprintProvider()


def get_fingerprint(*, lora_name: str | None = None) -> str | None:
    """Return the serving-identity fingerprint for the active runtime."""
    return _provider.get(lora_name=lora_name)


def invalidate_fingerprint_cache() -> None:
    """Invalidate the cached fingerprint (call after config changes, e.g. in tests)."""
    _provider.invalidate()
