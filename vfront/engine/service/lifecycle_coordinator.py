"""Lifecycle orchestration for local vLLM engine runtimes."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from fastapiex.settings import GetSettings

from vfront.shared.engine.managed import select_runtime_models

if TYPE_CHECKING:
    from vllm import AsyncLLMEngine

    from vfront.engine.service.local import LocalEngineRuntimeState

logger = logging.getLogger(__name__)

GetRuntimes = Callable[[], dict[str, "LocalEngineRuntimeState"]]
GetPrimaryEngine = Callable[[], "AsyncLLMEngine | None"]
CreateRuntimeState = Callable[..., "LocalEngineRuntimeState"]
SyncPrimaryRuntime = Callable[["LocalEngineRuntimeState"], None]
ResetPrimaryState = Callable[[], None]


def _log_lifecycle_event(
    level: int,
    message: str,
    *args: object,
    event: str,
    **fields: object,
) -> None:
    logger.log(level, message, *args, extra={"event": event, **fields})


def _shutdown_engine_instance(engine: object) -> None:
    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        shutdown()
        return

    shutdown_background_loop = getattr(engine, "shutdown_background_loop", None)
    if callable(shutdown_background_loop):
        shutdown_background_loop()
        return

    logger.warning("vLLM engine instance does not expose a known shutdown API: %r", engine)


def _cleanup_distributed_runtime_state() -> None:
    """Best-effort cleanup for distributed state that may outlive engine.shutdown()."""
    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()
        return
    except Exception:
        logger.debug("vLLM distributed cleanup helper unavailable; falling back to torch.", exc_info=True)

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        logger.debug("Torch distributed cleanup fallback failed.", exc_info=True)


class LocalEngineLifecycleCoordinator:
    """Coordinate local engine startup and shutdown while leaving state ownership to the manager."""

    def __init__(
        self,
        *,
        get_runtimes: GetRuntimes,
        get_primary_engine: GetPrimaryEngine,
        create_runtime_state: CreateRuntimeState,
        sync_primary_runtime: SyncPrimaryRuntime,
        reset_primary_state: ResetPrimaryState,
    ) -> None:
        self._get_runtimes = get_runtimes
        self._get_primary_engine = get_primary_engine
        self._create_runtime_state = create_runtime_state
        self._sync_primary_runtime = sync_primary_runtime
        self._reset_primary_state = reset_primary_state

    async def start(self) -> None:
        try:
            settings = GetSettings("engine.runtime")
        except Exception:
            self._get_runtimes().clear()
            self._reset_primary_state()
            _log_lifecycle_event(
                logging.INFO,
                "No runtime settings are registered in this process; skipping local vLLM startup.",
                event="engine.lifecycle.start.skipped",
            )
            return

        worker_runtime_model = GetSettings("engine_worker").runtime_model
        adapter_settings = GetSettings("engine.adapters")
        selected_runtime_models = select_runtime_models(settings.models, worker_runtime_model)
        _log_lifecycle_event(
            logging.INFO,
            "Starting local vLLM engine runtimes.",
            event="engine.lifecycle.start.begin",
            runtime_count=len(selected_runtime_models),
            worker_runtime_model=worker_runtime_model,
        )

        runtimes = self._get_runtimes()
        runtimes.clear()
        for runtime_model in selected_runtime_models:
            runtime = self._create_runtime_state(
                runtime_model=runtime_model,
                lora_settings=adapter_settings,
                default_fim_policy=settings.fim,
            )
            runtimes[runtime_model.id] = runtime
            _log_lifecycle_event(
                logging.INFO,
                "vLLM engine initialized with runtime '%s' (weights=%s)",
                runtime_model.id,
                runtime_model.weights,
                event="engine.lifecycle.runtime.initialized",
                runtime_model=runtime_model.id,
                weights=runtime_model.weights,
            )

        self._sync_primary_runtime(runtimes[selected_runtime_models[0].id])
        _log_lifecycle_event(
            logging.INFO,
            "Local vLLM engine startup complete.",
            event="engine.lifecycle.start.complete",
            runtime_count=len(selected_runtime_models),
            primary_runtime_model=selected_runtime_models[0].id,
            worker_runtime_model=worker_runtime_model,
        )

    async def shutdown(self) -> None:
        runtimes = self._get_runtimes()
        primary_engine = self._get_primary_engine()
        had_runtime_entries = bool(runtimes)
        had_loaded_engines = bool(runtimes) or primary_engine is not None
        shutdown_errors: list[Exception] = []
        _log_lifecycle_event(
            logging.INFO,
            "Shutting down local vLLM engine runtimes.",
            event="engine.lifecycle.shutdown.begin",
            runtime_count=len(runtimes),
            had_primary_engine=primary_engine is not None,
        )

        try:
            if runtimes:
                loaded_runtimes = list(runtimes.values())
                runtimes.clear()
                for runtime in loaded_runtimes:
                    try:
                        _shutdown_engine_instance(runtime.engine)
                        _log_lifecycle_event(
                            logging.INFO,
                            "vLLM runtime '%s' shut down.",
                            runtime.runtime_model,
                            event="engine.lifecycle.runtime.shutdown",
                            runtime_model=runtime.runtime_model,
                            weights=runtime.weights,
                        )
                    except Exception as exc:
                        shutdown_errors.append(exc)
                        logger.exception(
                            "Failed to shut down vLLM runtime '%s'.",
                            runtime.runtime_model,
                            extra={
                                "event": "engine.lifecycle.runtime.shutdown_failed",
                                "runtime_model": runtime.runtime_model,
                                "weights": runtime.weights,
                            },
                        )
                _log_lifecycle_event(
                    logging.INFO,
                    "All vLLM runtimes shut down.",
                    event="engine.lifecycle.shutdown.complete",
                    runtime_count=len(loaded_runtimes),
                    had_primary_engine=primary_engine is not None,
                    shutdown_error_count=len(shutdown_errors),
                )
            elif primary_engine is not None:
                try:
                    _shutdown_engine_instance(primary_engine)
                    _log_lifecycle_event(
                        logging.INFO,
                        "Primary vLLM engine shut down.",
                        event="engine.lifecycle.shutdown.primary_complete",
                    )
                except Exception as exc:
                    shutdown_errors.append(exc)
                    logger.exception(
                        "Failed to shut down primary vLLM engine.",
                        extra={"event": "engine.lifecycle.shutdown.primary_failed"},
                    )
        finally:
            if had_loaded_engines:
                _cleanup_distributed_runtime_state()
            self._reset_primary_state()

        if not had_runtime_entries and primary_engine is not None:
            _log_lifecycle_event(
                logging.INFO,
                "Local vLLM engine shutdown complete.",
                event="engine.lifecycle.shutdown.complete",
                runtime_count=0,
                had_primary_engine=True,
                shutdown_error_count=len(shutdown_errors),
            )

        if shutdown_errors:
            if len(shutdown_errors) == 1:
                raise shutdown_errors[0]
            raise ExceptionGroup(
                "Failed to shut down one or more local vLLM engines.",
                shutdown_errors,
            )
