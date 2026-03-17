"""Engine-side adapter settings."""

from __future__ import annotations

from fastapiex.settings import Settings

from vfront.shared.config.adapters import AdapterSettings as _BaseAdapterSettings


@Settings("engine.adapters")
class AdapterSettings(_BaseAdapterSettings):
    pass
