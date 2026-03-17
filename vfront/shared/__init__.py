"""Shared domain root.

Import shared config and shared engine LCAs so one stable root import registers
all cross-domain settings, including engine-worker settings.
"""

import vfront.shared.config as _config  # noqa: F401
import vfront.shared.engine as _engine  # noqa: F401
