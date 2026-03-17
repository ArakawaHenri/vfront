"""Engine domain root.

Import the engine service package plus shared config so engine consumers only
need this package root to register engine and shared settings paths.
"""

import vfront.engine.service as _service  # noqa: F401
import vfront.shared as _shared  # noqa: F401
