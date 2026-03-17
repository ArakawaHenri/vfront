"""Frontend domain root.

Import the frontend service LCA plus shared config so frontend consumers only
need to load this package root to register all frontend-facing settings paths.
"""

import vfront.frontend.service as _service  # noqa: F401
import vfront.shared as _shared  # noqa: F401
