"""Shared config package.

Import shared settings modules at package load so bare GetSettings(...) calls
for shared config paths resolve once this LCA package is imported.
"""

from vfront.shared.config import adapters as _adapters  # noqa: F401
from vfront.shared.config import app as _app  # noqa: F401
from vfront.shared.config import managed_workers as _managed_workers  # noqa: F401
from vfront.shared.config import server as _server  # noqa: F401
