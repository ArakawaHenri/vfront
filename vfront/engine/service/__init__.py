"""Engine service package.

Import settings modules at package load so bare GetSettings(...) calls in
engine-service consumers resolve against registered paths.
"""

from vfront.engine.service import adapters as _adapters  # noqa: F401
from vfront.engine.service import api_settings as _api_settings  # noqa: F401
from vfront.engine.service import settings as _settings  # noqa: F401
