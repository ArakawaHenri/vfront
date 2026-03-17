"""Frontend store service package.

Import settings at package load so bare GetSettings("frontend.store") resolves
wherever StoreService package modules are imported.
"""

from vfront.frontend.service.store import settings as _settings  # noqa: F401
