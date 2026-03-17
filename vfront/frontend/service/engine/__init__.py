"""Frontend engine service package.

Import settings at package load so bare GetSettings("frontend.routing") remains
available to sibling consumers without helper wrappers.
"""

from vfront.frontend.service.engine import settings as _settings  # noqa: F401
