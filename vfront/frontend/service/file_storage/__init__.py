"""Frontend file storage package.

Import settings at package load so bare GetSettings("frontend.file_storage") resolves
for package consumers.
"""

from vfront.frontend.service.file_storage import settings as _settings  # noqa: F401
