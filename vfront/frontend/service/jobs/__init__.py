"""Background job services.

Import settings at package load so bare GetSettings("frontend.jobs") resolves for job
service consumers.
"""

import vfront.frontend.service.engine as _engine_service  # noqa: F401
from vfront.frontend.service.jobs import settings as _settings  # noqa: F401
