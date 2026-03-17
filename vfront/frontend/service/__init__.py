"""Frontend service domain.

Import service-level settings packages so the frontend domain root can register
all frontend settings via this single LCA package.
"""

import vfront.frontend.service.engine as _engine  # noqa: F401
import vfront.frontend.service.file_storage as _file_storage  # noqa: F401
import vfront.frontend.service.jobs as _jobs  # noqa: F401
import vfront.frontend.service.mcp as _mcp  # noqa: F401
import vfront.frontend.service.store as _store  # noqa: F401
