"""ASGI entrypoint — used by Granian/uvicorn as `vfront.frontend.main:app`."""

from vfront.frontend.factory import create_application

app = create_application()
