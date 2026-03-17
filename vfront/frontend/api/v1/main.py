"""v1 API router — aggregates all /v1/* endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from vfront.frontend.api.v1.batches import router as batches_router
from vfront.frontend.api.v1.chat import router as chat_router
from vfront.frontend.api.v1.completions import router as completions_router
from vfront.frontend.api.v1.embeddings import router as embeddings_router
from vfront.frontend.api.v1.files import router as files_router
from vfront.frontend.api.v1.models import router as models_router
from vfront.frontend.api.v1.responses import router as responses_router

router = APIRouter(prefix="/v1")
router.include_router(chat_router, tags=["Chat Completions"])
router.include_router(completions_router, tags=["Completions"])
router.include_router(embeddings_router, tags=["Embeddings"])
router.include_router(models_router, tags=["Models"])
router.include_router(responses_router, tags=["Responses"])
router.include_router(files_router, tags=["Files"])
router.include_router(batches_router, tags=["Batches"])
