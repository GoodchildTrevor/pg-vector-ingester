from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from pg_ingester.config import get_settings, get_dense_model
from pg_ingester.db import init_db, get_session
from pg_ingester.embedder import embed_texts
from pg_ingester.loader import (
    fetch_messages_for_file,
    fetch_messages_without_embedding,
    save_embeddings,
    set_file_status,
)
from pg_ingester.schemas import (
    IngestRequest, IngestResponse,
    SyncRequest, SyncResponse,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="pg-vector-ingester", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Embed all messages for a given file and persist vectors into PostgreSQL.

    Pipeline:
    1. Load messages for file_id (only NULL embeddings unless force_reembed)
    2. Embed message.content via fastembed dense model
    3. Bulk-UPDATE messages.embedding in batches
    4. Set files.status = 'indexed'
    """
    settings = get_settings()
    model = get_dense_model()
    file_id = request.file_id
    batch_size = request.batch_size_override or settings.batch_size

    # Step 1 — load messages
    messages = await fetch_messages_for_file(
        session, file_id, force_reembed=request.force_reembed
    )
    skipped_count = 0
    if not messages:
        logger.info("No messages to embed for file %s", file_id)
        if not request.force_reembed:
            # all messages already have embeddings
            skipped_count_stmt = await fetch_messages_for_file(session, file_id, force_reembed=True)
            skipped_count = len(skipped_count_stmt)
        return IngestResponse(
            source_id=file_id,
            messages_embedded=0,
            messages_skipped=skipped_count,
        )

    # Step 2 — embed
    texts = [m.content for m in messages]
    try:
        vectors = await embed_texts(texts, model, batch_size=batch_size)
    except Exception as exc:
        logger.error("Embedding failed for file %s: %s", file_id, exc)
        await set_file_status(session, file_id, "error", error_message=str(exc))
        raise HTTPException(status_code=500, detail=f"Embedding error: {exc}")

    # Step 3 — persist
    try:
        embedded = await save_embeddings(
            session,
            message_ids=[m.id for m in messages],
            vectors=vectors,
            insert_batch_size=settings.insert_batch_size,
        )
    except Exception as exc:
        logger.error("DB update failed for file %s: %s", file_id, exc)
        await set_file_status(session, file_id, "error", error_message=str(exc))
        raise HTTPException(status_code=500, detail=f"DB update error: {exc}")

    # Step 4 — mark file as indexed
    await set_file_status(session, file_id, "indexed")

    return IngestResponse(
        source_id=file_id,
        messages_embedded=embedded,
        messages_skipped=0,
    )


@app.post("/sync", response_model=SyncResponse)
async def sync(
    request: SyncRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Incremental sync: find all messages with embedding IS NULL and embed them.

    Scoped to a single file if file_id is provided, otherwise global.
    Use this to recover after partial failures or to bootstrap a fresh DB.
    """
    settings = get_settings()
    model = get_dense_model()

    messages = await fetch_messages_without_embedding(session, file_id=request.file_id)
    if not messages:
        return SyncResponse(re_embedded=0, skipped=0)

    texts = [m.content for m in messages]
    try:
        vectors = await embed_texts(texts, model, batch_size=settings.batch_size)
    except Exception as exc:
        logger.error("Sync embedding failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Embedding error: {exc}")

    re_embedded = await save_embeddings(
        session,
        message_ids=[m.id for m in messages],
        vectors=vectors,
        insert_batch_size=settings.insert_batch_size,
    )

    return SyncResponse(re_embedded=re_embedded, skipped=0)
