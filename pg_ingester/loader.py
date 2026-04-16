from __future__ import annotations

import logging
import uuid
from typing import Sequence

from more_itertools import chunked
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from pg_ingester.models import File, Message

logger = logging.getLogger(__name__)


async def fetch_messages_for_file(
    session: AsyncSession,
    file_id: uuid.UUID,
    force_reembed: bool = False,
) -> list[Message]:
    """Return messages belonging to file_id.

    If force_reembed is False, only returns rows where embedding IS NULL.
    """
    stmt = select(Message).where(Message.file_id == file_id)
    if not force_reembed:
        stmt = stmt.where(Message.embedding.is_(None))
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def fetch_messages_without_embedding(
    session: AsyncSession,
    file_id: uuid.UUID | None = None,
) -> list[Message]:
    """Return ALL messages where embedding IS NULL, optionally filtered by file."""
    stmt = select(Message).where(Message.embedding.is_(None))
    if file_id is not None:
        stmt = stmt.where(Message.file_id == file_id)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def save_embeddings(
    session: AsyncSession,
    message_ids: Sequence[uuid.UUID],
    vectors: Sequence[list[float]],
    insert_batch_size: int = 64,
) -> int:
    """Bulk-update messages.embedding in batches. Returns number of updated rows."""
    updated = 0
    pairs = list(zip(message_ids, vectors))
    for batch in chunked(pairs, insert_batch_size):
        for msg_id, vec in batch:
            await session.execute(
                update(Message)
                .where(Message.id == msg_id)
                .values(embedding=vec)
            )
        await session.commit()
        updated += len(batch)
        logger.debug("Saved embedding batch of %d", len(batch))
    logger.info("Updated embeddings for %d messages", updated)
    return updated


async def set_file_status(
    session: AsyncSession,
    file_id: uuid.UUID,
    status: str,
    error_message: str | None = None,
) -> None:
    """Update files.status (and optionally files.error_message)."""
    values: dict = {"status": status}
    if error_message is not None:
        values["error_message"] = error_message
    await session.execute(
        update(File).where(File.id == file_id).values(**values)
    )
    await session.commit()
    logger.info("File %s status -> %s", file_id, status)
