from __future__ import annotations

import logging
import uuid
from typing import Sequence

from more_itertools import chunked
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from pg_ingester.models import Message

logger = logging.getLogger(__name__)


async def fetch_messages_by_ids(
    session: AsyncSession,
    message_ids: Sequence[uuid.UUID],
    force_reembed: bool = False,
) -> list[Message]:
    """Return messages matching the given IDs.

    If force_reembed is False, only returns rows where embedding IS NULL.
    """
    stmt = select(Message).where(Message.id.in_(message_ids))
    if not force_reembed:
        stmt = stmt.where(Message.embedding.is_(None))
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def fetch_messages_without_embedding(
    session: AsyncSession,
    user_id: str | None = None,
) -> list[Message]:
    """Return ALL messages where embedding IS NULL, optionally filtered by user."""
    stmt = select(Message).where(Message.embedding.is_(None))
    if user_id is not None:
        stmt = stmt.where(Message.user_id == user_id)
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
    for batch in chunked(zip(message_ids, vectors), insert_batch_size):
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
