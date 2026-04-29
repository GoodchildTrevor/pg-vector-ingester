from __future__ import annotations

import logging
import uuid
from typing import Sequence

from more_itertools import chunked
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from pg_ingester.models import Message

logger = logging.getLogger(__name__)

# Maximum rows loaded into RAM per sync page to prevent OOM on large tables
SYNC_PAGE_SIZE = 500


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
    limit: int = SYNC_PAGE_SIZE,
    offset: int = 0,
) -> list[Message]:
    """Return messages where embedding IS NULL, paginated to avoid OOM.

    Defaults to SYNC_PAGE_SIZE rows per call. Pass offset to page through results.
    Optionally scoped to a single user via user_id.
    """
    stmt = (
        select(Message)
        .where(Message.embedding.is_(None))
        .order_by(Message.id)
        .limit(limit)
        .offset(offset)
    )
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
    """Bulk-update messages.embedding using a single UPDATE ... FROM (VALUES ...) per batch.

    Replaces N individual UPDATE statements with one statement per batch,
    dramatically reducing round-trips on large payloads.
    Returns number of updated rows.
    """
    updated = 0
    for batch in chunked(zip(message_ids, vectors), insert_batch_size):
        batch = list(batch)
        # Build parameterised VALUES list: ($1::uuid, $2::vector), ($3::uuid, $4::vector), ...
        placeholders = ", ".join(
            f"(:id_{i}::uuid, :vec_{i}::vector)"
            for i in range(len(batch))
        )
        params: dict = {}
        for i, (msg_id, vec) in enumerate(batch):
            params[f"id_{i}"] = str(msg_id)
            params[f"vec_{i}"] = "[" + ",".join(str(v) for v in vec) + "]"

        stmt = text(
            f"""
            UPDATE messages
            SET embedding = v.embedding
            FROM (VALUES {placeholders}) AS v(id, embedding)
            WHERE messages.id = v.id
            """
        )
        await session.execute(stmt, params)
        await session.commit()
        updated += len(batch)
        logger.debug("Saved embedding batch of %d", len(batch))

    logger.info("Updated embeddings for %d messages", updated)
    return updated
