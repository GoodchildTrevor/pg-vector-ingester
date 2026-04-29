from __future__ import annotations

import asyncio
import logging
from typing import Sequence

from fastembed import TextEmbedding
from more_itertools import chunked

logger = logging.getLogger(__name__)


async def embed_texts(
    texts: Sequence[str],
    model: TextEmbedding,
    batch_size: int = 32,
) -> list[list[float]]:
    """Embed a list of texts using fastembed dense model.

    Runs synchronous fastembed in a thread pool to avoid blocking the event loop.
    Returns a list of float vectors in the same order as *texts*.
    """
    loop = asyncio.get_running_loop()

    def _embed_sync() -> list[list[float]]:
        results: list[list[float]] = []
        for batch in chunked(texts, batch_size):
            embeddings = list(model.embed(batch))
            results.extend(emb.tolist() for emb in embeddings)
        return results

    vectors = await loop.run_in_executor(None, _embed_sync)
    logger.info("Embedded %d texts -> %d vectors", len(texts), len(vectors))
    return vectors
