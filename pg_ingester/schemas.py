from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# /ingest
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    """Mirrors PgIngesterClient.trigger_ingestion — source_id is file_id."""
    source_id: uuid.UUID                    # file_id from dialogue-agent files table
    options: dict[str, Any] | None = None   # optional overrides

    @property
    def file_id(self) -> uuid.UUID:
        return self.source_id

    @property
    def force_reembed(self) -> bool:
        return bool((self.options or {}).get("force_reembed", False))

    @property
    def batch_size_override(self) -> int | None:
        v = (self.options or {}).get("batch_size")
        return int(v) if v is not None else None


class IngestResponse(BaseModel):
    source_id: uuid.UUID
    messages_embedded: int
    messages_skipped: int


# ---------------------------------------------------------------------------
# /sync
# ---------------------------------------------------------------------------

class SyncRequest(BaseModel):
    """Re-embed all messages that are missing an embedding.

    If file_id is provided, scope the sync to that file's messages only.
    If None, perform a global sync across all messages.
    """
    file_id: uuid.UUID | None = None


class SyncResponse(BaseModel):
    re_embedded: int
    skipped: int
