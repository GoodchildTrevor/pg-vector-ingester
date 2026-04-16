from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# /ingest
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    """Embed a specific list of messages by their IDs."""
    message_ids: list[uuid.UUID]             # IDs from dialogue-agent messages table
    options: dict[str, Any] | None = None    # optional overrides

    @property
    def force_reembed(self) -> bool:
        return bool((self.options or {}).get("force_reembed", False))

    @property
    def batch_size_override(self) -> int | None:
        v = (self.options or {}).get("batch_size")
        return int(v) if v is not None else None


class IngestResponse(BaseModel):
    messages_embedded: int
    messages_skipped: int


# ---------------------------------------------------------------------------
# /sync
# ---------------------------------------------------------------------------

class SyncRequest(BaseModel):
    """Re-embed all messages that are missing an embedding.

    If user_id is provided, scope the sync to that user's messages only.
    If None, perform a global sync across all messages.
    """
    user_id: str | None = None


class SyncResponse(BaseModel):
    re_embedded: int
    skipped: int
