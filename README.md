# pg-vector-ingester

FastAPI microservice that embeds conversation messages and persists dense vectors
into PostgreSQL via [pgvector](https://github.com/pgvector/pgvector).

Part of the [`dialogue-agent`](https://github.com/GoodchildTrevor/dialogue-agent) ecosystem.
Works alongside [`qdrant-ingester`](https://github.com/GoodchildTrevor/qdrant-ingester) —
that service indexes uploaded *files* into Qdrant; this service indexes *dialogue messages*
into PostgreSQL so that similar past conversations can be found via vector similarity search.

## Why this service exists

When the assistant resolves a user's problem after several iterations, the full conversation
is stored in the `messages` table. By embedding each message, the agent can later find
semantically similar past dialogues and jump straight to the solution — without repeating
the entire reasoning chain.

```sql
-- Find the 5 most relevant past messages for a new user query
SELECT content, metadata_json
FROM messages
ORDER BY embedding <=> $query_vector
LIMIT 5;
```

## Service ecosystem

| Service | Repo | Port | Purpose |
|---|---|---|---|
| `dialogue-agent` | [dialogue-agent](https://github.com/GoodchildTrevor/dialogue-agent) | `8000` | LangGraph orchestrator, SSE streaming, history search |
| `pg-vector-ingester` (this) | — | `8001` | Embeds dialogue messages, persists vectors to PostgreSQL |
| `qdrant-ingester` | [qdrant-ingester](https://github.com/GoodchildTrevor/qdrant-ingester) | `8002` | Chunks uploaded files, embeds and upserts into Qdrant |

### Repo layout (required for docker-compose build context)

```
projects/
├── dialogue-agent/
└── pg-vector-ingester/   ← this repo
```

## Architecture

```
dialogue-agent
  │
  ├── POST /ingest  ──►  pg-vector-ingester /ingest
  │                        1. Load messages by message_ids (WHERE embedding IS NULL)
  │                        2. Embed message.content via fastembed dense model
  │                        3. UPDATE messages SET embedding = $vec  (pgvector)
  │
  └── POST /sync   ──►  pg-vector-ingester /sync
                           Find ALL messages WHERE embedding IS NULL → embed
                           Optionally scoped to a single user_id
                           Use for recovery after partial failures or DB bootstrap
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/ingest` | Embed a specific list of messages by their IDs |
| POST | `/sync` | Incremental re-embed of all NULL embeddings |

### `POST /ingest`

Caller (dialogue-agent) passes the exact `message_ids` it wants embedded.
Status updates on `files` are **not** the responsibility of this service.

```json
// Request
{
  "message_ids": [
    "<message UUID>",
    "<message UUID>"
  ],
  "options": {
    "force_reembed": false,
    "batch_size": 32
  }
}

// Response
{
  "messages_embedded": 42,
  "messages_skipped": 0
}
```

### `POST /sync`

```json
// Request  (user_id null = global sync)
{ "user_id": "<string or null>" }

// Response
{ "re_embedded": 7, "skipped": 0 }
```

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

Service runs at `http://localhost:8001`. Swagger UI at `http://localhost:8001/docs`.

## Running without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
uvicorn pg_ingester.main:app --reload
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | — | `postgresql+asyncpg://user:pass@host/db` |
| `DENSE_MODEL_NAME` | `paraphrase-multilingual-mpnet-base-v2` | fastembed model (must match qdrant-ingester) |
| `BATCH_SIZE` | `32` | Messages per fastembed call |
| `INSERT_BATCH_SIZE` | `64` | DB UPDATE rows per transaction |

## Code structure

```
pg_ingester/
├── main.py          # FastAPI app: /health, /ingest, /sync
├── config.py        # pydantic-settings with lru_cache
├── db.py            # AsyncEngine + session factory, init_db()
├── models.py        # Minimal SQLAlchemy projection of dialogue-agent tables
├── schemas.py       # Pydantic request/response models
├── embedder.py      # fastembed in thread pool (run_in_executor)
└── loader.py        # fetch_messages_by_ids, save_embeddings
```
