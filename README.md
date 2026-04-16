# pg-vector-ingester

FastAPI microservice that embeds conversation messages and persists dense vectors
into PostgreSQL via [pgvector](https://github.com/pgvector/pgvector).

Part of the `dialogue-agent` ecosystem. Works alongside
[qdrant-ingester](https://github.com/GoodchildTrevor/qdrant-ingester) — that service
indexes uploaded *files* into Qdrant; this service indexes *dialogue messages* into
Postgres so that similar past conversations can be found with a vector similarity search.

## Architecture

```
dialogue-agent
  │
  ├── POST /ingest  ──►  pg-vector-ingester /ingest
  │                        1. Load messages by file_id (embedding IS NULL)
  │                        2. Embed via fastembed (multilingual dense model)
  │                        3. UPDATE messages SET embedding = $vec
  │                        4. SET files.status = 'indexed'
  │
  └── POST /sync   ──►  pg-vector-ingester /sync
                           Find all messages WHERE embedding IS NULL → embed
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/ingest` | Embed messages for a file |
| POST | `/sync` | Incremental re-embed of all NULL embeddings |

### `POST /ingest`

```json
// Request  (matches PgIngesterClient.trigger_ingestion contract)
{
  "source_id": "<file_id UUID>",
  "options": {
    "force_reembed": false,   // optional — overwrite existing embeddings
    "batch_size": 32          // optional — override default batch size
  }
}

// Response
{
  "source_id": "<file_id UUID>",
  "messages_embedded": 42,
  "messages_skipped": 0
}
```

### `POST /sync`

```json
// Request
{ "file_id": "<UUID or null>" }

// Response
{ "re_embedded": 7, "skipped": 0 }
```

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

Service runs at `http://localhost:8001`. Swagger UI at `/docs`.

## Running without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
uvicorn pg_ingester.main:app --reload
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | — | `postgresql+asyncpg://...` |
| `DENSE_MODEL_NAME` | `paraphrase-multilingual-mpnet-base-v2` | fastembed model |
| `BATCH_SIZE` | `32` | Embedding batch size |
| `INSERT_BATCH_SIZE` | `64` | DB UPDATE batch size |
