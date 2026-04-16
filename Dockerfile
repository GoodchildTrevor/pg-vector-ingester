FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

COPY pg_ingester ./pg_ingester

CMD ["uvicorn", "pg_ingester.main:app", "--host", "0.0.0.0", "--port", "8000"]
