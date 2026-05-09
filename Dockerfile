FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY proofrag ./proofrag
COPY configs ./configs
COPY examples ./examples

RUN pip install --no-cache-dir ".[api]"

EXPOSE 8000

CMD ["uvicorn", "proofrag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
