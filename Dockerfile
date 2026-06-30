FROM python:3.11-slim

WORKDIR /app

# System deps for PDF parsing + FAISS
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download models at build time so cold start is instant
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-m3'); \
from sentence_transformers import CrossEncoder; \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" || true

RUN mkdir -p /app/storage /app/documents

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
