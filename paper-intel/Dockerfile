# ─────────────────────────────────────────────────────────────────────────────
# Sanshodhak — Production Dockerfile
#
# Build:  docker build -t sanshodhak:latest .
# Run:    docker run -p 8087:8087 \
#           -v $(pwd)/enhanced_rag_index:/app/enhanced_rag_index \
#           -e INDEX_DIR=/app/enhanced_rag_index \
#           sanshodhak:latest
#
# Oracle Free Tier compatible (1 OCPU / 1 GB RAM — use small embedding model)
# For low-RAM deployment, set embedding.model to "all-MiniLM-L6-v2" in config.yaml
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first for layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────────────────
COPY . .

# ── NLTK data ─────────────────────────────────────────────────────────────────
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)" || true

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd -m -u 1000 sanshodhak && chown -R sanshodhak:sanshodhak /app
USER sanshodhak

# ── Environment defaults ──────────────────────────────────────────────────────
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO \
    CONFIG_PATH=/app/config.yaml

EXPOSE 8087

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8087/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["python", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8087", \
     "--workers", "1", \
     "--log-level", "info"]
