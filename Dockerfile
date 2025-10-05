# ---- Builder Stage ----
FROM python:3.11-slim as builder

# System settings
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install only essential build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip first
RUN pip install --upgrade pip setuptools wheel

# Copy requirements early for caching
COPY requirements.txt .

# Install dependencies in two groups (light + heavy)
RUN pip install \
    fastapi==0.115.4 \
    uvicorn[standard]==0.24.0.post1 \
    pydantic==2.9.2 \
    python-dotenv==1.0.0 \
    requests>=2.31.0 \
    tqdm==4.66.1

RUN pip install \
    google-genai==1.38.0 \
    sentence-transformers>=3.0.0 \
    huggingface-hub>=0.20.0 \
    weaviate-client==4.10.4

# Pre-download lightweight embedding model (small size!)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ---- Runtime Stage ----
FROM python:3.11-slim

# Runtime envs
ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/home/appuser/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface \
    HF_HUB_CACHE=/home/appuser/.cache/huggingface/hub

# Only curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd -r appuser && useradd -r -g appuser -u 1001 -m -d /home/appuser appuser

# Copy venv + model cache
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.cache /home/appuser/.cache
RUN chown -R appuser:appuser /home/appuser/.cache

# App setup
WORKDIR /app
COPY --chown=appuser:appuser . .

# Permissions
RUN chmod +x start.sh

USER appuser

# Railway exposes $PORT (default 8080)
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

CMD ["./start.sh"]
