# ---- Builder Stage ----
FROM python:3.11-slim as builder

# ---- Environment ----
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- Install system deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---- Virtual environment ----
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ---- Upgrade pip & install requirements ----
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# ---- Pre-download sentence-transformers model ----
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ---- Runtime Stage ----
FROM python:3.11-slim

# ---- Environment ----
ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/venv/.cache/huggingface \
    TRANSFORMERS_CACHE=/opt/venv/.cache/huggingface \
    HF_HUB_CACHE=/opt/venv/.cache/huggingface/hub

# ---- Only curl for healthcheck ----
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Non-root user ----
RUN groupadd -r appuser && useradd -r -g appuser -u 1001 -m -d /home/appuser appuser

# ---- Copy venv from builder ----
COPY --from=builder /opt/venv /opt/venv

# ---- App code ----
WORKDIR /app
COPY --chown=appuser:appuser . .

# ---- Permissions ----
RUN chmod +x start.sh
USER appuser

# ---- Railway port ----
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

CMD ["./start.sh"]
