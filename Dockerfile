# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim as builder

# Set environment variables for Python optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel for better caching
RUN pip install --upgrade pip setuptools wheel

# Copy requirements file first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies in virtual environment
# Split heavy dependencies for better caching
RUN pip install --no-cache-dir \
    fastapi==0.115.4 \
    uvicorn[standard]==0.24.0.post1 \
    pydantic==2.9.2 \
    python-dotenv==1.0.0 \
    requests>=2.31.0 \
    tqdm==4.66.1

# Install AI/ML dependencies (heavier packages)
RUN pip install --no-cache-dir \
    google-genai==1.38.0 \
    sentence-transformers>=3.0.0 \
    huggingface-hub>=0.20.0 \
    weaviate-client==4.10.4

# Pre-download the sentence transformer model to avoid runtime downloads
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Production stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/home/appuser/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface \
    HF_HUB_CACHE=/home/appuser/.cache/huggingface/hub

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy model cache from builder stage and set up cache directories
# Create non-root user for security with proper home directory
RUN groupadd -r appuser && useradd -r -g appuser -u 1001 -m -d /home/appuser appuser

# Copy model cache from builder stage and set up cache directories
COPY --from=builder /root/.cache /home/appuser/.cache
RUN mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /home/appuser/.cache

# Set working directory
WORKDIR /app

# Copy application files
COPY --chown=appuser:appuser . .

# Make startup script executable
RUN chmod +x start.sh

# Switch to non-root user
USER appuser

# Expose port that Render expects (default 10000)
EXPOSE 10000

# Health check - use default port 10000 for health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:10000/health || exit 1

# Use startup script to handle PORT environment variable properly
CMD ["./start.sh"]
