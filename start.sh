#!/bin/bash
set -e

# Use PORT environment variable if set, otherwise default to 10000 (Render's default)
PORT=${PORT:-10000}

echo "Starting FastAPI server on port $PORT"

# Start uvicorn with the configured port
exec uvicorn main:app --host 0.0.0.0 --port "$PORT" --workers 1