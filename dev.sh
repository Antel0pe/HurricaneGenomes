#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1

# Start backend
UVICORN_CMD="uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"
if command -v uvicorn >/dev/null 2>&1; then
  $UVICORN_CMD &
else
  python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000 &
fi

# Start frontend
cd frontend && npm run dev:all

