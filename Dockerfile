FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HELIX_PROVIDER_MODE=auto

# Install locked dependencies first for better layer caching.
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000

# `helix serve` launches uvicorn on 0.0.0.0:8000.
CMD ["python", "-m", "helix.cli.main", "serve"]
