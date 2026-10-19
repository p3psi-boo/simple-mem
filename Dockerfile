# syntax=docker/dockerfile:1
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SIMPLE_MEM_HOST=0.0.0.0 \
    SIMPLE_MEM_PORT=8765 \
    SIMPLE_MEM_DB_PATH=/data/simple-mem.db \
    SIMPLE_MEM_EXTRACT_INTERVAL=60 \
    SIMPLE_MEM_LLM_BASE_URL="" \
    SIMPLE_MEM_LLM_API_KEY="" \
    SIMPLE_MEM_LLM_MODEL="gpt-4o-mini" \
    SIMPLE_MEM_EMBED_BASE_URL="" \
    SIMPLE_MEM_EMBED_API_KEY="" \
    SIMPLE_MEM_EMBED_MODEL="text-embedding-3-small"

WORKDIR /app

# Create non-root user and persistent SQLite data directory
RUN groupadd --gid 10001 simplemem \
    && useradd --uid 10001 --gid simplemem --create-home --shell /usr/sbin/nologin simplemem \
    && mkdir -p /data \
    && chown -R simplemem:simplemem /data /app

# Install Hub dependencies and package
COPY pyproject.toml README.md LICENSE ./
COPY simple_mem ./simple_mem
RUN pip install --no-cache-dir .

USER simplemem
VOLUME ["/data"]
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import os, urllib.request; port = os.environ.get('SIMPLE_MEM_PORT', '8765'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3)" || exit 1

ENTRYPOINT ["simple-mem-hub"]
CMD ["serve"]
