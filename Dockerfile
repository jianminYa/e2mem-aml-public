FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy service code
COPY service /app/service

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    E2MEM_DB=/data/memory.sqlite3 \
    E2MEM_MODE=competition \
    E2MEM_HOST=0.0.0.0 \
    PORT=8000

RUN mkdir -p /data
VOLUME /data

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "service.api"]
