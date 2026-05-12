FROM python:3.12-slim

# System deps for aiortc (WebRTC) + audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libopus-dev \
    libvpx-dev \
    libsrtp2-dev \
    libssl-dev \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (cached layer — only rebuilds when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY bot.py server.py ./
COPY static/ ./static/

EXPOSE 8000

ENV HOST=0.0.0.0
ENV PORT=8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "server.py"]
