FROM python:3.12-slim

# System deps for aiortc (WebRTC) + av (audio processing) + opencv (headless)
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

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENV HOST=0.0.0.0
ENV PORT=8000

CMD ["python", "server.py"]
