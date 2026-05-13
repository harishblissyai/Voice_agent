import asyncio
import os
import sys
from collections import deque
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

# ── Capture server logs into a ring buffer for the frontend ──────────────────
server_logs: deque = deque(maxlen=200)

def _log_sink(message):
    record = message.record
    level = record["level"].name
    text  = record["message"]
    name  = record["name"]
    server_logs.append({"level": level, "text": f"[{name}] {text}"})

logger.add(_log_sink, level="DEBUG", format="{message}")

from aiortc.rtcconfiguration import RTCIceServer
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from bot import run_bot

load_dotenv()

# STUN servers needed for WebRTC behind NAT (e.g. EC2)
_ice_servers = [
    RTCIceServer(urls="stun:stun.l.google.com:19302"),
    RTCIceServer(urls="stun:stun1.l.google.com:19302"),
]
request_handler = SmallWebRTCRequestHandler(ice_servers=_ice_servers)

# In-memory transcript for testing UI (last 50 lines)
transcript: deque = deque(maxlen=50)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Blissy Restaurant bot server starting")
    yield
    await request_handler.close()
    logger.info("Blissy Restaurant bot server stopped")


app = FastAPI(title="Blissy Restaurant Bot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


@app.get("/demo")
async def serve_demo():
    return FileResponse("static/demo.html")


# ── API routes — all under /api/* for CloudFront behavior routing ─────────────

@app.post("/api/offer")
@app.post("/offer")
async def offer(request: Request):
    body = await request.json()
    llm_provider = body.pop("llm", "anthropic")
    tts_provider = body.pop("tts", "elevenlabs")
    stt_provider = body.pop("stt", "sarvam")
    voice_id     = body.pop("voice", None)
    expressive   = bool(body.pop("expressive", False))

    rtc_request = SmallWebRTCRequest.from_dict(body)

    async def bot_callback(connection):
        asyncio.create_task(run_bot(connection, llm_provider=llm_provider, tts_provider=tts_provider, stt_provider=stt_provider, voice_id=voice_id, expressive=expressive, transcript=transcript))

    answer = await request_handler.handle_web_request(rtc_request, bot_callback)
    return answer


@app.patch("/api/ice")
@app.patch("/ice")
async def ice(request: Request):
    body = await request.json()
    patch_request = SmallWebRTCPatchRequest(**body)
    await request_handler.handle_patch_request(patch_request)
    return {"status": "ok"}


@app.get("/api/transcript")
@app.get("/transcript")
async def get_transcript():
    return JSONResponse(list(transcript))


@app.get("/api/logs")
@app.get("/logs")
async def get_logs():
    return JSONResponse(list(server_logs))


@app.get("/api/health")
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Blissy Restaurant Bot"}


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8070)),
        reload=False,
        log_level="info",
    )
