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

from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from bot import run_bot

load_dotenv()

request_handler = SmallWebRTCRequestHandler()

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


@app.get("/voices")
async def get_voices():
    def _ev(key): return os.environ.get(key, "")
    return JSONResponse({
        "elevenlabs": [
            {"id": _ev("ELEVENLABS_VOICE_TA"), "label": "Default"},
            {"id": _ev("ELEVENLABS_VOICE_2"),  "label": "Voice 2"},
            {"id": _ev("ELEVENLABS_VOICE_3"),  "label": "Voice 3"},
        ],
        "sarvam": [
            {"id": "simran", "label": "Simran"},
            {"id": "priya",  "label": "Priya"},
            {"id": "kavya",  "label": "Kavya"},
            {"id": "neha",   "label": "Neha"},
        ],
        "cartesia": [
            {"id": _ev("CARTESIA_VOICE_ID"), "label": "Kavitha"},
            {"id": _ev("CARTESIA_VOICE_2"),  "label": "Voice 2"},
            {"id": _ev("CARTESIA_VOICE_3"),  "label": "Voice 3"},
        ],
        "rime": [
            {"id": "indira",  "label": "Indira"},
            {"id": "zara",    "label": "Zara"},
            {"id": "pita",    "label": "Pita"},
            {"id": "meadow",  "label": "Meadow"},
        ],
    })


@app.post("/offer")
async def offer(request: Request):
    body = await request.json()
    llm_provider = body.pop("llm", "groq")
    tts_provider = body.pop("tts", "elevenlabs")
    stt_provider = body.pop("stt", "sarvam")
    voice_id     = body.pop("voice", None)

    rtc_request = SmallWebRTCRequest.from_dict(body)

    async def bot_callback(connection):
        asyncio.create_task(run_bot(connection, llm_provider=llm_provider, tts_provider=tts_provider, stt_provider=stt_provider, voice_id=voice_id, transcript=transcript))

    answer = await request_handler.handle_web_request(rtc_request, bot_callback)
    return answer


@app.patch("/ice")
async def ice(request: Request):
    body = await request.json()
    patch_request = SmallWebRTCPatchRequest(**body)
    await request_handler.handle_patch_request(patch_request)
    return {"status": "ok"}


@app.get("/transcript")
async def get_transcript():
    return JSONResponse(list(transcript))


@app.get("/logs")
async def get_logs():
    return JSONResponse(list(server_logs))

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Blissy Restaurant Bot"}


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
        log_level="info",
    )
