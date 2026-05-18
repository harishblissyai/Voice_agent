import asyncio
import json
import os
import sys
from collections import deque
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response as PlainResponse
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

from bot import run_bot, run_bot_twilio, run_bot_plivo

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


# ── Twilio phone-call routes ───────────────────────────────────────────────────

@app.post("/api/twilio/incoming")
@app.post("/twilio/incoming")
async def twilio_incoming(request: Request):
    # TWILIO_STREAM_HOST overrides auto-detection — set to EC2 public IP/domain
    # to bypass CloudFront (which may not forward WebSocket connections)
    stream_host = os.environ.get("TWILIO_STREAM_HOST") or request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    ws_url = f"wss://{stream_host}/api/twilio/stream"
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Connect><Stream url="{ws_url}"/></Connect>'
        "</Response>"
    )
    return PlainResponse(content=twiml, media_type="text/xml")


@app.websocket("/api/twilio/stream")
@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    await websocket.accept()
    stream_sid = None
    call_sid = None
    async for raw in websocket.iter_text():
        msg = json.loads(raw)
        if msg.get("event") == "start":
            stream_sid = msg["start"]["streamSid"]
            call_sid   = msg["start"]["callSid"]
            break
        if msg.get("event") not in ("connected",):
            break
    if not stream_sid:
        await websocket.close()
        return
    await run_bot_twilio(websocket, stream_sid, call_sid, transcript=transcript)


# ── Plivo phone-call routes ────────────────────────────────────────────────────

@app.post("/api/plivo/incoming")
@app.post("/plivo/incoming")
async def plivo_incoming(request: Request):
    stream_host = os.environ.get("PLIVO_STREAM_HOST") or request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    ws_url = f"wss://{stream_host}/api/plivo/stream"
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Stream keepCallAlive="true" contentType="audio/x-mulaw;rate=8000" bidirectional="true">{ws_url}</Stream>'
        "</Response>"
    )
    return PlainResponse(content=xml, media_type="text/xml")


@app.websocket("/api/plivo/stream")
@app.websocket("/plivo/stream")
async def plivo_stream(websocket: WebSocket):
    await websocket.accept()
    stream_id = None
    call_id = None
    async for raw in websocket.iter_text():
        msg = json.loads(raw)
        if msg.get("event") == "start":
            start = msg.get("start", {})
            stream_id = start.get("streamId") or msg.get("streamId")
            call_id   = start.get("callId")   or msg.get("callId")
            break
        if msg.get("event") not in ("connected",):
            break
    if not stream_id:
        await websocket.close()
        return
    await run_bot_plivo(websocket, stream_id, call_id, transcript=transcript)


# ── Plivo call-me-back (outbound call trigger) ────────────────────────────────

@app.get("/callme")
async def serve_callme():
    return FileResponse("static/callme.html")


@app.post("/api/call-me")
async def call_me(request: Request):
    body = await request.json()
    to_number = body.get("phone", "").strip()
    if not to_number:
        return JSONResponse({"error": "phone number required"}, status_code=400)

    auth_id     = os.environ.get("PLIVO_AUTH_ID", "")
    auth_token  = os.environ.get("PLIVO_AUTH_TOKEN", "")
    from_number = os.environ.get("PLIVO_FROM_NUMBER", "")
    answer_url  = os.environ.get("PLIVO_ANSWER_URL", "https://www.blissyai.com/api/plivo/incoming")

    if not all([auth_id, auth_token, from_number]):
        return JSONResponse({"error": "PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_FROM_NUMBER not set"}, status_code=500)

    try:
        import plivo
        client = plivo.RestClient(auth_id, auth_token)
        response = client.calls.create(
            from_=from_number,
            to_=to_number,
            answer_url=answer_url,
            answer_method="POST",
        )
        logger.info(f"Plivo outbound call triggered → {to_number} | call_uuid={response.request_uuid}")
        return JSONResponse({"status": "calling", "to": to_number, "call_uuid": response.request_uuid})
    except Exception as e:
        logger.error(f"Plivo call-me failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8070)),
        reload=False,
        log_level="info",
    )
