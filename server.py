import asyncio
import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from bot import run_bot

load_dotenv()

request_handler = SmallWebRTCRequestHandler()


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


@app.post("/offer")
async def offer(request: Request):
    body = await request.json()
    rtc_request = SmallWebRTCRequest.from_dict(body)

    async def bot_callback(connection):
        asyncio.create_task(run_bot(connection))

    answer = await request_handler.handle_web_request(rtc_request, bot_callback)
    return answer


@app.patch("/ice")
async def ice(request: Request):
    body = await request.json()
    patch_request = SmallWebRTCPatchRequest(**body)
    await request_handler.handle_patch_request(patch_request)
    return {"status": "ok"}


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Blissy Restaurant Bot"}


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 7860)),
        reload=False,
        log_level="info",
    )
