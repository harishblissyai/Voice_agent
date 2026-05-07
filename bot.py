import asyncio
import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

load_dotenv()

SYSTEM_PROMPT = """# Personality
நீ Priya — Blissy Restaurant, Chennai-யின் table reservations host. அன்பான, நட்பான, பக்கத்து வீட்டு அக்காவை மாதிரி பேசு. Warm South Indian hospitality with quiet efficiency.

# Language — STRICT RULE
எல்லா replies-உம் TANGLISH-இல் இருக்கணும்: Tamil script + English words naturally mixed.
NEVER reply in pure English. NEVER reply in pure Tamil.
Example style: "சரி சார், எந்த date-க்கு table வேணும்?" / "நாளைக்கு seven o'clock available இருக்கு."
Honorifics: "சார்" for men, "மேடம்" for women, default "சார்".
Fillers: "சரி சரி", "okay சார்", "sure sure", "got it சார்".

# Environment
Inbound phone calls for: new bookings, modifications, cancellations, quick questions.
Only quote details explicitly available in this conversation.

# Goal
Identify the caller's need and handle it cleanly. For a new booking, collect strictly one at a time in this order:
1. Date
2. Time slot
3. Number of guests
4. Name for the booking

Once all four collected → call save_booking tool immediately → read back details in one warm Tanglish line → tell front desk will confirm.

# Availability Rules
- Evening eight o'clock — always fully booked. Never offer it.
- If caller picks a booked slot → acknowledge once, suggest one earlier and one later alternative.
- Never list full availability. Never repeat the same alternative twice.
- If caller refuses alternatives → ask what time works for them.

# Conversation Rules
- One question per turn only.
- Briefly acknowledge the caller's last answer before asking the next question.
- If silent or unclear → gently re-ask once in different words.
- Never begin a reply with "Okay,", "Wait,", "Let me", "So,".

# TTS / Output Rules
- Spoken reply only. No markdown, no symbols, no emoji, no bullet points.
- Numbers as words: "seven o'clock", "twenty sixth", "four guests".
- Maximum two short sentences per turn. Then stop.

# Tools
- save_booking: Call once all four details collected. After success say (in Tanglish): "சரி சார், noted — front desk confirm பண்ணுவாங்க." Do NOT say "booking confirmed".
- end_call: Call when caller says goodbye. Speak a warm Tanglish farewell first, THEN call end_call.

# Confirmation example
1. Collect date, time, guests, name.
2. Call save_booking.
3. Say: "சரி சார், noted — twenty sixth, nine o'clock, four guests, Gautam-பேரில். Front desk soon confirm பண்ணுவாங்க, Blissy-க்கு call பண்ணதுக்கு thanks."
4. Caller says thanks/bye → call end_call.

# Guardrails
- No menu prices, exact availability, or offers unless provided.
- No full credit card numbers — direct to front desk.
- Large groups (above eight) or private dining → capture details, offer events team follow-up.
- Allergies or accessibility → note clearly, assure kitchen and floor team will be informed.
- Never argue. Unavailable slot → gracious, offer one alternative."""

SAVE_BOOKING_SCHEMA = FunctionSchema(
    name="save_booking",
    description="Save the booking request once all four details are collected: date, time, guests, and name. Call this before giving the confirmation line.",
    properties={
        "name":     {"type": "string", "description": "Guest name for the booking"},
        "date":     {"type": "string", "description": "Booking date (e.g. 'May 26', 'tomorrow')"},
        "time":     {"type": "string", "description": "Booking time slot (e.g. 'seven PM', 'one PM')"},
        "guests":   {"type": "string", "description": "Number of guests"},
        "occasion": {"type": "string", "description": "Special occasion if mentioned (birthday, anniversary, etc.) — leave blank if not mentioned"},
        "dietary":  {"type": "string", "description": "Dietary needs or allergies if mentioned — leave blank if not mentioned"},
    },
    required=["name", "date", "time", "guests"],
)

END_CALL_SCHEMA = FunctionSchema(
    name="end_call",
    description="End the phone call. Call this after speaking the farewell line when the caller says goodbye or after confirming booking details.",
    properties={},
    required=[],
)


async def _post_to_n8n(data: dict) -> bool:
    webhook_url = os.environ.get("N8N_WEBHOOK_URL", "")
    if not webhook_url:
        logger.warning("N8N_WEBHOOK_URL not set — booking data not saved")
        return False
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=data, timeout=aiohttp.ClientTimeout(total=8)) as r:
                logger.info(f"n8n webhook response: {r.status}")
                return r.status < 300
    except Exception as e:
        logger.error(f"n8n webhook error: {e}")
        return False


async def run_bot(webrtc_connection):
    transport = SmallWebRTCTransport(
        webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
    )

    stt = DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        settings=DeepgramSTTService.Settings(
            model="nova-3-general",
            language=Language.EN,
            punctuate=True,
            interim_results=True,
            endpointing=400,
        ),
    )

    tools = ToolsSchema(standard_tools=[SAVE_BOOKING_SCHEMA, END_CALL_SCHEMA])
    context = LLMContext(tools=tools)
    pair = LLMContextAggregatorPair(context)

    llm = GoogleLLMService(
        api_key=os.environ["GOOGLE_API_KEY"],
        system_instruction=SYSTEM_PROMPT,
        settings=GoogleLLMService.Settings(
            model="gemini-2.5-flash-lite",
            max_tokens=512,
            temperature=0.7,
        ),
    )

    # Kavitha (Tamil voice) — no language lock so sonic-3 handles
    # Tamil script + English words (Tanglish) naturally
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            model="sonic-3",
            voice=os.environ["CARTESIA_VOICE_ID"],
            # No language restriction — let sonic-3 auto-handle Tamil/English mix
        ),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        pair.user(),
        llm,
        tts,
        transport.output(),
        pair.assistant(),
    ])

    task = PipelineTask(pipeline)

    async def handle_save_booking(params):
        args = params.arguments
        logger.info(f"save_booking called: {args}")
        ok = await _post_to_n8n(args)
        if ok:
            await params.result_callback("Details saved. Proceed with the confirmation line.")
        else:
            await params.result_callback("Details noted locally. Proceed with the confirmation line.")

    async def handle_end_call(params):
        logger.info("end_call triggered — shutting down pipeline")
        await params.result_callback("Call ended.")
        await task.cancel()

    llm.register_function("save_booking", handle_save_booking)
    llm.register_function("end_call", handle_end_call)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, connection):
        await task.queue_frame(
            TTSSpeakFrame("வணக்கம்! Blissy Restaurant-க்கு வருக. நான் Priya — table booking-க்கு எப்படி உதவட்டும் சார்?")
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, connection):
        await task.queue_frame(EndFrame())

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
