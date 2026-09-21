from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from aisha.voice.service import (
    MAX_AUDIO_BYTES,
    MAX_SPEECH_TEXT,
    VoiceInputError,
    VoiceUnavailableError,
)

router = APIRouter(prefix="/v1/voice", tags=["voice"])

LOCAL_ORIGINS = {
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
}

AUDIO_FORMATS = {
    "audio/webm": ".webm",
    "audio/mp4": ".mp4",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
}


def _check_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin is not None and origin not in LOCAL_ORIGINS:
        raise HTTPException(status_code=403, detail="Unrecognized local client origin")


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_SPEECH_TEXT)


@router.get("/status")
async def voice_status(request: Request):
    return request.app.state.aisha["voice"].status()


@router.post("/transcribe")
async def transcribe(request: Request, file: UploadFile):
    _check_origin(request)
    extension = AUDIO_FORMATS.get((file.content_type or "").split(";")[0].lower())
    if extension is None:
        raise HTTPException(status_code=415, detail="Unsupported audio format")
    try:
        payload = await file.read(MAX_AUDIO_BYTES + 1)
        if len(payload) > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="Recording exceeds 12 MiB")
        text, duration_ms = await request.app.state.aisha["voice"].transcribe(
            payload, extension,
        )
    except VoiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except VoiceInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()
    return {"text": text, "transcribe_ms": duration_ms}


@router.post("/synthesize")
async def synthesize(request: Request, speech: SpeechRequest):
    _check_origin(request)
    try:
        audio, duration_ms = await request.app.state.aisha["voice"].synthesize(speech.text)
    except VoiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except VoiceInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"X-AISHA-TTS-MS": str(duration_ms), "Cache-Control": "no-store"},
    )
