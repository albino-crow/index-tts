import io

import soundfile as sf
from fastapi import FastAPI, Depends, Form, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from endpoint_object.request.voice_request import VoiceRequest
from endpoint_object.respone.voice_response import VoiceResponse
from utils.model_register import get_model
from service.dubbing_service import dub_audio

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Welcome to the TTS API"}


@app.post("/tts", response_model=VoiceResponse)
def return_wav_file(
    voice_request: VoiceRequest,
    model=Depends(get_model),
):
    return dub_audio(model, voice_request)
