from typing import Optional
from pydantic import BaseModel


class GeneratedVoice(BaseModel):
    url: str
    startTime: float
    endTime: float


class VoiceResponse(BaseModel):
    generatedVoice: GeneratedVoice
