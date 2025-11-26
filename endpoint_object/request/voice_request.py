from typing import List
from pydantic import BaseModel


class SourceVoice(BaseModel):
    url: str
    startTime: float
    endTime: float


class DiarizationSegment(BaseModel):
    startTime: float
    endTime: float
    speaker: str


class EmotionSegment(BaseModel):
    startTime: float
    endTime: float
    emotion: str


class VoiceRequest(BaseModel):
    sourceVoice: SourceVoice
    text: str
    diarization: List[DiarizationSegment]
    emotion: List[EmotionSegment]
    targetLanguage: str
