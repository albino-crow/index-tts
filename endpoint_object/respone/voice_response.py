from typing import Optional
from pydantic import BaseModel


class GeneratedVoice(BaseModel):
    url: str
    startTime: float
    endTime: float


class VoiceResponse(BaseModel):
    generatedVoice: GeneratedVoice

    model_config = {
        "json_schema_extra": {
            "example": {
                "sourceVoice": {
                    "url": "http://127.0.0.1:9000/mybucket/two_emotion.wav",
                    "startTime": 0.0,
                    "endTime": 8.0,
                },
                "text": "Hello everyone, welcome to our session today!",
                "targetLanguage": "English",
                "diarization": [
                    {"startTime": 0.0, "endTime": 4.38, "speaker": "A"},
                    {"startTime": 4.38, "endTime": 8, "speaker": "B"},
                ],
                "emotion": [
                    {"startTime": 0.0, "endTime": 4.38, "emotion": "angry"},
                    {"startTime": 4.38, "endTime": 8, "emotion": "normal"},
                ],
            }
        }
    }
