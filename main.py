import io

import soundfile as sf
from fastapi import FastAPI, Depends, Form, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from model_register import get_model

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Welcome to the TTS API"}


@app.post("/audio")
def return_wav_file(
    model=Depends(get_model), file: UploadFile = File(...), text: str = Form(...)
) -> StreamingResponse:
    # Validate file type
    allowed_extensions = [".wav", ".mp3"]
    file_extension = None

    if file.filename:
        file_extension = (
            "." + file.filename.rsplit(".", 1)[-1].lower()
            if "." in file.filename
            else None
        )

    if not file_extension or file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Only .wav and .mp3 files are allowed. Received: {file.filename}",
        )
    if len(text) == 0:
        raise HTTPException(
            status_code=400,
            detail="Text is empty.",
        )
    try:
        print("here")
        audio_bytes = file.file.read()
        audio_buffer = io.BytesIO(audio_bytes)

        # --- Your function that returns (sr, wav_numpy) ---
        sampling_rate, wav_data = model.infer(
            output_path=None,
            spk_audio_prompt=audio_buffer,
            text=text,
            interval_silence=100,
            verbose=False,
        )

        output_buffer = io.BytesIO()
        sf.write(output_buffer, wav_data, sampling_rate, format="WAV")
        output_buffer.seek(0)  # Reset buffer position to start

        # --- Return the WAV file directly ---
        return_file = StreamingResponse(
            output_buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"},
        )
        return return_file
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail="some error happened")
