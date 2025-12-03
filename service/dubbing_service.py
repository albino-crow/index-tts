from dataclasses import dataclass
import enum
import io
import math
import numpy as np
import soundfile as sf
import subprocess
import tempfile
import os
from urllib.parse import urlparse

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import librosa

from endpoint_object.request.voice_request import (
    DiarizationSegment,
    EmotionSegment,
    VoiceRequest,
)
from endpoint_object.respone.voice_response import GeneratedVoice, VoiceResponse
from utils.minio_manager import get_minio_client
from utils.textwrap import (
    resize_sentence,
    split_sentence_by_custom_ratios_preserved_ch,
    split_sentence_by_custom_ratios_preserved_en,
)


class ResizeMode(str, enum.Enum):
    LONGER = "longer"
    SHORTER = "shorter"


@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    percent: float
    speaker: str
    emotion: str


def create_segment(
    diarization: list[DiarizationSegment], emotion: list[EmotionSegment]
):
    segments = []
    total_duration = sum(d.endTime - d.startTime for d in diarization)
    for d in diarization:
        for e in emotion:
            # find overlap
            start = max(d.startTime, e.startTime)
            end = min(d.endTime, e.endTime)
            if start < end:  # there is an overlap
                percent = (end - start) / total_duration * 100
                segments.append(
                    AudioSegment(
                        start_time=start,
                        end_time=end,
                        percent=percent,
                        speaker=d.speaker,
                        emotion=e.emotion,
                    )
                )
    total_segment_percent = sum(s.percent for s in segments)
    if total_segment_percent != 100:
        for s in segments:
            s.percent = (s.percent / total_segment_percent) * 100
    return segments


def get_length_of_voice(sample_rate, wav_data):
    """
    Calculate the length of the audio in seconds.
    :param sample_rate: Sample rate of the audio
    :param wav_data: Audio data as a numpy array
    :return: Length of the audio in seconds
    """
    if wav_data is None or len(wav_data) == 0:
        return 0.0
    return len(wav_data) / sample_rate


def compare_length_original_and_dub(length_dub, length_original) -> (float, ResizeMode):  # type: ignore
    compare = (length_dub) / length_original
    if compare < 1:
        mode = ResizeMode.LONGER
    else:
        mode = ResizeMode.SHORTER
    percentage = math.fabs(compare - 1)
    return percentage, mode


def generate_output_path(input_uri: str, prefix: str = "generated_") -> str:
    """
    Generate output path by adding a prefix to the filename.
    Example: s3://mybucket/voices/voice_01.wav -> s3://mybucket/voices/generated_voice_01.wav

    :param input_uri: Input URI (can be local path, S3 URI, etc.)
    :param prefix: Prefix to add to the filename (default: "generated_")
    :return: Output URI with prefixed filename
    """
    # Find the last slash to separate path from filename
    last_slash_index = input_uri.rfind("/")

    if last_slash_index == -1:
        # No slash found, it's just a filename
        return prefix + input_uri

    # Split into path and filename
    path = input_uri[: last_slash_index + 1]
    filename = input_uri[last_slash_index + 1 :]

    # Combine path with prefixed filename
    return path + prefix + filename


def valid_input(request: VoiceRequest) -> (bool, str):  # type: ignore
    allowed_extensions = [".wav", ".mp3"]
    file_extension = None
    filename = request.sourceVoice.url
    if filename:
        file_extension = (
            "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else None
        )

    if not file_extension or file_extension not in allowed_extensions:
        return (
            False,
            f"Invalid file type. Only .wav and .mp3 files are allowed. Received: {filename}",
        )
    if len(request.text) == 0:
        return False, "Text is empty."

    return True, None


def parse_storage_url(url: str) -> tuple[str, str]:
    """
    Parse storage URL to extract bucket name and object path.
    Supports both S3 URIs and HTTP/HTTPS URLs.

    Examples:
        s3://mybucket/path/to/file.wav -> ("mybucket", "path/to/file.wav")
        http://127.0.0.1:9000/mybucket/path/to/file.wav -> ("mybucket", "path/to/file.wav")

    :param url: Storage URL (S3 URI or HTTP URL)
    :return: Tuple of (bucket_name, object_name)
    """
    if url.startswith("s3://"):
        # S3 URI format: s3://bucket-name/path/to/file
        uri_parts = url.replace("s3://", "").split("/", 1)
        bucket_name = uri_parts[0]
        object_name = uri_parts[1] if len(uri_parts) > 1 else ""
    elif url.startswith(("http://", "https://")):
        # HTTP URL format: http://host:port/bucket-name/path/to/file
        parsed = urlparse(url)
        path_parts = parsed.path.lstrip("/").split("/", 1)
        bucket_name = path_parts[0] if len(path_parts) > 0 else ""
        object_name = path_parts[1] if len(path_parts) > 1 else ""
    else:
        # Fallback: treat as local path or unknown format
        raise ValueError(
            f"Unsupported URL format: {url}. Expected s3://, http://, or https://"
        )

    return bucket_name, object_name


def create_silence(sampling_rate, duration_ms):
    silence = np.zeros(int(duration_ms * sampling_rate))
    return silence


def adjust_audio_speed(wav_data, sampling_rate, target_duration):
    """
    Adjust audio speed to match target duration using ffmpeg's atempo filter.
    :param wav_data: Audio data as numpy array
    :param sampling_rate: Sample rate of the audio
    :param target_duration: Target duration in seconds
    :return: Speed-adjusted audio data
    """
    current_duration = len(wav_data) / sampling_rate
    if current_duration == 0:
        return wav_data

    speed_factor = current_duration / target_duration

    # ffmpeg's atempo filter only supports speed factors between 0.5 and 2.0
    # For larger changes, we need to chain multiple atempo filters
    if speed_factor < 0.5 or speed_factor > 100:
        # If speed change is too extreme, fall back to returning original
        return wav_data

    # Create temporary files for ffmpeg processing
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
        temp_input_path = temp_input.name
        sf.write(temp_input_path, wav_data, sampling_rate)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
        temp_output_path = temp_output.name

    try:
        # Build atempo filter chain for speed factors outside 0.5-2.0 range
        atempo_filters = []
        remaining_factor = speed_factor

        while remaining_factor > 2.0:
            atempo_filters.append("atempo=2.0")
            remaining_factor /= 2.0

        while remaining_factor < 0.5:
            atempo_filters.append("atempo=0.5")
            remaining_factor /= 0.5

        if remaining_factor != 1.0:
            atempo_filters.append(f"atempo={remaining_factor}")

        filter_chain = ",".join(atempo_filters) if atempo_filters else "anull"

        # Run ffmpeg command
        cmd = [
            "ffmpeg",
            "-i",
            temp_input_path,
            "-af",
            filter_chain,
            "-y",  # Overwrite output file
            temp_output_path,
        ]

        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if result.returncode != 0:
            return wav_data

        # Read the adjusted audio
        wav_adjusted, _ = sf.read(temp_output_path)

        # Calculate actual duration after adjustment

        return wav_adjusted

    finally:
        # Clean up temporary files
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)


def create_output_seq(dubs, segments, duration):
    output_sequence = []
    current_time = 0.0
    sampling_rate = dubs[0][1]
    for index, segment in enumerate(segments):
        # Add silence before the segment if needed
        if segment.start_time > current_time:
            silence_duration = segment.start_time - current_time
            silence = create_silence(sampling_rate, silence_duration * 1000)
            output_sequence.append(silence)

        # Add the dubbed audio segment
        dub_audio, _ = dubs[index]
        output_sequence.append(dub_audio)
        current_time = segment.end_time

    if current_time < duration:
        silence_duration = duration - current_time
        silence = create_silence(sampling_rate, silence_duration * 1000)
        output_sequence.append(silence)
    return np.concatenate(output_sequence)


def read_audio_from_minio(minio_client, url: str):
    # Parse URL to get bucket and object
    bucket_name, object_name = parse_storage_url(url)

    # Download from MinIO to BytesIO
    response = minio_client.get_object(bucket_name, object_name)
    audio_buffer = io.BytesIO(response.read())
    response.close()
    response.release_conn()
    audio_buffer.seek(0)

    # Read audio directly from BytesIO using soundfile
    audio, sample_rate = sf.read(audio_buffer)
    audio_buffer.close()

    return audio, sample_rate


def dub_audio(model, request: VoiceRequest):
    try:
        minio_client = get_minio_client()
        valid, message = valid_input(request)
        if not valid:
            return VoiceResponse(
                generatedVoice=None,
                status=400,
                message=f"Invalid request data: {message}",
            )
        segments = create_segment(request.diarization, request.emotion)
        total_length = sum(d.endTime - d.startTime for d in request.diarization)
        segments_percent = [s.percent for s in segments]
        if request.targetLanguage.lower() == "english":
            parts = split_sentence_by_custom_ratios_preserved_en(
                request.text, segments_percent
            )

        else:
            parts = split_sentence_by_custom_ratios_preserved_ch(
                request.text, segments_percent
            )

        audio, sample_rate = read_audio_from_minio(
            minio_client, request.sourceVoice.url
        )

        start = 0
        number_of_try = 3
        best_dub = []

        for index, (part, percent) in enumerate(zip(parts, segments_percent)):
            end = start + int(len(audio) * (percent / 100))

            audio_segment = (
                audio[start:end] if index < len(parts) - 1 else audio[start:]
            )

            start = end
            is_length_acceptable = False
            created_dubs = []
            for try_num in range(number_of_try):
                # Write audio segment to BytesIO buffer

                audio_buffer = io.BytesIO()
                sf.write(audio_buffer, audio_segment, sample_rate, format="WAV")
                audio_buffer.seek(0)

                try:
                    sampling_rate, wav_data = model.infer(
                        output_path=None,
                        spk_audio_prompt=audio_buffer,
                        text=part,
                        interval_silence=100,
                        verbose=False,
                    )
                except Exception as e:
                    raise e
                finally:
                    audio_buffer.close()

                length_of_dub = get_length_of_voice(sampling_rate, wav_data)
                length_diff, mode = compare_length_original_and_dub(
                    length_of_dub, total_length * percent / 100
                )
                created_dubs.append(
                    ((sampling_rate, wav_data), mode, length_of_dub, length_diff)
                )

                if length_diff <= 0.1:  # within 10% difference
                    is_length_acceptable = True

                    break
                part = resize_sentence(part, length_diff, mode, "English")

            selected_dub = None
            if is_length_acceptable:
                selected_dub = created_dubs[-1]
            else:
                # If none acceptable, take the last one
                smallest_diff = float("inf")

                for dub in created_dubs:
                    if dub[3] < smallest_diff:
                        smallest_diff = dub[3]
                        selected_dub = dub

            sample_rate, wave_data = selected_dub[0]
            adjust_wave_data = adjust_audio_speed(
                wave_data, sample_rate, total_length * percent / 100
            )
            best_dub.append((adjust_wave_data, sample_rate))

        output_sequence = create_output_seq(best_dub, segments, total_length)
        output_sequence = adjust_audio_speed(output_sequence, sample_rate, total_length)

        # Generate output path based on input URI
        output_path = generate_output_path(request.sourceVoice.url)

        # Parse storage URL for output (supports both S3 URI and HTTP URL)
        bucket_name, object_name = parse_storage_url(output_path)

        # Write audio to BytesIO buffer
        output_buffer = io.BytesIO()
        sf.write(output_buffer, output_sequence, best_dub[0][1], format="WAV")
        output_buffer.seek(0)

        # Upload to MinIO
        minio_client.put_object(
            bucket_name,
            object_name,
            output_buffer,
            length=output_buffer.getbuffer().nbytes,
            content_type="audio/wav",
        )
        output_buffer.close()

        response = VoiceResponse(
            generatedVoice=GeneratedVoice(
                url=output_path,
                startTime=0.0,
                endTime=len(output_sequence) / sampling_rate,
            ),
            status=200,
            message=None,
        )
        return response
    except Exception as e:
        # raise e
        return VoiceResponse(generatedVoice=None, status=500, message=str(e))
