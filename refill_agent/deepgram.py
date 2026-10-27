"""Fixed-origin Deepgram REST audio adapters with no implicit network calls."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import requests


MAX_AUDIO_BYTES = 25 * 1024 * 1024


class AudioServiceError(RuntimeError):
    """Sanitized service failure; credentials and response bodies are excluded."""


def validate_wav(audio: bytes) -> None:
    if not audio or len(audio) > MAX_AUDIO_BYTES:
        raise ValueError("WAV audio must be nonempty and at most 25 MiB.")
    try:
        with wave.open(io.BytesIO(audio), "rb") as stream:
            if stream.getnframes() < 1 or stream.getcomptype() != "NONE":
                raise ValueError("Expected nonempty uncompressed PCM WAV audio.")
            if stream.getsampwidth() != 2 or stream.getnchannels() != 1:
                raise ValueError("Expected mono, 16-bit PCM WAV audio.")
            if len(stream.readframes(stream.getnframes())) != stream.getnframes() * 2:
                raise ValueError("Truncated PCM WAV audio.")
    except (wave.Error, EOFError) as exc:
        raise ValueError("Expected valid PCM WAV audio.") from exc


class DeepgramAudio:
    """Nova-3 prerecorded recognition and Flux batch synthesis for file turns."""

    def __init__(self, api_key: str, session=None) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("Set DEEPGRAM_API_KEY before using the audio command.")
        self._api_key = api_key.strip()
        self._session = session if session is not None else requests.Session()
        self._session.trust_env = False

    def _post(self, endpoint: str, **kwargs):
        try:
            response = self._session.post(
                "https://api.deepgram.com" + endpoint,
                headers={"Authorization": "Token " + self._api_key, **kwargs.pop("headers", {})},
                timeout=(10, 60), allow_redirects=False, stream=True, **kwargs,
            )
        except requests.RequestException:
            raise AudioServiceError("Deepgram network request failed.") from None
        if response.status_code != 200:
            response.close()
            raise AudioServiceError(f"Deepgram returned HTTP {response.status_code}.")
        return response

    @staticmethod
    def _read(response) -> bytes:
        parts = []
        size = 0
        try:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                size += len(chunk)
                if size > MAX_AUDIO_BYTES:
                    raise AudioServiceError("Deepgram response exceeded 25 MiB.")
                parts.append(chunk)
        except requests.RequestException:
            raise AudioServiceError("Deepgram response transfer failed.") from None
        finally:
            response.close()
        return b"".join(parts)

    def transcribe(self, path: Path) -> str:
        if path.stat().st_size > MAX_AUDIO_BYTES:
            raise ValueError("Input WAV exceeds 25 MiB.")
        audio = path.read_bytes()
        validate_wav(audio)
        response = self._post(
            "/v1/listen", params={"model": "nova-3", "smart_format": "true", "language": "en"},
            headers={"Content-Type": "audio/wav"}, data=audio,
        )
        try:
            import json
            document = json.loads(self._read(response))
            transcript = document["results"]["channels"][0]["alternatives"][0]["transcript"]
            if not isinstance(transcript, str) or not transcript.strip() or len(transcript) > 2000:
                raise ValueError
            return transcript.strip()
        except (ValueError, KeyError, IndexError, TypeError):
            raise AudioServiceError("Deepgram returned no valid transcript for this turn.") from None

    def synthesize(self, text: str) -> bytes:
        if not text or len(text) > 4000:
            raise ValueError("Speech text must contain 1 to 4000 characters.")
        response = self._post(
            "/v2/speak", params={"model": "flux-haley-en", "encoding": "linear16",
                                 "container": "wav", "sample_rate": 24000},
            headers={"Content-Type": "application/json"}, json={"text": text},
        )
        audio = self._read(response)
        try:
            validate_wav(audio)
        except ValueError:
            raise AudioServiceError("Deepgram returned invalid WAV speech audio.") from None
        return audio

    def close(self) -> None:
        self._session.close()
