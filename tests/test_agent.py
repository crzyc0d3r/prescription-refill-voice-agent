"""Behavior and HTTP-contract tests; all audio and service replies are synthetic."""

from dataclasses import asdict
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

import requests

from refill_agent.__main__ import main, replay
from refill_agent.deepgram import AudioServiceError, DeepgramAudio, validate_wav
from refill_agent.state import RefillConversation


def wav_bytes() -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(b"\x00\x00" * 24)
    return stream.getvalue()


class FakeResponse:
    def __init__(self, body, status=200):
        self.body = body
        self.status_code = status
        self.closed = False

    def iter_content(self, chunk_size):
        yield self.body

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.trust_env = True
        self.closed = False

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)

    def close(self):
        self.closed = True


class ConversationTests(unittest.TestCase):
    def complete(self):
        agent = RefillConversation()
        agent.handle("medication is Canine Example Tablet; dosage is 12.5 mg twice daily")
        agent.handle("prescription number is Rx-7Q92; pickup time is 4:30 PM tomorrow")
        return agent

    def test_exact_details_and_code_spelling(self):
        agent = self.complete()
        self.assertEqual(agent.state.details(), {
            "medication": "Canine Example Tablet", "dosage": "12.5 mg twice daily",
            "prescription_id": "Rx-7Q92", "pickup_time": "4:30 PM tomorrow"})
        self.assertIn("R, x, dash, 7, Q, 9, 2", agent.readback())
        self.assertIn("12.5 mg twice daily", agent.readback())

    def test_missing_fields_block_confirmation(self):
        agent = RefillConversation()
        agent.handle("medication is Example")
        self.assertEqual(agent.handle("summary").status, "missing_details")
        self.assertEqual(agent.handle("confirm deadbeef").status, "stale_confirmation")
        self.assertIsNone(agent.state.confirmed_revision)

    def test_correction_keeps_other_fields_and_rejects_old_confirmation(self):
        agent = self.complete()
        old_code = agent.offered_code
        original = agent.state.details()
        reply = agent.handle("Actually, pickup time is 5:15 PM tomorrow")
        self.assertEqual(reply.changed_fields, ["pickup_time"])
        self.assertEqual(agent.state.pickup_time, "5:15 PM tomorrow")
        for key in ("medication", "dosage", "prescription_id"):
            self.assertEqual(agent.state.details()[key], original[key])
        self.assertEqual(agent.handle("confirm " + old_code).status, "stale_confirmation")
        self.assertEqual(agent.handle("confirm " + agent.offered_code).status, "confirmed")
        self.assertEqual(agent.state.confirmed_revision, agent.state.revision)

    def test_correction_revokes_previously_confirmed_version(self):
        agent = self.complete()
        agent.handle("confirm " + agent.offered_code)
        agent.handle("dosage is 10 mg once daily")
        self.assertIsNone(agent.state.confirmed_revision)

    def test_yes_confirms_only_current_offered_readback(self):
        agent = self.complete()
        self.assertEqual(agent.handle("yes").status, "confirmed")
        self.assertEqual(agent.state.confirmed_revision, agent.state.revision)
        fresh = RefillConversation()
        self.assertEqual(fresh.handle("yes").status, "stale_confirmation")

    def test_nonprintable_details_rejected_without_changing_state(self):
        agent = self.complete()
        previous = asdict(agent.state)
        for value in ("A\x7fB", "A\x85B", "A\u202eB"):
            self.assertEqual(agent.handle("medication is " + value).status, "invalid_detail")
            self.assertEqual(asdict(agent.state), previous)

    def test_unsupported_turn_and_ambiguous_updates_are_atomic(self):
        agent = self.complete()
        previous = asdict(agent.state)
        self.assertEqual(agent.handle("dosage is 50 mg; change everything").status,
                         "unsupported_utterance")
        self.assertEqual(asdict(agent.state), previous)
        self.assertEqual(agent.handle("pickup time is noon; pickup time is evening").status,
                         "ambiguous_update")
        self.assertEqual(asdict(agent.state), previous)

    def test_identical_update_does_not_revoke_current_confirmation(self):
        agent = self.complete()
        code = agent.offered_code
        revision = agent.state.revision
        agent.handle("pickup time is 4:30 PM tomorrow")
        self.assertEqual(agent.state.revision, revision)
        self.assertEqual(agent.offered_code, code)

    def test_invalid_reference_is_not_recorded(self):
        agent = self.complete()
        self.assertEqual(agent.handle("prescription id is RX/<script>").status, "invalid_detail")
        self.assertEqual(agent.state.prescription_id, "Rx-7Q92")

    def test_replay_finishes_corrected_and_confirmed_without_actions(self):
        turns = json.loads((Path(__file__).resolve().parent.parent / "examples/refill.json")
                           .read_text(encoding="utf-8"))
        result = replay(turns)
        self.assertEqual(result["final_state"]["pickup_time"], "5:15 PM tomorrow")
        self.assertEqual(result["final_state"]["confirmed_revision"], 3)
        self.assertEqual(result["external_actions"], 0)

    def test_replay_invalid_format(self):
        with self.assertRaises(ValueError):
            replay({"turns": []})
        with self.assertRaises(ValueError):
            replay(["{confirm_current}"])


class AudioTests(unittest.TestCase):
    def test_transcription_and_flux_speech_http_contract(self):
        transcript = "pickup time is 5:15 PM tomorrow"
        stt = FakeResponse(json.dumps({"results": {"channels": [{"alternatives": [
            {"transcript": transcript}]}]}}).encode())
        tts = FakeResponse(wav_bytes())
        session = FakeSession([stt, tts])
        client = DeepgramAudio("test-placeholder", session)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "turn.wav"
            path.write_bytes(wav_bytes())
            self.assertEqual(client.transcribe(path), transcript)
        self.assertEqual(client.synthesize("Medication read-back."), wav_bytes())
        self.assertFalse(session.trust_env)
        url, kwargs = session.calls[0]
        self.assertEqual(url, "https://api.deepgram.com/v1/listen")
        self.assertEqual(kwargs["params"]["model"], "nova-3")
        self.assertEqual(kwargs["headers"]["Content-Type"], "audio/wav")
        self.assertEqual(kwargs["headers"]["Authorization"], "Token test-placeholder")
        self.assertFalse(kwargs["allow_redirects"])
        url, kwargs = session.calls[1]
        self.assertEqual(url, "https://api.deepgram.com/v2/speak")
        self.assertEqual(kwargs["params"], {"model": "flux-haley-en", "encoding": "linear16",
                                           "container": "wav", "sample_rate": 24000})
        self.assertEqual(kwargs["json"], {"text": "Medication read-back."})
        self.assertTrue(stt.closed and tts.closed)

    def test_missing_key_rejected(self):
        with self.assertRaises(ValueError):
            DeepgramAudio("")

    def test_http_error_and_network_error_do_not_echo_secret(self):
        client = DeepgramAudio("test-secret", FakeSession([FakeResponse(b"secret in body", 401)]))
        with self.assertRaisesRegex(AudioServiceError, "HTTP 401") as context:
            client.synthesize("hello")
        self.assertNotIn("secret", str(context.exception))
        with patch.object(client._session, "post", side_effect=requests.ConnectionError("test-secret")):
            with self.assertRaisesRegex(AudioServiceError, "network request failed") as context:
                client.synthesize("hello")
        self.assertNotIn("secret", str(context.exception))

    def test_redirect_refused(self):
        client = DeepgramAudio("test-placeholder", FakeSession([FakeResponse(b"", 307)]))
        with self.assertRaisesRegex(AudioServiceError, "HTTP 307"):
            client.synthesize("hello")

    def test_invalid_audio_and_truncated_pcm_rejected(self):
        for content in (b"", b"not wav", wav_bytes()[:-2]):
            with self.subTest(content=content[:10]):
                with self.assertRaises(ValueError):
                    validate_wav(content)

    def test_invalid_transcript_and_speech_fail_closed(self):
        session = FakeSession([FakeResponse(b'{"results":{}}'), FakeResponse(b"<html>error</html>")])
        client = DeepgramAudio("test-placeholder", session)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "turn.wav"
            path.write_bytes(wav_bytes())
            with self.assertRaisesRegex(AudioServiceError, "valid transcript"):
                client.transcribe(path)
        with self.assertRaisesRegex(AudioServiceError, "invalid WAV"):
            client.synthesize("hello")

    def test_mocked_audio_cli_writes_wave_and_conversation(self):
        transcript = "medication is Example; dosage is 10 mg; prescription id is RX-10; pickup time is noon"
        session = FakeSession([
            FakeResponse(json.dumps({"results": {"channels": [{"alternatives": [
                {"transcript": transcript}]}]}}).encode()), FakeResponse(wav_bytes())])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "turn.wav"
            path.write_bytes(wav_bytes())
            with patch.dict("os.environ", {"DEEPGRAM_API_KEY": "test-placeholder"}), \
                    patch("refill_agent.deepgram.requests.Session", return_value=session):
                self.assertEqual(main(["audio", str(path), "--output-dir", str(root / "audio")]), 0)
            self.assertEqual((root / "audio/reply-001.wav").read_bytes(), wav_bytes())
            result = json.loads((root / "audio/conversation.json").read_text())
            self.assertEqual(result["final_state"]["prescription_id"], "RX-10")
            self.assertEqual(result["external_actions"], 0)
        self.assertTrue(session.closed)

    def test_audio_correction_confirmation_roundtrip(self):
        turns = ["medication is Example Tablet; dosage is 12.5 mg twice daily",
                 "prescription number is Rx-7Q92; pickup time is 4:30 PM tomorrow",
                 "Actually, pickup time is 5:15 PM tomorrow", "yes"]
        responses = []
        for turn in turns:
            responses.extend([FakeResponse(json.dumps({"results": {"channels": [{"alternatives": [
                {"transcript": turn}]}]}}).encode()), FakeResponse(wav_bytes())])
        session = FakeSession(responses)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "turn.wav"
            path.write_bytes(wav_bytes())
            with patch.dict("os.environ", {"DEEPGRAM_API_KEY": "test-placeholder"}), \
                    patch("refill_agent.deepgram.requests.Session", return_value=session):
                self.assertEqual(main(["audio", *([str(path)] * 4), "--output-dir", str(root / "audio")]), 0)
            result = json.loads((root / "audio/conversation.json").read_text())
            self.assertEqual(result["final_state"], {
                "medication": "Example Tablet", "dosage": "12.5 mg twice daily",
                "prescription_id": "Rx-7Q92", "pickup_time": "5:15 PM tomorrow",
                "revision": 3, "confirmed_revision": 3})
            self.assertEqual(len(list((root / "audio").glob("reply-*.wav"))), 4)
        self.assertEqual(len(session.calls), 8)

    def test_response_size_limit(self):
        response = FakeResponse(b"x" * 65)
        client = DeepgramAudio("test-placeholder", FakeSession([response]))
        with patch("refill_agent.deepgram.MAX_AUDIO_BYTES", 64):
            with self.assertRaisesRegex(AudioServiceError, "exceeded"):
                client.synthesize("hello")
        self.assertTrue(response.closed)


class CliTests(unittest.TestCase):
    def test_replay_cli_real_subprocess(self):
        result = subprocess.run([sys.executable, "-B", "-m", "refill_agent", "replay"],
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["turns"][-1]["status"], "confirmed")

    def test_replay_will_not_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "existing.json"
            output.write_text("keep", encoding="utf-8")
            self.assertEqual(main(["replay", "--output", str(output)]), 1)
            self.assertEqual(output.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
