"""CLI for offline read-back replay, text conversation, and recorded audio turns."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

from .state import RefillConversation


def replay(turns: list[str]) -> dict:
    if not isinstance(turns, list) or not all(isinstance(turn, str) for turn in turns):
        raise ValueError("Replay input must be a JSON array of utterance strings.")
    conversation = RefillConversation()
    trace = []
    for turn in turns:
        # The fixture uses this marker only after the current read-back was offered.
        if turn == "{confirm_current}":
            if conversation.offered_code is None:
                raise ValueError("No current read-back is available to confirm.")
            turn = "confirm " + conversation.offered_code
        trace.append({"utterance": turn, **asdict(conversation.handle(turn))})
    return {"turns": trace, "final_state": asdict(conversation.state), "external_actions": 0}


def interactive_paths():
    for line in sys.stdin:
        if line.strip().lower() == "quit":
            return
        if line.strip():
            yield Path(line.strip().strip('"'))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Retain and read back synthetic refill details.")
    commands = parser.add_subparsers(dest="command", required=True)
    offline = commands.add_parser("replay", help="Run a synthetic JSON conversation without network.")
    offline.add_argument("--input", type=Path, default=Path(__file__).resolve().parent.parent / "examples" / "refill.json")
    offline.add_argument("--output", type=Path)
    commands.add_parser("chat", help="Use the explicit detail grammar interactively; no network.")
    live = commands.add_parser("audio", help="Send each WAV turn to Deepgram and save Flux WAV replies.")
    live.add_argument("turns", type=Path, nargs="*")
    live.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "replay":
            result = replay(json.loads(args.input.read_text(encoding="utf-8")))
            text = json.dumps(result, indent=2, ensure_ascii=False)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x", encoding="utf-8") as stream:
                    stream.write(text + "\n")
            print(text)
        elif args.command == "chat":
            conversation = RefillConversation()
            print("Enter labeled details; separate fields with semicolons. Say summary. Type quit to exit.")
            for line in sys.stdin:
                if line.strip().lower() == "quit":
                    break
                print(conversation.handle(line).text, flush=True)
        else:
            from .deepgram import DeepgramAudio, validate_wav
            # Validate all paths before creating output or sending any audio.
            for path in args.turns:
                if path.stat().st_size > 25 * 1024 * 1024:
                    raise ValueError("Input WAV exceeds 25 MiB.")
                validate_wav(path.read_bytes())
            audio = DeepgramAudio(os.environ.get("DEEPGRAM_API_KEY", ""))
            try:
                args.output_dir.mkdir(parents=True, exist_ok=False)
                conversation = RefillConversation()
                trace = []
                if args.turns:
                    paths = iter(args.turns)
                else:
                    print("Enter a recorded WAV path for each turn; play each saved reply. Type quit to finish.")
                    paths = interactive_paths()
                for index, path in enumerate(paths, 1):
                    transcript = audio.transcribe(path)
                    reply = conversation.handle(transcript)
                    speech = audio.synthesize(reply.text)
                    with (args.output_dir / f"reply-{index:03}.wav").open("xb") as stream:
                        stream.write(speech)
                    trace.append({"turn": index, "utterance": transcript, **asdict(reply)})
                    (args.output_dir / "conversation.json").write_text(
                        json.dumps({"turns": trace, "final_state": asdict(conversation.state),
                                    "external_actions": 0}, indent=2), encoding="utf-8")
                    print(f"Turn {index}: {reply.text}", flush=True)
            finally:
                audio.close()
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
