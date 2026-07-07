#!/usr/bin/env python3
"""Run SoulX-Singer preprocessing for a directory of wav files."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run SoulX-Singer alignment.")
    parser.add_argument("--audio-root", required=True, type=Path)
    parser.add_argument("--soulx-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--conda-env", default="align")
    parser.add_argument("--language", default="Mandarin")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-merge-duration", default="30000")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio_root = args.audio_root.resolve()
    soulx_root = args.soulx_root.resolve()
    output_root = args.output_root.resolve()
    logs_root = output_root / "logs"
    results_path = output_root / "alignment_results.jsonl"

    wavs = sorted(audio_root.rglob("*.wav"))
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    completed = set()
    if args.resume and results_path.exists():
        with results_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("status") == "success":
                    completed.add(item.get("audio_path"))

    with results_path.open("a" if args.resume else "w", encoding="utf-8") as report:
        for idx, audio_path in enumerate(wavs, start=1):
            audio_path = audio_path.resolve()
            if str(audio_path) in completed:
                continue

            item_id = audio_path.stem
            item_output = output_root / "items" / item_id
            item_output.mkdir(parents=True, exist_ok=True)
            log_path = logs_root / f"{item_id}.log"

            cmd = [
                "conda",
                "run",
                "-n",
                args.conda_env,
                "python",
                "-m",
                "preprocess.pipeline",
                "--audio_path",
                str(audio_path),
                "--save_dir",
                str(item_output),
                "--language",
                args.language,
                "--device",
                args.device,
                "--vocal_sep",
                "False",
                "--max_merge_duration",
                str(args.max_merge_duration),
                "--midi_transcribe",
                "True",
            ]

            print(f"[{idx}/{len(wavs)}] {audio_path.name}")
            started = time.time()
            proc = subprocess.run(
                cmd,
                cwd=soulx_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            elapsed = time.time() - started
            log_path.write_text(proc.stdout, encoding="utf-8")

            metadata_path = item_output / "metadata.json"
            record = {
                "id": item_id,
                "audio_path": str(audio_path),
                "language": args.language,
                "output_dir": str(item_output),
                "metadata_path": str(metadata_path),
                "log_path": str(log_path),
                "returncode": proc.returncode,
                "elapsed_sec": round(elapsed, 3),
                "status": "success" if proc.returncode == 0 and metadata_path.exists() else "failed",
            }
            report.write(json.dumps(record, ensure_ascii=False) + "\n")
            report.flush()

            if record["status"] != "success":
                print(f"Failed: {audio_path.name}; see {log_path}")
                return proc.returncode or 1

    print(f"Wrote results to {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
