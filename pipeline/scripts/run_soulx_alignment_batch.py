#!/usr/bin/env python3
"""Run SoulX-Singer preprocessing for a directory of audio files."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Iterable


DEFAULT_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run SoulX-Singer alignment.")
    parser.add_argument("--audio-root", required=True, type=Path)
    parser.add_argument("--soulx-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--conda-env", default="align")
    parser.add_argument("--language", default="Mandarin")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-merge-duration", default="30000")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help="Audio extensions to include. Defaults to .wav .flac .mp3 .m4a .ogg.",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files and directories while scanning.",
    )
    parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="Follow symlinked directories while scanning.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def normalize_extensions(extensions: Iterable[str]) -> set[str]:
    normalized = set()
    for ext in extensions:
        ext = ext.strip().lower()
        if not ext:
            continue
        normalized.add(ext if ext.startswith(".") else f".{ext}")
    return normalized


def is_hidden(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    return any(part.startswith(".") for part in relative.parts)


def iter_audio_files(
    root: Path,
    extensions: set[str],
    *,
    include_hidden: bool,
    follow_symlinks: bool,
) -> list[Path]:
    if follow_symlinks:
        paths = (
            path
            for path in root.rglob("*")
            if (include_hidden or not is_hidden(path, root))
            and path.is_file()
            and path.suffix.lower() in extensions
        )
        return sorted(paths, key=lambda path: path.relative_to(root).as_posix())

    discovered: list[Path] = []
    stack = [root]
    while stack:
        current = stack.pop()
        for child in sorted(current.iterdir(), key=lambda item: item.name):
            if not include_hidden and child.name.startswith("."):
                continue
            if child.is_dir() and not child.is_symlink():
                stack.append(child)
            elif child.is_file() and child.suffix.lower() in extensions:
                discovered.append(child)
    return sorted(discovered, key=lambda path: path.relative_to(root).as_posix())


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def make_item_id(audio_path: Path, audio_root: Path, stem_counts: dict[str, int]) -> str:
    if stem_counts.get(audio_path.stem, 0) <= 1:
        return audio_path.stem
    relative = audio_path.relative_to(audio_root).as_posix()
    return f"{audio_path.stem}_{short_hash(relative)}"


def main() -> int:
    args = parse_args()
    audio_root = args.audio_root.resolve()
    soulx_root = args.soulx_root.resolve()
    output_root = args.output_root.resolve()
    logs_root = output_root / "logs"
    results_path = output_root / "alignment_results.jsonl"
    extensions = normalize_extensions(args.extensions)

    if not extensions:
        print("No valid extensions were provided.")
        return 2

    audio_files = iter_audio_files(
        audio_root,
        extensions,
        include_hidden=args.include_hidden,
        follow_symlinks=args.follow_symlinks,
    )
    stem_counts: dict[str, int] = {}
    for audio_path in audio_files:
        stem_counts[audio_path.stem] = stem_counts.get(audio_path.stem, 0) + 1

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
        for idx, source_path in enumerate(audio_files, start=1):
            audio_path = source_path.resolve()
            if str(audio_path) in completed:
                continue

            item_id = make_item_id(source_path, audio_root, stem_counts)
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

            print(f"[{idx}/{len(audio_files)}] {source_path.name}")
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
                "relative_path": source_path.relative_to(audio_root).as_posix(),
                "extension": source_path.suffix.lower(),
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
