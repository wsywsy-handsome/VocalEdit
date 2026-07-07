#!/usr/bin/env python3
"""Create a small audio-only GTSinger sample dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path


LANGUAGES = ("Chinese", "English")
EXCLUDED_DIR_NAMES = {".ipynb_checkpoints", "Paired_Speech_Group"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly copy audio-only samples from GTSinger Chinese and English."
    )
    parser.add_argument(
        "--gtsinger-root",
        required=True,
        type=Path,
        help="Path to the original GTSinger dataset root.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Output directory for the mini dataset.",
    )
    parser.add_argument(
        "--samples-per-language",
        type=int,
        default=20,
        help="Number of wav files to sample per language.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260706,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return parser.parse_args()


def find_wavs(language_root: Path) -> list[Path]:
    return sorted(
        path
        for path in language_root.rglob("*.wav")
        if path.is_file() and EXCLUDED_DIR_NAMES.isdisjoint(path.parts)
    )


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def make_record(
    *,
    language: str,
    index: int,
    source_path: Path,
    gtsinger_root: Path,
    output_path: Path,
    output_root: Path,
) -> dict[str, object]:
    source_relative = source_path.relative_to(gtsinger_root).as_posix()
    output_relative = output_path.relative_to(output_root).as_posix()
    stat = output_path.stat()

    return {
        "id": f"{language.lower()}_{index:03d}_{short_hash(source_relative)}",
        "language": language,
        "sample_index": index,
        "audio_path": str(output_path.resolve()),
        "relative_path": output_relative,
        "source_audio_path": str(source_path.resolve()),
        "source_relative_path": source_relative,
        "filename": output_path.name,
        "extension": output_path.suffix.lower(),
        "size_bytes": stat.st_size,
        "status": "sampled",
    }


def main() -> int:
    args = parse_args()
    gtsinger_root = args.gtsinger_root.expanduser().resolve()
    output_root = args.output_root.expanduser()

    if not gtsinger_root.is_dir():
        print(f"GTSinger root is not a directory: {gtsinger_root}", file=sys.stderr)
        return 2
    if args.samples_per_language < 1:
        print("--samples-per-language must be positive.", file=sys.stderr)
        return 2
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        print(
            f"Output directory already exists and is not empty: {output_root}\n"
            "Pass --overwrite to allow adding files there.",
            file=sys.stderr,
        )
        return 2

    rng = random.Random(args.seed)
    records = []

    for language in LANGUAGES:
        language_root = gtsinger_root / language
        if not language_root.is_dir():
            print(f"Missing language directory: {language_root}", file=sys.stderr)
            return 2

        candidates = find_wavs(language_root)
        if len(candidates) < args.samples_per_language:
            print(
                f"Not enough wav files for {language}: "
                f"{len(candidates)} found, {args.samples_per_language} requested.",
                file=sys.stderr,
            )
            return 2

        sampled = rng.sample(candidates, args.samples_per_language)
        sampled.sort(key=lambda path: path.relative_to(gtsinger_root).as_posix())

        language_output = output_root / language
        language_output.mkdir(parents=True, exist_ok=True)

        for index, source_path in enumerate(sampled, start=1):
            source_relative = source_path.relative_to(gtsinger_root).as_posix()
            target_name = f"{language.lower()}_{index:03d}_{short_hash(source_relative)}.wav"
            output_path = language_output / target_name
            shutil.copy2(source_path, output_path)
            records.append(
                make_record(
                    language=language,
                    index=index,
                    source_path=source_path,
                    gtsinger_root=gtsinger_root,
                    output_path=output_path,
                    output_root=output_root,
                )
            )

    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "source_dataset": str(gtsinger_root),
        "output_dataset": str(output_root.resolve()),
        "seed": args.seed,
        "samples_per_language": args.samples_per_language,
        "total_samples": len(records),
        "languages": list(LANGUAGES),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Copied {len(records)} wav files to {output_root}")
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
