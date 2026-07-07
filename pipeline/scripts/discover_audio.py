#!/usr/bin/env python3
"""Recursively discover audio files and write a JSONL manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively find audio files and write 00_discovered.jsonl."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Root directory to scan recursively.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSONL manifest path.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help="Audio extensions to include. Defaults to common audio formats.",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files and directories.",
    )
    parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="Follow symlinked directories while scanning.",
    )
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
) -> Iterable[Path]:
    if follow_symlinks:
        for path in root.rglob("*"):
            if not include_hidden and is_hidden(path, root):
                continue
            if path.is_file() and path.suffix.lower() in extensions:
                yield path
        return

    stack = [root]
    while stack:
        current = stack.pop()
        for child in sorted(current.iterdir(), key=lambda item: item.name):
            if not include_hidden and child.name.startswith("."):
                continue
            if child.is_dir() and not child.is_symlink():
                stack.append(child)
            elif child.is_file() and child.suffix.lower() in extensions:
                yield child


def stable_id(relative_path: str) -> str:
    return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:12]


def make_record(path: Path, root: Path) -> dict[str, object]:
    resolved_path = path.resolve()
    relative_path = path.relative_to(root).as_posix()
    parts = Path(relative_path).parts
    stat = path.stat()

    return {
        "id": stable_id(relative_path),
        "audio_path": str(resolved_path),
        "relative_path": relative_path,
        "filename": path.name,
        "stem": path.stem,
        "extension": path.suffix.lower(),
        "speaker": parts[0] if len(parts) > 1 else None,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "status": "discovered",
    }


def main() -> int:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    output = args.output.expanduser()
    extensions = normalize_extensions(args.extensions)

    if not root.exists():
        print(f"Dataset root does not exist: {root}", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"Dataset root is not a directory: {root}", file=sys.stderr)
        return 2
    if not extensions:
        print("No valid extensions were provided.", file=sys.stderr)
        return 2

    records = [
        make_record(path, root)
        for path in iter_audio_files(
            root,
            extensions,
            include_hidden=args.include_hidden,
            follow_symlinks=args.follow_symlinks,
        )
    ]
    records.sort(key=lambda record: str(record["relative_path"]))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
