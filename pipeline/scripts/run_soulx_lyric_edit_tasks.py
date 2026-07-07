#!/usr/bin/env python3
"""Run SoulX-Singer SVS baseline for lyric edit tasks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


class SoulXTaskError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SoulX-Singer baseline inference from lyric edit tasks."
    )
    parser.add_argument(
        "--task-manifest",
        type=Path,
        default=Path("pipeline/manifests/02_lyric_edit_tasks.gtsinger_mini_100_chinese.jsonl"),
    )
    parser.add_argument("--soulx-root", type=Path, default=Path("SoulX-Singer"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pipeline/runs/soulx_lyric_edit_gtsinger_mini_100_chinese"),
    )
    parser.add_argument("--conda-env", default="align")
    parser.add_argument("--model-path", default="pretrained_models/SoulX-Singer/model.pt")
    parser.add_argument("--config", default="soulxsinger/config/soulxsinger.yaml")
    parser.add_argument("--phoneset-path", default="soulxsinger/utils/phoneme/phone_set.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--control", choices=["melody", "score"], default="melody")
    parser.add_argument("--pitch-shift", type=int, default=0)
    parser.add_argument("--auto-shift", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only write prompt/target metadata and result records; do not run SoulX inference.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SoulXTaskError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def is_lyric_token(token: str, note_type: int) -> bool:
    return token not in {"<SP>", "<AP>"} and note_type in {2, 3}


def g2p_edited_lyrics(edited_lyrics: str, language: str, soulx_root: Path) -> list[str]:
    sys.path.insert(0, str(soulx_root.resolve()))
    from preprocess.tools.g2p import g2p_transform  # type: ignore

    chars = list(edited_lyrics)
    phonemes = g2p_transform(chars, language)
    if len(phonemes) != len(chars):
        raise SoulXTaskError(
            f"g2p length mismatch: lyrics={len(chars)} phonemes={len(phonemes)}"
        )
    return phonemes


def rewrite_segment_for_edited_lyrics(
    segment: dict[str, Any],
    edited_chars: list[str],
    edited_phonemes: list[str],
    *,
    expected_original_lyrics: str,
) -> dict[str, Any]:
    words = str(segment["text"]).split()
    phonemes = str(segment["phoneme"]).split()
    note_types = [int(x) for x in str(segment["note_type"]).split()]

    if not (len(words) == len(phonemes) == len(note_types)):
        raise SoulXTaskError(
            "metadata field length mismatch: "
            f"text={len(words)} phoneme={len(phonemes)} note_type={len(note_types)}"
        )

    rewritten_words = list(words)
    rewritten_phonemes = list(phonemes)
    original_chars: list[str] = []
    current_char_index: int | None = None

    for idx, (word, note_type) in enumerate(zip(words, note_types)):
        normalized = word.replace("<AP>", "<SP>")
        if normalized == "<SP>" or note_type == 1:
            rewritten_words[idx] = "<SP>"
            rewritten_phonemes[idx] = "<SP>"
            current_char_index = None
            continue

        if note_type == 2:
            current_char_index = len(original_chars)
            original_chars.append(normalized)
        elif note_type == 3:
            if current_char_index is None:
                raise SoulXTaskError("note_type=3 appears before any note_type=2 lyric token")
        else:
            raise SoulXTaskError(f"Unsupported note_type={note_type}")

        if current_char_index >= len(edited_chars):
            raise SoulXTaskError(
                f"edited lyric is shorter than metadata lyric at token {idx}: "
                f"char_index={current_char_index}, edited_len={len(edited_chars)}"
            )
        rewritten_words[idx] = edited_chars[current_char_index]
        rewritten_phonemes[idx] = edited_phonemes[current_char_index]

    original_lyrics = "".join(original_chars)
    if original_lyrics != expected_original_lyrics:
        raise SoulXTaskError(
            f"metadata lyrics do not match task original_lyrics: "
            f"{original_lyrics!r} != {expected_original_lyrics!r}"
        )
    if len(edited_chars) != len(original_chars):
        raise SoulXTaskError(
            f"edited lyrics length must match metadata lyrics length: "
            f"{len(edited_chars)} != {len(original_chars)}"
        )

    rewritten = dict(segment)
    rewritten["text"] = " ".join(rewritten_words)
    rewritten["phoneme"] = " ".join(rewritten_phonemes)
    return rewritten


def make_target_metadata(task: dict[str, Any], metadata: list[dict[str, Any]], soulx_root: Path) -> list[dict[str, Any]]:
    if len(metadata) != 1:
        raise SoulXTaskError(
            f"Expected one metadata segment for {task['id']}, got {len(metadata)}"
        )
    segment = metadata[0]
    language = str(segment.get("language") or "Mandarin")
    edited_lyrics = str(task["edited_lyrics"])
    edited_chars = list(edited_lyrics)
    edited_phonemes = g2p_edited_lyrics(edited_lyrics, language, soulx_root)
    return [
        rewrite_segment_for_edited_lyrics(
            segment,
            edited_chars,
            edited_phonemes,
            expected_original_lyrics=str(task["original_lyrics"]),
        )
    ]


def build_soulx_command(
    *,
    args: argparse.Namespace,
    prompt_wav_path: Path,
    prompt_metadata_path: Path,
    target_metadata_path: Path,
    item_output_dir: Path,
) -> list[str]:
    cmd = [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "python",
        "-m",
        "cli.inference",
        "--device",
        args.device,
        "--model_path",
        args.model_path,
        "--config",
        args.config,
        "--prompt_wav_path",
        str(prompt_wav_path),
        "--prompt_metadata_path",
        str(prompt_metadata_path),
        "--target_metadata_path",
        str(target_metadata_path),
        "--phoneset_path",
        args.phoneset_path,
        "--save_dir",
        str(item_output_dir),
        "--pitch_shift",
        str(args.pitch_shift),
        "--control",
        args.control,
    ]
    if args.auto_shift:
        cmd.append("--auto_shift")
    if args.fp16:
        cmd.append("--fp16")
    return cmd


def main() -> int:
    args = parse_args()
    task_manifest = args.task_manifest.resolve()
    soulx_root = args.soulx_root.resolve()
    output_dir = args.output_dir.resolve()
    metadata_dir = output_dir / "metadata"
    wav_dir = output_dir / "wavs"
    logs_dir = output_dir / "logs"
    results_path = output_dir / "inference_results.jsonl"

    tasks = load_jsonl(task_manifest)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0
    with results_path.open("w", encoding="utf-8") as results:
        for idx, task in enumerate(tasks, start=1):
            item_id = str(task["id"])
            print(f"[{idx}/{len(tasks)}] {item_id}")
            item_output_dir = wav_dir / item_id
            generated_path = item_output_dir / "generated.wav"
            target_metadata_path = metadata_dir / f"{item_id}.target.json"
            prompt_metadata_path = Path(task["metadata_path"]).resolve()
            prompt_wav_path = Path(task["audio_path"]).resolve()
            log_path = logs_dir / f"{item_id}.log"

            base_record = {
                "id": item_id,
                "audio_path": str(prompt_wav_path),
                "prompt_metadata_path": str(prompt_metadata_path),
                "target_metadata_path": str(target_metadata_path),
                "output_path": str(generated_path),
                "original_lyrics": task.get("original_lyrics"),
                "edited_lyrics": task.get("edited_lyrics"),
                "original_word": task.get("original_word"),
                "replacement_word": task.get("replacement_word"),
                "edit_start_sec": task.get("edit_start_sec"),
                "edit_end_sec": task.get("edit_end_sec"),
                "control": args.control,
                "auto_shift": bool(args.auto_shift),
                "pitch_shift": args.pitch_shift,
                "fp16": bool(args.fp16),
            }

            try:
                if generated_path.exists() and not args.overwrite and not args.prepare_only:
                    record = dict(base_record, status="skipped")
                    success += 1
                else:
                    metadata = load_json(prompt_metadata_path)
                    if not isinstance(metadata, list) or not metadata:
                        raise SoulXTaskError(f"Invalid prompt metadata: {prompt_metadata_path}")
                    target_metadata = make_target_metadata(task, metadata, soulx_root)
                    write_json(target_metadata_path, target_metadata)

                    if args.prepare_only:
                        record = dict(base_record, status="prepared")
                        success += 1
                    else:
                        item_output_dir.mkdir(parents=True, exist_ok=True)
                        cmd = build_soulx_command(
                            args=args,
                            prompt_wav_path=prompt_wav_path,
                            prompt_metadata_path=prompt_metadata_path,
                            target_metadata_path=target_metadata_path,
                            item_output_dir=item_output_dir,
                        )
                        proc = subprocess.run(
                            cmd,
                            cwd=soulx_root,
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                        )
                        log_path.write_text(proc.stdout, encoding="utf-8")
                        if proc.returncode != 0 or not generated_path.exists():
                            raise SoulXTaskError(
                                f"SoulX inference failed with returncode={proc.returncode}; see {log_path}"
                            )
                        record = dict(base_record, status="success", returncode=proc.returncode)
                        success += 1
            except Exception as exc:
                failed += 1
                error_text = traceback.format_exc() if args.verbose else str(exc)
                log_path.write_text(error_text, encoding="utf-8")
                record = dict(base_record, status="failed", error=str(exc), log_path=str(log_path))
                print(f"  failed: {exc}")

            results.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.flush()

    print(f"Done. success={success}, failed={failed}, results={results_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
