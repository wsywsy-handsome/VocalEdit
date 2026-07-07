#!/usr/bin/env python3
"""Run YingMusic-Singer-Plus lyric edit tasks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YingMusic lyric edit tasks.")
    parser.add_argument(
        "--task-manifest",
        type=Path,
        default=Path("pipeline/manifests/02_lyric_edit_tasks.gtsinger_mini_40_chinese.jsonl"),
    )
    parser.add_argument("--yingmusic-root", type=Path, default=Path("YingMusic-Singer-Plus"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pipeline/runs/yingmusic_lyric_edit_gtsinger_mini_40_chinese"),
    )
    parser.add_argument("--conda-env", default="ymsp")
    parser.add_argument("--ckpt-path", default="ASLP-lab/YingMusic-Singer-Plus")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nfe-step", type=int, default=32)
    parser.add_argument("--cfg-strength", type=float, default=3.0)
    parser.add_argument("--t-shift", type=float, default=0.5)
    parser.add_argument("--sil-len-to-end", type=float, default=0.5)
    parser.add_argument(
        "--mask-start-offset-sec",
        type=float,
        default=0.0,
        help="Expand the lyric edit mask this many seconds before edit_start_sec.",
    )
    parser.add_argument(
        "--mask-end-offset-sec",
        type=float,
        default=0.0,
        help="Expand the lyric edit mask this many seconds after edit_end_sec.",
    )
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-conda-reexec", action="store_true")
    return parser.parse_args()


def maybe_reexec_in_conda(args: argparse.Namespace) -> None:
    if args.no_conda_reexec or not args.conda_env:
        return
    if os.environ.get("CONDA_DEFAULT_ENV") == args.conda_env:
        return

    cmd = [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "python",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--no-conda-reexec",
    ]
    raise SystemExit(subprocess.call(cmd))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main() -> int:
    args = parse_args()
    maybe_reexec_in_conda(args)

    import torch
    import torchaudio

    task_manifest = args.task_manifest.resolve()
    output_dir = args.output_dir.resolve()
    yingmusic_root = args.yingmusic_root.resolve()
    if args.mask_start_offset_sec < 0 or args.mask_end_offset_sec < 0:
        raise ValueError("--mask-start-offset-sec and --mask-end-offset-sec must be non-negative")

    ckpt_path = (
        str(Path(args.ckpt_path).resolve())
        if Path(args.ckpt_path).exists()
        else args.ckpt_path
    )

    os.chdir(yingmusic_root)
    sys.path.insert(0, str(yingmusic_root))

    from src.YingMusicSinger.infer.YingMusicSinger import YingMusicSinger

    tasks = load_jsonl(task_manifest)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    wav_dir = output_dir / "wavs"
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "inference_results.jsonl"

    model = YingMusicSinger.from_pretrained(ckpt_path)
    model = model.to(args.device)
    model.eval()

    success = 0
    failed = 0
    with results_path.open("w", encoding="utf-8") as results:
        for idx, task in enumerate(tasks, start=1):
            item_id = task["id"]
            out_path = wav_dir / f"{item_id}.wav"
            log_path = logs_dir / f"{item_id}.log"
            print(f"[{idx}/{len(tasks)}] {item_id}")

            if out_path.exists() and not args.overwrite:
                record = {
                    "id": item_id,
                    "audio_path": task["audio_path"],
                    "output_path": str(out_path),
                    "status": "skipped",
                }
                results.write(json.dumps(record, ensure_ascii=False) + "\n")
                results.flush()
                success += 1
                continue

            try:
                with torch.inference_mode():
                    audio, sr = model(
                        ref_audio_path=None,
                        melody_audio_path=task["audio_path"],
                        ref_text=task["original_lyrics"],
                        target_text=task["edited_lyrics"],
                        lrc_align_mode="sentence_level",
                        sil_len_to_end=args.sil_len_to_end,
                        t_shift=args.t_shift,
                        nfe_step=args.nfe_step,
                        cfg_strength=args.cfg_strength,
                        seed=args.seed + idx,
                        edit_start_sec=task["edit_start_sec"],
                        edit_end_sec=task["edit_end_sec"],
                        edit_mask_start_offset_sec=args.mask_start_offset_sec,
                        edit_mask_end_offset_sec=args.mask_end_offset_sec,
                    )
                torchaudio.save(str(out_path), audio.cpu(), sample_rate=sr)
                record = {
                    "id": item_id,
                    "audio_path": task["audio_path"],
                    "output_path": str(out_path),
                    "original_lyrics": task["original_lyrics"],
                    "edited_lyrics": task["edited_lyrics"],
                    "original_word": task["original_word"],
                    "replacement_word": task["replacement_word"],
                    "edit_start_sec": task["edit_start_sec"],
                    "edit_end_sec": task["edit_end_sec"],
                    "mask_start_offset_sec": args.mask_start_offset_sec,
                    "mask_end_offset_sec": args.mask_end_offset_sec,
                    "mask_start_sec": max(0.0, float(task["edit_start_sec"]) - args.mask_start_offset_sec),
                    "mask_end_sec": float(task["edit_end_sec"]) + args.mask_end_offset_sec,
                    "status": "success",
                }
                log_path.write_text("", encoding="utf-8")
                success += 1
            except Exception as exc:
                failed += 1
                error_text = traceback.format_exc() if args.verbose else str(exc)
                log_path.write_text(error_text, encoding="utf-8")
                record = {
                    "id": item_id,
                    "audio_path": task.get("audio_path"),
                    "output_path": str(out_path),
                    "log_path": str(log_path),
                    "status": "failed",
                    "error": str(exc),
                }
                print(f"  failed: {exc}")

            results.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.flush()

    print(f"Done. success={success}, failed={failed}, results={results_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
