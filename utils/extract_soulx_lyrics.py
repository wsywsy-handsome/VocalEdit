#!/usr/bin/env python3
"""Extract lyrics from SoulX-style aligned metadata JSON/JSONL files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CHINESE_RE = re.compile(r"^[\u4e00-\u9fff]+$")
ENGLISH_TOKEN_RE = re.compile(r"^[A-Za-z]+(?:['-][A-Za-z]+)*$")
SUPPORTED_LANGUAGES = {"auto", "Chinese", "English"}


@dataclass
class LyricUnit:
    char: str
    char_index: int
    start_sec: float
    end_sec: float
    source_segment: int
    source_token_index: int
    note_type: int
    token_index: int | None = None
    text_start: int | None = None
    text_end: int | None = None


@dataclass
class LyricSegment:
    segment_index: int
    segment_id: str
    start_sec: float
    end_sec: float
    lyrics: str
    units: list[LyricUnit]
    language: str = "Chinese"


class LyricParseError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract parsed lyrics from SoulX aligned metadata."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Input .json metadata file, or .jsonl with one metadata object per line.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Output file path.",
    )
    parser.add_argument(
        "--mode",
        choices=("lines", "json"),
        default="lines",
        help="lines writes one lyric segment per line; json writes all parsed metadata.",
    )
    parser.add_argument(
        "--language",
        choices=sorted(SUPPORTED_LANGUAGES),
        default="auto",
        help="Lyrics language. auto uses metadata language, defaulting to Chinese.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split()]


def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split()]


def is_chinese_text(text: str) -> bool:
    return bool(CHINESE_RE.fullmatch(text))


def is_english_token(text: str) -> bool:
    return bool(ENGLISH_TOKEN_RE.fullmatch(text))


def normalize_soulx_token(token: str) -> str:
    return token.replace("<AP>", "<SP>")


def resolve_language(requested: str, metadata_language: str | None, source_record: dict[str, Any] | None) -> str:
    if requested != "auto":
        return requested
    candidate = metadata_language or (source_record or {}).get("language")
    if str(candidate).lower() == "english":
        return "English"
    return "Chinese"


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LyricParseError(f"Invalid JSON at {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[tuple[int, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append((line_no, json.loads(line)))
            except json.JSONDecodeError as exc:
                raise LyricParseError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def extract_segment_list(data: Any, *, source_label: str) -> list[dict[str, Any]]:
    if isinstance(data, dict) and "soulx_segments" in data:
        data = data["soulx_segments"]

    if not isinstance(data, list) or not data:
        raise LyricParseError(f"SoulX metadata has no segments: {source_label}")
    if not all(isinstance(item, dict) for item in data):
        raise LyricParseError(f"SoulX metadata segments must be JSON objects: {source_label}")
    return data


def load_input_records(input_path: Path) -> list[tuple[int | None, Any, list[dict[str, Any]]]]:
    if input_path.suffix.lower() == ".jsonl":
        loaded = []
        for line_no, record in read_jsonl(input_path):
            segments = extract_segment_list(record, source_label=f"{input_path}:{line_no}")
            loaded.append((line_no, record, segments))
        return loaded

    record = load_json(input_path)
    segments = extract_segment_list(record, source_label=str(input_path))
    return [(None, record, segments)]


def parse_soulx_segments(
    segments: list[dict[str, Any]],
    *,
    language: str = "auto",
    source_record: dict[str, Any] | None = None,
    source_label: str = "<input>",
) -> list[LyricSegment]:
    first_language = str(segments[0].get("language", ""))
    resolved_language = resolve_language(language, first_language, source_record)

    parsed_segments: list[LyricSegment] = []
    for seg_idx, seg in enumerate(segments):
        try:
            seg_start_sec = float(seg["time"][0]) / 1000.0
            seg_end_sec = float(seg["time"][1]) / 1000.0
            segment_id = str(seg.get("index") or f"segment_{seg_idx:03d}")
            words = [normalize_soulx_token(x) for x in str(seg["text"]).split()]
            durs = parse_float_list(str(seg["duration"]))
            types = parse_int_list(str(seg["note_type"]))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LyricParseError(f"Malformed SoulX segment in {source_label}: {exc}") from exc

        if not (len(words) == len(durs) == len(types)):
            raise LyricParseError(
                f"Length mismatch in {source_label}: "
                f"text={len(words)} duration={len(durs)} note_type={len(types)}"
            )

        units: list[LyricUnit] = []
        lyric_parts: list[str] = []
        cursor = seg_start_sec
        for token_idx, (word, dur, note_type) in enumerate(zip(words, durs, types)):
            start_sec = cursor
            end_sec = cursor + float(dur)
            cursor = end_sec

            if word == "<SP>" or note_type == 1:
                continue

            if note_type == 2:
                if resolved_language == "Chinese":
                    if not is_chinese_text(word):
                        continue
                    units.append(
                        LyricUnit(
                            char=word,
                            char_index=len(units),
                            start_sec=start_sec,
                            end_sec=end_sec,
                            source_segment=seg_idx,
                            source_token_index=token_idx,
                            note_type=note_type,
                        )
                    )
                else:
                    if not is_english_token(word):
                        continue
                    if lyric_parts:
                        lyric_parts.append(" ")
                    text_start = sum(len(part) for part in lyric_parts)
                    lyric_parts.append(word)
                    text_end = text_start + len(word)
                    units.append(
                        LyricUnit(
                            char=word,
                            char_index=text_start,
                            start_sec=start_sec,
                            end_sec=end_sec,
                            source_segment=seg_idx,
                            source_token_index=token_idx,
                            note_type=note_type,
                            token_index=len(units),
                            text_start=text_start,
                            text_end=text_end,
                        )
                    )
                continue

            if note_type == 3:
                if units:
                    units[-1].end_sec = end_sec
                continue

            raise LyricParseError(f"Unsupported note_type={note_type} in {source_label}")

        lyrics = "".join(item.char for item in units) if resolved_language == "Chinese" else "".join(lyric_parts)
        if lyrics:
            parsed_segments.append(
                LyricSegment(
                    segment_index=seg_idx,
                    segment_id=segment_id,
                    start_sec=seg_start_sec,
                    end_sec=seg_end_sec,
                    lyrics=lyrics,
                    units=units,
                    language=resolved_language,
                )
            )

    if not parsed_segments:
        raise LyricParseError(f"No {resolved_language} lyrics found in {source_label}")
    return parsed_segments


def merge_lyrics(parsed_segments: list[LyricSegment]) -> tuple[str, list[LyricUnit]]:
    lyrics_parts: list[str] = []
    units: list[LyricUnit] = []
    for segment in parsed_segments:
        unit_offset = len(units)
        text_offset = sum(len(part) for part in lyrics_parts)
        if lyrics_parts and segment.language == "English":
            lyrics_parts.append(" ")
            text_offset += 1
        lyrics_parts.append(segment.lyrics)

        for unit in segment.units:
            units.append(
                LyricUnit(
                    char=unit.char,
                    char_index=(
                        text_offset + unit.char_index
                        if segment.language == "English"
                        else unit_offset + unit.char_index
                    ),
                    start_sec=unit.start_sec,
                    end_sec=unit.end_sec,
                    source_segment=unit.source_segment,
                    source_token_index=unit.source_token_index,
                    note_type=unit.note_type,
                    token_index=(unit_offset + unit.token_index if unit.token_index is not None else None),
                    text_start=(text_offset + unit.text_start if unit.text_start is not None else None),
                    text_end=(text_offset + unit.text_end if unit.text_end is not None else None),
                )
            )
    return "".join(lyrics_parts), units


def parse_all_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    parsed_records = []
    loaded_records = load_input_records(args.input)

    for line_no, record, raw_segments in loaded_records:
        source_record = record if isinstance(record, dict) else None
        source_label = f"{args.input}:{line_no}" if line_no is not None else str(args.input)
        parsed_segments = parse_soulx_segments(
            raw_segments,
            language=args.language,
            source_record=source_record,
            source_label=source_label,
        )
        lyrics, units = merge_lyrics(parsed_segments)

        parsed_records.append(
            {
                "input_line": line_no,
                "language": parsed_segments[0].language,
                "lyrics": lyrics,
                "n_units": len(units),
                "segments": [
                    {
                        "segment_index": segment.segment_index,
                        "segment_id": segment.segment_id,
                        "start_sec": round(segment.start_sec, 6),
                        "end_sec": round(segment.end_sec, 6),
                        "duration_sec": round(segment.end_sec - segment.start_sec, 6),
                        "language": segment.language,
                        "lyrics": segment.lyrics,
                        "n_units": len(segment.units),
                        "units": [
                            {
                                "char": unit.char,
                                "start_sec": unit.start_sec,
                                "end_sec": unit.end_sec,
                            }
                            for unit in segment.units
                        ],
                    }
                    for segment in parsed_segments
                ],
            }
        )
    return parsed_records


def write_lines_output(output_path: Path, parsed_records: list[dict[str, Any]]) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record_idx, record in enumerate(parsed_records):
            if record_idx > 0:
                handle.write("\n")
            for segment in record["segments"]:
                handle.write(segment["lyrics"] + "\n")


def write_json_output(output_path: Path, parsed_records: list[dict[str, Any]], input_path: Path) -> None:
    payload = {
        "input": str(input_path),
        "n_records": len(parsed_records),
        "records": parsed_records,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        print(f"Output already exists: {args.output}. Pass --overwrite to replace.", file=sys.stderr)
        return 2

    try:
        parsed_records = parse_all_records(args)
    except Exception as exc:
        print(f"Failed to parse lyrics: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "lines":
        write_lines_output(args.output, parsed_records)
    else:
        write_json_output(args.output, parsed_records, args.input)

    n_segments = sum(len(record["segments"]) for record in parsed_records)
    n_units = sum(record["n_units"] for record in parsed_records)
    print(
        f"Wrote {args.mode} output to {args.output} "
        f"({len(parsed_records)} record(s), {n_segments} segment(s), {n_units} unit(s))."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
