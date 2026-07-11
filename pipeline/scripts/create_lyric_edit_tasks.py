#!/usr/bin/env python3
"""Create lyric edit tasks from SoulX-Singer aligned metadata."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from pypinyin import Style, lazy_pinyin


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com/chat/completions"
CHINESE_RE = re.compile(r"^[\u4e00-\u9fff]+$")
ENGLISH_TOKEN_RE = re.compile(r"^[A-Za-z]+(?:['-][A-Za-z]+)*$")
SUPPORTED_LANGUAGES = {"auto", "Chinese", "English"}


@dataclass
class LyricChar:
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
    chars: list[LyricChar]
    language: str = "Chinese"


class TaskError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate lyric edit tasks from SoulX metadata."
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=Path("pipeline/manifests/01_aligned.gtsinger_mini_40_chinese.jsonl"),
        help="Aligned manifest JSONL with metadata_path fields.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pipeline/manifests/02_lyric_edit_tasks.gtsinger_mini_40_chinese.jsonl"),
        help="Output success task JSONL.",
    )
    parser.add_argument(
        "--failed-output",
        type=Path,
        default=None,
        help="Output failed task JSONL. Defaults to <output stem>.failed.jsonl.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("pipeline/.env"),
        help="Env file containing deepseek_api_key.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-word-len", type=int, default=4)
    parser.add_argument(
        "--language",
        choices=sorted(SUPPORTED_LANGUAGES),
        default="auto",
        help="Task language. auto uses the aligned manifest/metadata language.",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse metadata and print candidate summaries without calling DeepSeek.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def load_env_key(env_file: Path) -> str:
    if not env_file.exists():
        raise TaskError(f"Env file does not exist: {env_file}")
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "deepseek_api_key":
            value = value.strip().strip('"').strip("'")
            if value:
                return value
    raise TaskError(f"deepseek_api_key was not found in {env_file}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise TaskError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return items


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


def resolve_language(
    requested: str,
    aligned_item: dict[str, Any] | None,
    metadata_language: str | None,
) -> str:
    if requested != "auto":
        return requested
    candidate = metadata_language or (aligned_item or {}).get("language")
    if str(candidate).lower() == "english":
        return "English"
    return "Chinese"


def parse_soulx_metadata_segments(
    metadata_path: Path,
    *,
    language: str = "auto",
    aligned_item: dict[str, Any] | None = None,
) -> list[LyricSegment]:
    segments = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(segments, list) or not segments:
        raise TaskError(f"SoulX metadata has no segments: {metadata_path}")

    first_language = str(segments[0].get("language", "")) if isinstance(segments[0], dict) else None
    resolved_language = resolve_language(language, aligned_item, first_language)
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
            raise TaskError(f"Malformed SoulX segment in {metadata_path}: {exc}") from exc

        if not (len(words) == len(durs) == len(types)):
            raise TaskError(
                f"Length mismatch in {metadata_path}: "
                f"text={len(words)} duration={len(durs)} note_type={len(types)}"
            )

        chars: list[LyricChar] = []
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
                    chars.append(
                        LyricChar(
                            char=word,
                            char_index=len(chars),
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
                    chars.append(
                        LyricChar(
                            char=word,
                            char_index=text_start,
                            start_sec=start_sec,
                            end_sec=end_sec,
                            source_segment=seg_idx,
                            source_token_index=token_idx,
                            note_type=note_type,
                            token_index=len(chars),
                            text_start=text_start,
                            text_end=text_end,
                        )
                    )
                continue
            if note_type == 3:
                if chars:
                    chars[-1].end_sec = end_sec
                continue

            raise TaskError(f"Unsupported note_type={note_type} in {metadata_path}")

        lyrics = "".join(item.char for item in chars) if resolved_language == "Chinese" else "".join(lyric_parts)
        if lyrics:
            parsed_segments.append(
                LyricSegment(
                    segment_index=seg_idx,
                    segment_id=segment_id,
                    start_sec=seg_start_sec,
                    end_sec=seg_end_sec,
                    lyrics=lyrics,
                    chars=chars,
                    language=resolved_language,
                )
            )

    if not parsed_segments:
        raise TaskError(f"No {resolved_language} lyrics found in {metadata_path}")
    return parsed_segments


def parse_soulx_metadata(metadata_path: Path) -> tuple[str, list[LyricChar]]:
    segments = parse_soulx_metadata_segments(metadata_path)
    lyrics_parts: list[str] = []
    chars: list[LyricChar] = []
    for segment in segments:
        offset = len(chars)
        text_offset = sum(len(part) for part in lyrics_parts)
        if lyrics_parts and segment.language == "English":
            lyrics_parts.append(" ")
            text_offset += 1
        lyrics_parts.append(segment.lyrics)
        for char in segment.chars:
            chars.append(
                LyricChar(
                    char=char.char,
                    char_index=(text_offset + char.char_index if segment.language == "English" else offset + char.char_index),
                    start_sec=char.start_sec,
                    end_sec=char.end_sec,
                    source_segment=char.source_segment,
                    source_token_index=char.source_token_index,
                    note_type=char.note_type,
                    token_index=(offset + char.token_index if char.token_index is not None else None),
                    text_start=(text_offset + char.text_start if char.text_start is not None else None),
                    text_end=(text_offset + char.text_end if char.text_end is not None else None),
                )
            )
    return "".join(lyrics_parts), chars

def pinyin_no_tone(text: str) -> list[str]:
    return lazy_pinyin(text, style=Style.NORMAL, errors="default")


def find_english_token_span(
    *,
    lyrics: str,
    chars: list[LyricChar],
    original_word: str,
    token_start: int | None,
    token_end: int | None,
    char_start: int | None,
    char_end: int | None,
) -> tuple[int, int, int, int]:
    if token_start is not None and token_end is not None:
        if not (0 <= token_start < token_end <= len(chars)):
            raise TaskError("token_start/token_end out of range")
        selected = chars[token_start:token_end]
        text_start = selected[0].text_start
        text_end = selected[-1].text_end
        if text_start is None or text_end is None:
            raise TaskError("English token is missing text offsets")
        span = lyrics[text_start:text_end]
        if span != original_word:
            raise TaskError(f"original_word does not match token span: {original_word!r} != {span!r}")
        return text_start, text_end, token_start, token_end

    if char_start is None or char_end is None:
        raise TaskError("English edit must provide token_start/token_end or char_start/char_end")
    if not (0 <= char_start < char_end <= len(lyrics)):
        raise TaskError("char_start/char_end out of range")
    if lyrics[char_start:char_end] != original_word:
        raise TaskError(
            f"original_word does not match lyrics span: {original_word!r} != {lyrics[char_start:char_end]!r}"
        )

    matched = [
        idx
        for idx, token in enumerate(chars)
        if token.text_start is not None
        and token.text_end is not None
        and token.text_start >= char_start
        and token.text_end <= char_end
    ]
    if not matched:
        raise TaskError("char span does not cover any English token")
    if chars[matched[0]].text_start != char_start or chars[matched[-1]].text_end != char_end:
        raise TaskError("char span must align to complete English token boundaries")
    return char_start, char_end, matched[0], matched[-1] + 1


def validate_edit(
    *,
    lyrics: str,
    chars: list[LyricChar],
    response: dict[str, Any],
    max_word_len: int,
    language: str,
) -> dict[str, Any]:
    required = {"original_word", "replacement_word"}
    if language == "English":
        if not ({"token_start", "token_end"} <= response.keys() or {"char_start", "char_end"} <= response.keys()):
            raise TaskError("Missing English span keys: provide token_start/token_end or char_start/char_end")
    else:
        required |= {"char_start", "char_end"}
    missing = required - response.keys()
    if missing:
        raise TaskError(f"Missing keys: {sorted(missing)}")

    original_word = str(response["original_word"]).strip()
    replacement_word = str(response["replacement_word"]).strip()

    if language == "English":
        try:
            token_start = int(response["token_start"]) if "token_start" in response else None
            token_end = int(response["token_end"]) if "token_end" in response else None
            char_start = int(response["char_start"]) if "char_start" in response else None
            char_end = int(response["char_end"]) if "char_end" in response else None
        except (TypeError, ValueError) as exc:
            raise TaskError("English span indexes must be integers") from exc

        char_start, char_end, token_start, token_end = find_english_token_span(
            lyrics=lyrics,
            chars=chars,
            original_word=original_word,
            token_start=token_start,
            token_end=token_end,
            char_start=char_start,
            char_end=char_end,
        )
        original_tokens = original_word.split()
        replacement_tokens = replacement_word.split()
        if not (1 <= len(original_tokens) <= max_word_len):
            raise TaskError(f"selected phrase must contain 1 to {max_word_len} English words")
        if len(original_tokens) != len(replacement_tokens):
            raise TaskError("replacement_word must have the same number of English words as original_word")
        if any(not is_english_token(token) for token in original_tokens + replacement_tokens):
            raise TaskError("English words may only contain letters, apostrophes, or hyphens")
        if original_word.lower() == replacement_word.lower():
            raise TaskError("replacement_word is identical to original_word")

        edited_lyrics = lyrics[:char_start] + replacement_word + lyrics[char_end:]
        edit_chars = chars[token_start:token_end]
        return {
            "original_word": original_word,
            "replacement_word": replacement_word,
            "char_start": char_start,
            "char_end": char_end,
            "edited_char_end": char_start + len(replacement_word),
            "token_start": token_start,
            "token_end": token_end,
            "edit_start_sec": round(edit_chars[0].start_sec, 6),
            "edit_end_sec": round(edit_chars[-1].end_sec, 6),
            "edited_lyrics": edited_lyrics,
            "original_pinyin": [],
            "replacement_pinyin": [],
        }

    try:
        char_start = int(response["char_start"])
        char_end = int(response["char_end"])
    except (TypeError, ValueError) as exc:
        raise TaskError("char_start and char_end must be integers") from exc

    if (
        0 <= char_start <= char_end < len(lyrics)
        and lyrics[char_start : char_end + 1] == original_word
    ):
        char_end += 1

    if not (0 <= char_start < char_end <= len(lyrics)):
        raise TaskError("char_start/char_end out of range")
    if char_end - char_start > max_word_len:
        raise TaskError(f"selected word is longer than max_word_len={max_word_len}")
    if not is_chinese_text(original_word) or not is_chinese_text(replacement_word):
        raise TaskError("original_word and replacement_word must be Chinese only")
    if len(original_word) != len(replacement_word):
        raise TaskError("replacement_word must have the same length as original_word")
    if lyrics[char_start:char_end] != original_word:
        raise TaskError(
            f"original_word does not match lyrics span: "
            f"{original_word!r} != {lyrics[char_start:char_end]!r}"
        )
    if original_word == replacement_word:
        raise TaskError("replacement_word is identical to original_word")

    original_pinyin = pinyin_no_tone(original_word)
    replacement_pinyin = pinyin_no_tone(replacement_word)
    if len(original_pinyin) != len(replacement_pinyin):
        raise TaskError("pinyin length mismatch")
    for idx, (src, dst) in enumerate(zip(original_pinyin, replacement_pinyin)):
        if src == dst:
            raise TaskError(
                f"same pinyin at replacement char {idx}: {original_word[idx]}->{replacement_word[idx]}={src}"
            )

    edited_lyrics = lyrics[:char_start] + replacement_word + lyrics[char_end:]
    if edited_lyrics == lyrics:
        raise TaskError("edited lyrics did not change")

    edit_chars = chars[char_start:char_end]
    if not edit_chars:
        raise TaskError("empty edit char span")

    return {
        "original_word": original_word,
        "replacement_word": replacement_word,
        "char_start": char_start,
        "char_end": char_end,
        "edited_char_end": char_start + len(replacement_word),
        "edit_start_sec": round(edit_chars[0].start_sec, 6),
        "edit_end_sec": round(edit_chars[-1].end_sec, 6),
        "edited_lyrics": edited_lyrics,
        "original_pinyin": original_pinyin,
        "replacement_pinyin": replacement_pinyin,
    }

def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise TaskError("LLM response JSON must be an object")
    return obj


def build_prompt(
    *,
    lyrics: str,
    chars: list[LyricChar],
    max_word_len: int,
    previous_error: str | None,
    language: str,
) -> list[dict[str, str]]:
    if language == "English":
        indexed_tokens = " ".join(f"{idx}:{item.char}" for idx, item in enumerate(chars))
        user_content = f"""Design a local lyric replacement task for the following English lyrics.

Original lyrics: {lyrics}
Indexed words: {indexed_tokens}

Requirements:
1. Choose a natural English phrase containing 1 to {max_word_len} complete words.
2. replacement_word must contain the same number of words as original_word.
3. replacement_word should sound clearly different from original_word, but no phoneme validation is required.
4. The edited lyric should still read like a natural lyric line.
5. Do not choose punctuation, silence markers, partial words, or a span that does not exist.
6. Output JSON only, with no explanation.

token_end must be exclusive, equal to token_start + the number of selected words.
For example, if indexed words contain 3:give 4:you, selecting "give you" must return "token_start":3,"token_end":5.

JSON format:
{{"original_word":"old phrase","replacement_word":"new phrase","token_start":0,"token_end":1}}
"""
        system_content = "You are an English lyric editing assistant. Output exactly one strict JSON object."
    else:
        indexed_chars = " ".join(f"{item.char_index}:{item.char}" for item in chars)
        user_content = f"""请为下面的中文歌词设计一个局部歌词替换任务。

原歌词：{lyrics}
带索引字符：{indexed_chars}

要求：
1. 选择一个自然中文词语，长度为 1 到 {max_word_len} 个汉字。
2. replacement_word 必须和 original_word 字数相同。
3. replacement_word 的每个对应汉字都必须和 original_word 对应汉字拼音不同，可以不考虑声调。
4. replacement_word 应该让修改后的歌词仍然像一句自然歌词。
5. 不要选择标点、空白或不存在的跨度。
6. 只输出 JSON，不要解释。

char_end 必须是 exclusive 结束索引，也就是 char_start + original_word 的字数。
例如带索引字符为 3:世 4:界，选择“世界”时必须返回 "char_start":3,"char_end":5。

JSON 格式：
{{"original_word":"原词","replacement_word":"新词","char_start":0,"char_end":1}}
"""
        system_content = "你是中文歌词编辑助手，只能输出一个严格 JSON 对象。"

    if previous_error:
        user_content += f"\n上一次输出不合规，错误是：{previous_error}\n请重新输出。" if language == "Chinese" else f"\nThe previous output was invalid: {previous_error}\nPlease output a corrected JSON object."

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

def call_deepseek(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: float,
    seed: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 256,
        "stream": False,
        "response_format": {"type": "json_object"},
        "seed": seed,
    }
    response = requests.post(
        base_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise TaskError(f"DeepSeek API error {response.status_code}: {response.text[:500]}")
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise TaskError(f"Unexpected DeepSeek response shape: {data}") from exc


def build_success_record(
    aligned_item: dict[str, Any],
    segment: LyricSegment,
    edit: dict[str, Any],
    *,
    total_segments: int,
) -> dict[str, Any]:
    base_id = str(aligned_item["id"])
    task_id = base_id if total_segments == 1 else f"{base_id}_seg{segment.segment_index:03d}"
    return {
        "id": task_id,
        "source_id": base_id,
        "audio_path": aligned_item["audio_path"],
        "metadata_path": aligned_item["metadata_path"],
        "segment_index": segment.segment_index,
        "segment_id": segment.segment_id,
        "segment_start_sec": round(segment.start_sec, 6),
        "segment_end_sec": round(segment.end_sec, 6),
        "segment_duration_sec": round(segment.end_sec - segment.start_sec, 6),
        "language": segment.language,
        "original_lyrics": segment.lyrics,
        "edited_lyrics": edit["edited_lyrics"],
        "original_word": edit["original_word"],
        "replacement_word": edit["replacement_word"],
        "char_start": edit["char_start"],
        "char_end": edit["char_end"],
        "edited_char_end": edit.get("edited_char_end"),
        "token_start": edit.get("token_start"),
        "token_end": edit.get("token_end"),
        "edit_start_sec": edit["edit_start_sec"],
        "edit_end_sec": edit["edit_end_sec"],
        "local_edit_start_sec": round(edit["edit_start_sec"] - segment.start_sec, 6),
        "local_edit_end_sec": round(edit["edit_end_sec"] - segment.start_sec, 6),
        "original_pinyin": edit["original_pinyin"],
        "replacement_pinyin": edit["replacement_pinyin"],
        "status": "success",
    }


def build_task_for_segment(
    *,
    item: dict[str, Any],
    segment: LyricSegment,
    total_segments: int,
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
    max_retries: int,
    max_word_len: int,
    seed: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    previous_error = None
    raw_response = None

    for attempt in range(1, max_retries + 1):
        try:
            messages = build_prompt(
                lyrics=segment.lyrics,
                chars=segment.chars,
                max_word_len=max_word_len,
                previous_error=previous_error,
                language=segment.language,
            )
            raw_response = call_deepseek(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=messages,
                timeout=timeout,
                seed=seed + attempt,
            )
            response = extract_json_object(raw_response)
            edit = validate_edit(
                lyrics=segment.lyrics,
                chars=segment.chars,
                response=response,
                max_word_len=max_word_len,
                language=segment.language,
            )
            return build_success_record(
                item,
                segment,
                edit,
                total_segments=total_segments,
            ), None
        except Exception as exc:
            previous_error = str(exc)
            if attempt < max_retries:
                time.sleep(min(2 * attempt, 5))

    failed = {
        "id": f"{item.get('id')}_seg{segment.segment_index:03d}",
        "source_id": item.get("id"),
        "audio_path": item.get("audio_path"),
        "metadata_path": item.get("metadata_path"),
        "segment_index": segment.segment_index,
        "segment_id": segment.segment_id,
        "segment_start_sec": round(segment.start_sec, 6),
        "segment_end_sec": round(segment.end_sec, 6),
        "original_lyrics": segment.lyrics,
        "status": "failed",
        "error": previous_error,
        "raw_response": raw_response,
    }
    return None, failed


def failed_output_path(output_path: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    return output_path.with_name(f"{output_path.stem}.failed{output_path.suffix}")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    if args.output.exists() and not args.overwrite and not args.dry_run:
        print(f"Output already exists: {args.output}. Pass --overwrite to replace.", file=sys.stderr)
        return 2

    items = load_jsonl(args.input_manifest)
    if args.limit is not None:
        items = items[: args.limit]

    if args.dry_run:
        for item in items:
            segments = parse_soulx_metadata_segments(
                Path(str(item["metadata_path"])),
                language=args.language,
                aligned_item=item,
            )
            for segment in segments:
                print(
                    json.dumps(
                        {
                            "id": item["id"],
                            "segment_index": segment.segment_index,
                            "segment_id": segment.segment_id,
                            "segment_start_sec": round(segment.start_sec, 3),
                            "segment_end_sec": round(segment.end_sec, 3),
                            "language": segment.language,
                            "lyrics": segment.lyrics,
                            "n_units": len(segment.chars),
                            "first_units": [
                                {
                                    "char": c.char,
                                    "start_sec": round(c.start_sec, 3),
                                    "end_sec": round(c.end_sec, 3),
                                }
                                for c in segment.chars[:8]
                            ],
                        },
                        ensure_ascii=False,
                    )
                )
        return 0

    api_key = load_env_key(args.env_file)
    failed_path = failed_output_path(args.output, args.failed_output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failed_count = 0
    with args.output.open("w", encoding="utf-8") as success_handle, failed_path.open(
        "w", encoding="utf-8"
    ) as failed_handle:
        for idx, item in enumerate(items, start=1):
            print(f"[{idx}/{len(items)}] {item.get('id')}")
            try:
                segments = parse_soulx_metadata_segments(
                    Path(str(item["metadata_path"])),
                    language=args.language,
                    aligned_item=item,
                )
            except Exception as exc:
                failed = {
                    "id": item.get("id"),
                    "audio_path": item.get("audio_path"),
                    "metadata_path": item.get("metadata_path"),
                    "status": "failed",
                    "error": str(exc),
                }
                failed_handle.write(json.dumps(failed, ensure_ascii=False) + "\n")
                failed_handle.flush()
                failed_count += 1
                print(f"  failed: {failed['error']}")
                continue

            for seg_pos, segment in enumerate(segments, start=1):
                print(
                    f"  segment {seg_pos}/{len(segments)} "
                    f"@ {segment.start_sec:.3f}-{segment.end_sec:.3f}s"
                )
                success, failed = build_task_for_segment(
                    item=item,
                    segment=segment,
                    total_segments=len(segments),
                    api_key=api_key,
                    base_url=args.base_url,
                    model=args.model,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                    max_word_len=args.max_word_len,
                    seed=args.seed + idx * 1000 + segment.segment_index * 100,
                )
                if success is not None:
                    success_handle.write(json.dumps(success, ensure_ascii=False) + "\n")
                    success_handle.flush()
                    success_count += 1
                    print(
                        f"    ok: {success['original_word']} -> {success['replacement_word']} "
                        f"@ {success['edit_start_sec']:.3f}-{success['edit_end_sec']:.3f}s"
                    )
                if failed is not None:
                    failed_handle.write(json.dumps(failed, ensure_ascii=False) + "\n")
                    failed_handle.flush()
                    failed_count += 1
                    print(f"    failed: {failed['error']}")

    print(f"Wrote {success_count} success task(s) to {args.output}")
    print(f"Wrote {failed_count} failed task(s) to {failed_path}")
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
