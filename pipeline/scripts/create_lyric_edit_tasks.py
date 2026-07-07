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


@dataclass
class LyricChar:
    char: str
    char_index: int
    start_sec: float
    end_sec: float
    source_segment: int
    source_token_index: int
    note_type: int


class TaskError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Chinese lyric edit tasks from SoulX metadata."
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


def normalize_soulx_token(token: str) -> str:
    return token.replace("<AP>", "<SP>")


def parse_soulx_metadata(metadata_path: Path) -> tuple[str, list[LyricChar]]:
    segments = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(segments, list) or not segments:
        raise TaskError(f"SoulX metadata has no segments: {metadata_path}")

    chars: list[LyricChar] = []
    for seg_idx, seg in enumerate(segments):
        try:
            seg_start_sec = float(seg["time"][0]) / 1000.0
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

        cursor = seg_start_sec
        for token_idx, (word, dur, note_type) in enumerate(zip(words, durs, types)):
            start_sec = cursor
            end_sec = cursor + float(dur)
            cursor = end_sec

            if word == "<SP>" or note_type == 1:
                continue
            if note_type == 2:
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
                continue
            if note_type == 3:
                if chars:
                    chars[-1].end_sec = end_sec
                continue

            raise TaskError(f"Unsupported note_type={note_type} in {metadata_path}")

    lyrics = "".join(item.char for item in chars)
    if not lyrics:
        raise TaskError(f"No Chinese lyrics found in {metadata_path}")
    return lyrics, chars


def pinyin_no_tone(text: str) -> list[str]:
    return lazy_pinyin(text, style=Style.NORMAL, errors="default")


def validate_edit(
    *,
    lyrics: str,
    chars: list[LyricChar],
    response: dict[str, Any],
    max_word_len: int,
) -> dict[str, Any]:
    required = {"original_word", "replacement_word", "char_start", "char_end"}
    missing = required - response.keys()
    if missing:
        raise TaskError(f"Missing keys: {sorted(missing)}")

    original_word = str(response["original_word"]).strip()
    replacement_word = str(response["replacement_word"]).strip()
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
) -> list[dict[str, str]]:
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
    if previous_error:
        user_content += f"\n上一次输出不合规，错误是：{previous_error}\n请重新输出。"

    return [
        {
            "role": "system",
            "content": "你是中文歌词编辑助手，只能输出一个严格 JSON 对象。",
        },
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
    lyrics: str,
    edit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": aligned_item["id"],
        "audio_path": aligned_item["audio_path"],
        "metadata_path": aligned_item["metadata_path"],
        "original_lyrics": lyrics,
        "edited_lyrics": edit["edited_lyrics"],
        "original_word": edit["original_word"],
        "replacement_word": edit["replacement_word"],
        "char_start": edit["char_start"],
        "char_end": edit["char_end"],
        "edit_start_sec": edit["edit_start_sec"],
        "edit_end_sec": edit["edit_end_sec"],
        "original_pinyin": edit["original_pinyin"],
        "replacement_pinyin": edit["replacement_pinyin"],
        "status": "success",
    }


def build_task_for_item(
    *,
    item: dict[str, Any],
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
    max_retries: int,
    max_word_len: int,
    seed: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    metadata_path = Path(str(item["metadata_path"]))
    lyrics, chars = parse_soulx_metadata(metadata_path)
    previous_error = None
    raw_response = None

    for attempt in range(1, max_retries + 1):
        try:
            messages = build_prompt(
                lyrics=lyrics,
                chars=chars,
                max_word_len=max_word_len,
                previous_error=previous_error,
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
                lyrics=lyrics,
                chars=chars,
                response=response,
                max_word_len=max_word_len,
            )
            return build_success_record(item, lyrics, edit), None
        except Exception as exc:
            previous_error = str(exc)
            if attempt < max_retries:
                time.sleep(min(2 * attempt, 5))

    failed = {
        "id": item.get("id"),
        "audio_path": item.get("audio_path"),
        "metadata_path": item.get("metadata_path"),
        "original_lyrics": lyrics,
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
            lyrics, chars = parse_soulx_metadata(Path(str(item["metadata_path"])))
            print(
                json.dumps(
                    {
                        "id": item["id"],
                        "lyrics": lyrics,
                        "n_chars": len(chars),
                        "first_chars": [
                            {
                                "char": c.char,
                                "start_sec": round(c.start_sec, 3),
                                "end_sec": round(c.end_sec, 3),
                            }
                            for c in chars[:8]
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
            success, failed = build_task_for_item(
                item=item,
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                timeout=args.timeout,
                max_retries=args.max_retries,
                max_word_len=args.max_word_len,
                seed=args.seed + idx * 1000,
            )
            if success is not None:
                success_handle.write(json.dumps(success, ensure_ascii=False) + "\n")
                success_handle.flush()
                success_count += 1
                print(
                    f"  ok: {success['original_word']} -> {success['replacement_word']} "
                    f"@ {success['edit_start_sec']:.3f}-{success['edit_end_sec']:.3f}s"
                )
            if failed is not None:
                failed_handle.write(json.dumps(failed, ensure_ascii=False) + "\n")
                failed_handle.flush()
                failed_count += 1
                print(f"  failed: {failed['error']}")

    print(f"Wrote {success_count} success task(s) to {args.output}")
    print(f"Wrote {failed_count} failed task(s) to {failed_path}")
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
