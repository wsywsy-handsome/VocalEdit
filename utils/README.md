# Utils

## `extract_soulx_lyrics.py`

从 SoulX 对齐 metadata 中还原歌词。脚本只做解析和导出，不调用 LLM，也不依赖 pipeline 里的任务生成逻辑。

### 支持的输入

JSON 文件：

- 顶层是一个 metadata 对象，并包含 `soulx_segments` 字段。
- 也兼容顶层直接是 SoulX segment 列表。

JSONL 文件：

- 每一行都是一个完整 metadata JSON 对象，格式和 JSON 文件输入一致。
- 也就是说，每行都应该长得像 `test.json`：包含 `soulx_segments`，而不是 manifest 或 `metadata_path` 引用。

JSON 输入样例：

```json
{
  "soulx_segments": [
    {
      "time": [26600, 39320],
      "text": "<SP> 我 要 穿 越 这 片 沙 漠 <SP>",
      "duration": "0.05 0.10 0.16 0.14 0.40 0.20 0.28 0.26 0.24 0.36",
      "note_type": "1 2 2 2 2 2 2 2 2 1"
    }
  ],
  "align_status": "success"
}
```

JSONL 输入样例：

```jsonl
{"soulx_segments":[{"time":[26600,39320],"text":"<SP> 我 要 穿 越 这 片 沙 漠 <SP>","duration":"0.05 0.10 0.16 0.14 0.40 0.20 0.28 0.26 0.24 0.36","note_type":"1 2 2 2 2 2 2 2 2 1"}],"align_status":"success"}
{"soulx_segments":[{"time":[39320,52040],"text":"<SP> 突 然 之 间 出 现 <SP>","duration":"0.33 0.12 0.22 0.20 0.40 0.38 0.60 0.48","note_type":"1 2 2 2 2 2 2 1"}],"align_status":"success"}
```

SoulX segment 需要包含这些字段：

- `time`：毫秒级 `[start, end]`
- `text`：空格分隔 token
- `duration`：空格分隔秒级 token 时长
- `note_type`：空格分隔类型

`note_type` 解析规则和 `pipeline/scripts/create_lyric_edit_tasks.py` 保持一致：

- `1`：静音或 `<SP>/<AP>`，跳过
- `2`：新歌词 token，中文按字加入，英文按 word 加入
- `3`：延长前一个歌词 token 的结束时间，不新增歌词

中文 metadata 中如果混有英文 word，英文 word 会被保留，不会被丢弃。输出文本会在中英文边界和连续英文 word 之间插入空格，例如 `我 love you 你`。

### 输出模式

`lines` 模式：

每个解析出的 segment 输出一行歌词，只包含歌词文本。
当输入是 JSONL 且包含多条 metadata 时，不同 metadata 的歌词之间会额外插入一个空行。

```bash
python3 utils/extract_soulx_lyrics.py \
  --input test.json \
  --output outputs/lyrics_lines.txt \
  --mode lines \
  --overwrite
```

`json` 模式：

输出解析信息，包括整首合并歌词、每段歌词、segment 时间，以及每个歌词单元的文本和起止时间。

```bash
python3 utils/extract_soulx_lyrics.py \
  --input test.json \
  --output outputs/lyrics_full.json \
  --mode json \
  --overwrite
```

### JSONL 示例命令

如果 `metadata_batch.jsonl` 中每一行都是一个包含 `soulx_segments` 的 metadata 对象，运行：

```bash
python3 utils/extract_soulx_lyrics.py \
  --input metadata_batch.jsonl \
  --output outputs/lyrics_lines.txt \
  --mode lines \
  --overwrite
```

### 常用参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--input, -i` | 输入 `.json` 或 `.jsonl` | 必填 |
| `--output, -o` | 输出文件路径 | 必填 |
| `--mode` | `lines` 或 `json` | `lines` |
| `--language` | `auto`、`Chinese`、`English` | `auto` |
| `--overwrite` | 覆盖已有输出文件 | 不开启 |

### 输出 JSON 结构

顶层结构示例：

```json
{
  "input": "test.json",
  "n_records": 1,
  "records": []
}
```

顶层字段：

| 字段 | 说明 |
| --- | --- |
| `input` | 本次解析的输入文件路径。 |
| `n_records` | 输入中解析出的 metadata 条数；单个 JSON 通常为 `1`，JSONL 等于有效行数。 |
| `records` | 每条 metadata 的解析结果列表。 |

每个 `record` 字段：

| 字段 | 说明 |
| --- | --- |
| `input_line` | 来源 JSONL 行号，从 `1` 开始；单个 JSON 输入时为 `null`。 |
| `language` | 解析使用的语言，`Chinese` 或 `English`。 |
| `lyrics` | 当前 metadata 下所有 segment 合并后的完整歌词。 |
| `n_units` | 当前 metadata 下解析出的歌词单元总数；中文为汉字数，英文为 word 数；中文混英文时英文 word 也计为一个 unit。 |
| `segments` | 当前 metadata 下每个 SoulX segment 的解析结果。 |

每个 `segment` 字段：

| 字段 | 说明 |
| --- | --- |
| `segment_index` | segment 在原始 `soulx_segments` 中的下标，从 `0` 开始。 |
| `segment_id` | 原始 segment 的 `index` 字段；没有时生成 `segment_000` 这类 id。 |
| `start_sec` | segment 起始时间，单位秒，由原始 `time[0]` 毫秒转换而来。 |
| `end_sec` | segment 结束时间，单位秒，由原始 `time[1]` 毫秒转换而来。 |
| `duration_sec` | segment 时长，单位秒，等于 `end_sec - start_sec`。 |
| `language` | 当前 segment 的解析语言。 |
| `lyrics` | 当前 segment 解析出的歌词文本。 |
| `n_units` | 当前 segment 中解析出的歌词单元数量。 |
| `units` | 当前 segment 中每个歌词单元的文本和时间信息。 |

每个 `unit` 字段：

| 字段 | 说明 |
| --- | --- |
| `char` | 歌词单元文本；中文为单个汉字，英文或中文混英文时可为一个英文 word。 |
| `start_sec` | 该歌词单元起始时间，单位秒。 |
| `end_sec` | 该歌词单元结束时间，单位秒；遇到 `note_type=3` 时会延长到重复发声结束。 |
