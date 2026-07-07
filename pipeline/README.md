# VocalEdit Pipeline 使用文档

本文档说明如何搭建并运行当前 VocalEdit 工作流：

1. 从数据集抽取一个小音频集。
2. 用 SoulX-Singer 做 ASR 转写和歌词时间对齐。
3. 从 SoulX 标注中生成歌词修改任务。
4. 用修改后的 YingMusic-Singer-Plus 推理脚本做局部歌词编辑。
5. 用 Streamlit 可视化逐条检查结果。

当前主线只关注中文数据。英文数据暂时不进入后续流程。

## 目录约定

从仓库根目录运行命令：

```text
VocalEdit/
  SoulX-Singer/              # SoulX-Singer，负责预处理、ASR、对齐
  YingMusic-Singer-Plus/     # YingMusic-Singer-Plus，负责推理
  data/                      # 小数据集输出位置
  pipeline/
    configs/                 # 配置示例
    manifests/               # 各阶段 JSONL 清单
    runs/                    # 各阶段运行结果、日志、生成音频
    scripts/                 # 本工作流 CLI 脚本
```

推荐所有中间结果都写到 `pipeline/manifests` 和 `pipeline/runs`，不要直接改原始数据集。

## 环境约定

本工作流使用两个 Conda 环境：

- `align`：运行 SoulX-Singer 预处理和对齐。
- `ymsp`：运行 YingMusic-Singer-Plus 推理和 Streamlit 可视化。

DeepSeek API key 放在：

```text
pipeline/.env
```

文件内容格式：

```bash
deepseek_api_key=你的_api_key
```

国内访问 HuggingFace 时，可以在推理命令前加：

```bash
HF_ENDPOINT=https://hf-mirror.com
```

## 当前已经跑通的默认输入输出

已生成的小数据和结果路径如下，后续命令默认围绕这些路径：

```text
data/gtsinger_mini_40/Chinese/*.wav
pipeline/manifests/01_aligned.gtsinger_mini_40_chinese.jsonl
pipeline/manifests/02_lyric_edit_tasks.gtsinger_mini_40_chinese.jsonl
pipeline/runs/yingmusic_lyric_edit_gtsinger_mini_40_chinese_hardmask/
```

如果只是查看现有结果，可以直接跳到“步骤 6：可视化检查”。

## 步骤 1：从 GTSinger 抽取小数据集

脚本：

```text
pipeline/scripts/create_gtsinger_mini_dataset.py
```

用途：从 GTSinger 的 `Chinese` 和 `English` 目录中随机抽取 wav，复制成一个独立的小数据集。抽样时会跳过任何路径中包含 `Paired_Speech_Group` 的 wav，因为该目录保存朗读音频而不是歌声音频。当前后续只用中文目录。

示例命令：

```bash
python pipeline/scripts/create_gtsinger_mini_dataset.py \
  --gtsinger-root /inspire/hdd/project/embodied-multimodality/chenxie-25019/shuyiwang/data/GTSinger \
  --output-root data/gtsinger_mini_40 \
  --samples-per-language 20 \
  --seed 20260706 \
  --overwrite
```

输出：

```text
data/gtsinger_mini_40/Chinese/*.wav
data/gtsinger_mini_40/English/*.wav
data/gtsinger_mini_40/manifest.jsonl
data/gtsinger_mini_40/summary.json
```

常用参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--gtsinger-root` | 原始 GTSinger 数据集根目录 | 必填 |
| `--output-root` | 小数据集输出目录 | 必填 |
| `--samples-per-language` | 每种语言抽取多少条 wav | `20` |
| `--seed` | 随机种子，保证可复现 | `20260706` |
| `--overwrite` | 允许写入已有输出目录 | 不开启 |

## 步骤 2：用 SoulX-Singer 批量对齐中文音频

脚本：

```text
pipeline/scripts/run_soulx_alignment_batch.py
```

用途：递归读取 `--audio-root` 下的 wav，逐条调用 SoulX-Singer 的 `preprocess.pipeline`，得到 ASR 歌词、音符/字级时间、F0、切片等预处理结果。

示例命令：

```bash
python pipeline/scripts/run_soulx_alignment_batch.py \
  --audio-root data/gtsinger_mini_40/Chinese \
  --soulx-root SoulX-Singer \
  --output-root pipeline/runs/soulx_align_gtsinger_mini_40_chinese \
  --conda-env align \
  --language Mandarin \
  --device cuda \
  --max-merge-duration 30000
```

输出：

```text
pipeline/runs/soulx_align_gtsinger_mini_40_chinese/alignment_results.jsonl
pipeline/runs/soulx_align_gtsinger_mini_40_chinese/items/<task_id>/metadata.json
pipeline/runs/soulx_align_gtsinger_mini_40_chinese/items/<task_id>/vocal.wav
pipeline/runs/soulx_align_gtsinger_mini_40_chinese/logs/<task_id>.log
```

`alignment_results.jsonl` 每行包含：

```json
{
  "id": "chinese_001_xxx",
  "audio_path": ".../chinese_001_xxx.wav",
  "metadata_path": ".../metadata.json",
  "log_path": ".../logs/chinese_001_xxx.log",
  "status": "success"
}
```

为了让后续脚本使用默认路径，可以复制一份为阶段 manifest：

```bash
cp pipeline/runs/soulx_align_gtsinger_mini_40_chinese/alignment_results.jsonl \
  pipeline/manifests/01_aligned.gtsinger_mini_40_chinese.jsonl
```

常用参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--audio-root` | 待对齐 wav 根目录 | 必填 |
| `--soulx-root` | SoulX-Singer 项目根目录 | 必填 |
| `--output-root` | 对齐结果输出目录 | 必填 |
| `--conda-env` | SoulX 对齐环境 | `align` |
| `--language` | SoulX 语言参数，中文用 `Mandarin` | `Mandarin` |
| `--device` | 推理设备 | `cuda` |
| `--max-merge-duration` | SoulX 最大合并时长参数，单位按 SoulX 脚本定义 | `30000` |
| `--resume` | 跳过已经成功的条目，继续未完成任务 | 不开启 |

注意：SoulX 的 `metadata.json` 中 `note_type` 有三类：

- `1`：静音或空白，如 `<SP>`。
- `2`：正常歌词字。
- `3`：前一个歌词字的延长或重复发声。

后续歌词任务脚本会自动忽略 `1`，把 `2` 作为歌词字，并把 `3` 的时长合并到前一个字上。

## 步骤 3：生成歌词修改任务

脚本：

```text
pipeline/scripts/create_lyric_edit_tasks.py
```

用途：读取 SoulX 对齐结果中的 `metadata.json`，还原 ASR 歌词和每个中文字的时间戳，然后调用 DeepSeek 选择一个词并替换为同字数、不同拼音的词。

示例命令：

```bash
python pipeline/scripts/create_lyric_edit_tasks.py \
  --input-manifest pipeline/manifests/01_aligned.gtsinger_mini_40_chinese.jsonl \
  --output pipeline/manifests/02_lyric_edit_tasks.gtsinger_mini_40_chinese.jsonl \
  --env-file pipeline/.env \
  --model deepseek-chat \
  --overwrite
```

如果只想检查 SoulX metadata 是否能解析，不调用 LLM：

```bash
python pipeline/scripts/create_lyric_edit_tasks.py \
  --input-manifest pipeline/manifests/01_aligned.gtsinger_mini_40_chinese.jsonl \
  --dry-run
```

输出成功任务：

```text
pipeline/manifests/02_lyric_edit_tasks.gtsinger_mini_40_chinese.jsonl
```

失败任务默认输出到：

```text
pipeline/manifests/02_lyric_edit_tasks.gtsinger_mini_40_chinese.failed.jsonl
```

成功任务示例：

```json
{
  "id": "chinese_001_fb61ea373d",
  "audio_path": ".../chinese_001_fb61ea373d.wav",
  "metadata_path": ".../metadata.json",
  "original_lyrics": "朦胧的世界我们留了多远",
  "edited_lyrics": "明亮的世界我们留了多远",
  "original_word": "朦胧",
  "replacement_word": "明亮",
  "char_start": 0,
  "char_end": 2,
  "edit_start_sec": 0.15,
  "edit_end_sec": 1.23,
  "status": "success"
}
```

常用参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--input-manifest` | SoulX 对齐结果 JSONL，需要包含 `metadata_path` | `pipeline/manifests/01_aligned.gtsinger_mini_40_chinese.jsonl` |
| `--output` | 成功任务输出 JSONL | `pipeline/manifests/02_lyric_edit_tasks.gtsinger_mini_40_chinese.jsonl` |
| `--failed-output` | 失败任务输出 JSONL | `<output stem>.failed.jsonl` |
| `--env-file` | 包含 `deepseek_api_key` 的 env 文件 | `pipeline/.env` |
| `--model` | DeepSeek 模型名 | `deepseek-v4-flash` |
| `--base-url` | DeepSeek Chat Completions API 地址 | `https://api.deepseek.com/chat/completions` |
| `--max-retries` | LLM 返回不合法时的最大重试次数 | `3` |
| `--max-word-len` | 允许修改的最长词长 | `4` |
| `--timeout` | 单次 API 请求超时秒数 | `60.0` |
| `--seed` | 随机种子 | `20260706` |
| `--limit` | 只处理前 N 条，用于调试 | 不限制 |
| `--dry-run` | 只解析 metadata，不调用 LLM | 不开启 |
| `--overwrite` | 覆盖已有输出文件 | 不开启 |

任务生成约束：

- 只处理中文歌词。
- 替换词与原词字数相同。
- 替换后歌词长度不变。
- 每个对应中文字的无声调拼音必须不同。
- 修改片段时间戳来自 SoulX 字级对齐结果，并正确处理 `note_type=3` 的延长时长。

## 步骤 4：用 YingMusic-Singer-Plus 做局部歌词编辑推理

脚本：

```text
pipeline/scripts/run_yingmusic_lyric_edit_tasks.py
```

用途：读取步骤 3 的歌词修改任务，逐条调用修改后的 YingMusic-Singer-Plus 推理脚本，生成编辑后的音频。

当前推理逻辑已经做过以下修改：

- 不需要单独指定音色参考音频，脚本会复用原始 melody audio 作为音色来源。
- 音频条件前半段是音色 prompt，后半段是原始 melody audio latent。
- 被修改词对应的时间区间做 hard mask。默认不额外加 overlap 或 transition。
- 可以通过 `--mask-start-offset-sec` 和 `--mask-end-offset-sec` 只扩大 mask 时间窗。例如起点提前 0.2 秒、终点推迟 0.3 秒，会让被生成区域增加 0.5 秒。
- 采样完成后，未被 mask 的 latent 会被替换回原始 latent，只保留 masked 区间的模型生成结果。
- 歌词条件和 melody/MIDI 条件保持原模型流程。

示例命令：

```bash
HF_ENDPOINT=https://hf-mirror.com python pipeline/scripts/run_yingmusic_lyric_edit_tasks.py \
  --task-manifest pipeline/manifests/02_lyric_edit_tasks.gtsinger_mini_40_chinese.jsonl \
  --yingmusic-root YingMusic-Singer-Plus \
  --output-dir pipeline/runs/yingmusic_lyric_edit_gtsinger_mini_40_chinese_hardmask \
  --conda-env ymsp \
  --ckpt-path ASLP-lab/YingMusic-Singer-Plus \
  --device cuda:0 \
  --mask-start-offset-sec 0.2 \
  --mask-end-offset-sec 0.3 \
  --overwrite
```

如果权重已经下载在本地，也可以传本地路径：

```bash
python pipeline/scripts/run_yingmusic_lyric_edit_tasks.py \
  --ckpt-path ../YingMusic-Singer-Plus \
  --output-dir pipeline/runs/yingmusic_lyric_edit_gtsinger_mini_40_chinese_hardmask \
  --overwrite
```

只跑一条做 smoke test：

```bash
HF_ENDPOINT=https://hf-mirror.com python pipeline/scripts/run_yingmusic_lyric_edit_tasks.py \
  --limit 1 \
  --output-dir pipeline/runs/yingmusic_lyric_edit_smoke \
  --overwrite
```

输出：

```text
pipeline/runs/yingmusic_lyric_edit_gtsinger_mini_40_chinese_hardmask/inference_results.jsonl
pipeline/runs/yingmusic_lyric_edit_gtsinger_mini_40_chinese_hardmask/wavs/<task_id>.wav
pipeline/runs/yingmusic_lyric_edit_gtsinger_mini_40_chinese_hardmask/logs/<task_id>.log
```

常用参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--task-manifest` | 歌词修改任务 JSONL | `pipeline/manifests/02_lyric_edit_tasks.gtsinger_mini_40_chinese.jsonl` |
| `--yingmusic-root` | YingMusic-Singer-Plus 项目根目录 | `YingMusic-Singer-Plus` |
| `--output-dir` | 推理输出目录 | `pipeline/runs/yingmusic_lyric_edit_gtsinger_mini_40_chinese` |
| `--conda-env` | YingMusic 推理环境；脚本会自动用 `conda run` 重进该环境 | `ymsp` |
| `--ckpt-path` | HuggingFace repo id 或本地权重目录 | `ASLP-lab/YingMusic-Singer-Plus` |
| `--device` | 推理设备 | `cuda:0` |
| `--nfe-step` | 扩散/ODE 采样步数 | `32` |
| `--cfg-strength` | CFG 强度 | `3.0` |
| `--t-shift` | 采样时间偏移参数 | `0.5` |
| `--sil-len-to-end` | 音色 prompt 后拼接的静音秒数 | `0.5` |
| `--mask-start-offset-sec` | 将 mask 起点向前扩张的秒数；必须非负，只会增加 mask 区域 | `0.0` |
| `--mask-end-offset-sec` | 将 mask 终点向后扩张的秒数；必须非负，只会增加 mask 区域 | `0.0` |
| `--seed` | 随机种子；每条任务会使用 `seed + idx` | `20260706` |
| `--limit` | 只跑前 N 条，用于调试 | 不限制 |
| `--overwrite` | 已存在 wav 时重新生成 | 不开启 |
| `--verbose` | 失败时写完整 traceback 到 log | 不开启 |
| `--no-conda-reexec` | 禁止脚本自动重进 Conda 环境 | 不开启 |

## 步骤 5：检查推理结果

统计成功数：

```bash
python - <<'EOF'
import json
from pathlib import Path
p = Path('pipeline/runs/yingmusic_lyric_edit_gtsinger_mini_40_chinese_hardmask/inference_results.jsonl')
rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
print('rows', len(rows))
print('success', sum(r.get('status') == 'success' for r in rows))
print('failed', [r['id'] for r in rows if r.get('status') != 'success'])
EOF
```

统计生成 wav 数量：

```bash
find pipeline/runs/yingmusic_lyric_edit_gtsinger_mini_40_chinese_hardmask/wavs \
  -name '*.wav' | wc -l
```

查看失败日志：

```bash
ls pipeline/runs/yingmusic_lyric_edit_gtsinger_mini_40_chinese_hardmask/logs
```

如果某条失败，建议先用 `--limit 1 --verbose` 或者单独缩小 task manifest 来复现。

## 步骤 6：可视化检查

脚本：

```text
pipeline/scripts/view_lyric_edit_tasks.py
```

用途：用 Streamlit 逐 task 展示：

- 原歌词。
- 修改后歌词。
- 修改词和替换词。
- 修改起止时间。
- 类 DAW 的双音轨 waveform，分别展示修改前和修改后音频。
- 红色竖线播放头穿过两条音轨，播放时随时间移动。
- 点击音轨或按钮选择当前播放音轨，每次只播放选中的一条。
- 按空格键播放/暂停，点击或拖动播放头区域可以定位。
- 编辑区高亮显示修改时间范围。

启动命令：

```bash
conda run -n ymsp streamlit run pipeline/scripts/view_lyric_edit_tasks.py \
  --server.headless true \
  --server.address 0.0.0.0 \
  --server.port 8501
```

浏览器访问：

```text
http://127.0.0.1:8501
```

默认读取：

```text
pipeline/manifests/02_lyric_edit_tasks.gtsinger_mini_40_chinese.jsonl
pipeline/runs/yingmusic_lyric_edit_gtsinger_mini_40_chinese_hardmask/inference_results.jsonl
```

如果要查看别的任务文件或推理结果，可以在页面左侧 sidebar 里修改 `Task manifest` 和 `Result manifest` 路径。

## 通用音频扫描脚本

脚本：

```text
pipeline/scripts/discover_audio.py
```

用途：递归扫描任意数据集目录，生成音频清单。这个脚本适合扩展到非 GTSinger 数据集时使用。

示例命令：

```bash
python pipeline/scripts/discover_audio.py \
  --dataset-root /path/to/dataset \
  --output pipeline/manifests/00_discovered.jsonl
```

包含更多音频格式：

```bash
python pipeline/scripts/discover_audio.py \
  --dataset-root /path/to/dataset \
  --output pipeline/manifests/00_discovered.jsonl \
  --extensions .wav .flac .mp3 .m4a .ogg \
  --include-hidden \
  --follow-symlinks
```

参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--dataset-root` | 要递归扫描的数据集根目录 | 必填 |
| `--output` | 输出 JSONL manifest | 必填 |
| `--extensions` | 要包含的音频扩展名 | 常见音频格式 |
| `--include-hidden` | 包含隐藏文件和隐藏目录 | 不开启 |
| `--follow-symlinks` | 递归扫描符号链接目录 | 不开启 |

## 推荐的一键顺序

从 GTSinger 抽样、对齐、生成任务、推理、可视化，按下面顺序执行：

```bash
python pipeline/scripts/create_gtsinger_mini_dataset.py \
  --gtsinger-root /inspire/hdd/project/embodied-multimodality/chenxie-25019/shuyiwang/data/GTSinger \
  --output-root data/gtsinger_mini_40 \
  --samples-per-language 20 \
  --seed 20260706 \
  --overwrite

python pipeline/scripts/run_soulx_alignment_batch.py \
  --audio-root data/gtsinger_mini_40/Chinese \
  --soulx-root SoulX-Singer \
  --output-root pipeline/runs/soulx_align_gtsinger_mini_40_chinese \
  --conda-env align \
  --language Mandarin \
  --device cuda \
  --max-merge-duration 30000

cp pipeline/runs/soulx_align_gtsinger_mini_40_chinese/alignment_results.jsonl \
  pipeline/manifests/01_aligned.gtsinger_mini_40_chinese.jsonl

python pipeline/scripts/create_lyric_edit_tasks.py \
  --input-manifest pipeline/manifests/01_aligned.gtsinger_mini_40_chinese.jsonl \
  --output pipeline/manifests/02_lyric_edit_tasks.gtsinger_mini_40_chinese.jsonl \
  --env-file pipeline/.env \
  --model deepseek-chat \
  --overwrite

HF_ENDPOINT=https://hf-mirror.com python pipeline/scripts/run_yingmusic_lyric_edit_tasks.py \
  --task-manifest pipeline/manifests/02_lyric_edit_tasks.gtsinger_mini_40_chinese.jsonl \
  --yingmusic-root YingMusic-Singer-Plus \
  --output-dir pipeline/runs/yingmusic_lyric_edit_gtsinger_mini_40_chinese_hardmask \
  --conda-env ymsp \
  --ckpt-path ASLP-lab/YingMusic-Singer-Plus \
  --device cuda:0 \
  --mask-start-offset-sec 0.2 \
  --mask-end-offset-sec 0.3 \
  --overwrite

conda run -n ymsp streamlit run pipeline/scripts/view_lyric_edit_tasks.py \
  --server.headless true \
  --server.address 0.0.0.0 \
  --server.port 8501
```

## 常见问题

### DeepSeek API 报错

先检查 `pipeline/.env` 是否存在，且 key 名必须是：

```bash
deepseek_api_key=...
```

可以先用 `--dry-run` 验证 SoulX metadata 是否正常，排除对齐结果问题。

### HuggingFace 权重下载慢或失败

在推理命令前加：

```bash
HF_ENDPOINT=https://hf-mirror.com
```

如果机器上已经有本地权重，直接用 `--ckpt-path /path/to/local/ckpt`。

### SoulX 对齐中断

重新运行时加 `--resume`：

```bash
python pipeline/scripts/run_soulx_alignment_batch.py \
  --audio-root data/gtsinger_mini_40/Chinese \
  --soulx-root SoulX-Singer \
  --output-root pipeline/runs/soulx_align_gtsinger_mini_40_chinese \
  --resume
```

### Streamlit 页面没有显示修改音频

确认 `Result manifest` 指向的是推理输出目录下的 `inference_results.jsonl`，且其中每条记录的 `output_path` 文件存在。

### 只想快速验证一条

歌词任务生成和 YingMusic 推理都支持 `--limit 1`：

```bash
python pipeline/scripts/create_lyric_edit_tasks.py --limit 1 --overwrite
HF_ENDPOINT=https://hf-mirror.com python pipeline/scripts/run_yingmusic_lyric_edit_tasks.py --limit 1 --overwrite
```
