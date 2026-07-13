# VocalEdit Pipeline 使用文档

本文档描述当前 VocalEdit 工作流：从音频发现/抽样，到 SoulX-Singer 对齐，再到 LLM 创建歌词修改任务，最后用 YingMusic-Singer-Plus 做局部歌词编辑并用 Streamlit 检查结果。

当前 pipeline 支持中文和英文任务。中文任务按字处理，并校验替换字拼音不同；英文任务按 word token 处理，不做英文音素校验。

## 目录约定

所有命令默认从仓库根目录运行：

```text
VocalEdit/
  SoulX-Singer/              # 上游 SoulX-Singer：预处理、ASR、对齐、SoulX 对照实验
  YingMusic-Singer-Plus/     # 上游 YingMusic-Singer-Plus：主推理模型
  data/                      # 可选的小数据集/扫描 manifest 输出
  patches/                   # 对上游项目的补丁
  pipeline/
    manifests/               # 阶段 JSONL 清单
    runs/                    # 阶段运行结果、日志、生成音频
    scripts/                 # 本项目 pipeline CLI 脚本
    assets/                  # 技术报告图片
    README.md
    requirements.txt         # pipeline 脚本的轻量依赖
```

推荐所有中间结果写入 `pipeline/manifests` 和 `pipeline/runs`，不要直接改原始数据集。

## 环境约定

本项目实际使用三类环境。不要把所有依赖装进一个环境，SoulX 和 YingMusic 的 PyTorch/NeMo 依赖容易冲突。

| 环境 | 用途 | 典型命令入口 |
| --- | --- | --- |
| `pipeline` 或 base | 运行轻量脚本：扫描音频、抽样、创建任务、管理 JSONL | `pipeline/scripts/*.py` 中不直接加载大模型的脚本 |
| `align` | SoulX 中文对齐；也可跑 SoulX 对照实验 | `run_soulx_alignment_batch.py --conda-env align` |
| `align_en` | SoulX 英文对齐，包含 NeMo Parakeet ASR | `run_soulx_alignment_batch.py --conda-env align_en --language English` |
| `ymsp` | YingMusic 推理和 Streamlit 可视化 | `run_yingmusic_lyric_edit_tasks.py --conda-env ymsp`、`streamlit run` |

### Pipeline 轻量依赖

```bash
python -m pip install -r pipeline/requirements.txt
```

该文件只包含 pipeline 脚本直接用到的轻量包，例如 `requests`、`pypinyin`、`streamlit`、`soundfile`、`numpy`。它不包含 PyTorch、NeMo、FunASR 或 YingMusic 的模型依赖。

### SoulX 中文对齐环境

中文对齐推荐使用 `align` 环境。可按 SoulX 根目录依赖安装：

```bash
conda create -n align python=3.11 -y
conda activate align
pip install -r SoulX-Singer/requirements.txt
```

如果你只跑预处理，也可以参考：

```bash
pip install -r SoulX-Singer/preprocess/requirements.txt
```

SoulX 预训练权重应位于：

```text
SoulX-Singer/pretrained_models/SoulX-Singer-Preprocess/
SoulX-Singer/pretrained_models/SoulX-Singer/
```

### SoulX 英文对齐环境

英文 ASR 使用 NeMo Parakeet-TDT。建议单独建 `align_en`，不要直接覆盖中文 `align` 环境：

```bash
conda create -n align_en python=3.11 -y
conda activate align_en
pip install -r SoulX-Singer/preprocess/requirements.txt
pip install "nemo_toolkit[asr]==2.6.1" lhotse
```

英文模型文件应存在：

```text
SoulX-Singer/pretrained_models/SoulX-Singer-Preprocess/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo
```

本仓库对 SoulX 英文 ASR 有两个兼容修复：

- 兼容新版 PyTorch `Sampler` API。
- 禁用 NeMo TDT CUDA graph decoder，避免 `cuda-bindings` 返回值数量不一致导致的崩溃。

如果重新 clone 原始 SoulX-Singer，请应用补丁：

```bash
git -C SoulX-Singer apply ../patches/0001-Patch-English-ASR-sampler-compatibility.patch
```

### YingMusic 推理环境

YingMusic 使用 `ymsp` 环境：

```bash
conda create -n ymsp python=3.10 -y
conda activate ymsp
pip install -r YingMusic-Singer-Plus/requirements.txt
```

YingMusic 的歌词编辑 infilling 修改保存在补丁中。如果重新 clone 原始 YingMusic-Singer-Plus，请应用：

```bash
git -C YingMusic-Singer-Plus apply ../patches/0001-Add-lyric-edit-audio-infilling-controls.patch
```

国内下载 HuggingFace 权重时可加：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

DeepSeek API key 放在：

```text
pipeline/.env
```

格式：

```bash
deepseek_api_key=你的_api_key
```

## Pipeline 阶段文件

推荐使用下面的命名：

```text
pipeline/manifests/00_discovered.<dataset>.jsonl
pipeline/manifests/01_align.<dataset>_<lang>_<segment>.jsonl
pipeline/manifests/02_lyric_edit_tasks.<dataset>_<lang>_<segment>.jsonl
pipeline/runs/soulx_align_<dataset>_<lang>_<segment>/
pipeline/runs/yingmusic_lyric_edit_<dataset>_<lang>_<segment>/
pipeline/runs/soulx_lyric_edit_<dataset>_<lang>_<segment>/
```

`run_soulx_alignment_batch.py` 输出的 `alignment_results.jsonl` 通常复制到 `pipeline/manifests/01_*.jsonl` 作为下一阶段输入。

## 步骤 0：发现或抽取音频

### 通用音频扫描

适用于非 GTSinger 格式的数据，例如 `../music_example`。只扫描，不复制音频。

```bash
python pipeline/scripts/discover_audio.py \
  --dataset-root ../music_example/en/audios \
  --output pipeline/manifests/00_discovered.music_example_en.jsonl \
  --extensions .wav .flac .mp3 .m4a .ogg
```

常用参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--dataset-root` | 要递归扫描的数据集根目录 | 必填 |
| `--output` | 输出 JSONL manifest | 必填 |
| `--extensions` | 要包含的扩展名 | `.wav .flac .mp3 .m4a .ogg` |
| `--include-hidden` | 包含隐藏文件/目录 | 不开启 |
| `--follow-symlinks` | 跟随符号链接目录 | 不开启 |

### 从 GTSinger 抽取小数据集

会跳过路径中包含 `Paired_Speech_Group` 的朗读音频。

```bash
python pipeline/scripts/create_gtsinger_mini_dataset.py \
  --gtsinger-root /path/to/GTSinger \
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

## 步骤 1：SoulX 对齐

脚本：

```text
pipeline/scripts/run_soulx_alignment_batch.py
```

它递归扫描 `--audio-root` 下的音频，逐条调用 SoulX `preprocess.pipeline`，输出 ASR、歌词时间戳、音符、F0 和切片信息。默认支持 `.wav/.flac/.mp3/.m4a/.ogg`。

### 中文对齐示例

```bash
python pipeline/scripts/run_soulx_alignment_batch.py \
  --audio-root ../music_example/zh/audios \
  --soulx-root SoulX-Singer \
  --output-root pipeline/runs/soulx_align_music_example_zh_15s \
  --conda-env align \
  --language Mandarin \
  --device cuda \
  --max-merge-duration 15000 \
  --extensions .wav .flac .mp3 .m4a .ogg \
  --resume

cp pipeline/runs/soulx_align_music_example_zh_15s/alignment_results.jsonl \
  pipeline/manifests/01_align.music_example_zh_15s.jsonl
```

### 英文对齐示例

```bash
python pipeline/scripts/run_soulx_alignment_batch.py \
  --audio-root ../music_example/en/audios \
  --soulx-root SoulX-Singer \
  --output-root pipeline/runs/soulx_align_music_example_en_15s \
  --conda-env align_en \
  --language English \
  --device cuda \
  --max-merge-duration 15000 \
  --extensions .wav .flac .mp3 .m4a .ogg \
  --resume

cp pipeline/runs/soulx_align_music_example_en_15s/alignment_results.jsonl \
  pipeline/manifests/01_align.music_example_en_15s.jsonl
```

输出结构：

```text
pipeline/runs/soulx_align_<name>/alignment_results.jsonl
pipeline/runs/soulx_align_<name>/items/<item_id>/metadata.json
pipeline/runs/soulx_align_<name>/items/<item_id>/vocal.wav
pipeline/runs/soulx_align_<name>/logs/<item_id>.log
```

`metadata.json` 是 segment 数组。`--max-merge-duration 15000` 会让 SoulX 尽量把每个合并 segment 控制在约 15 秒以内。长音频会自然得到多个 segment，后续任务生成脚本会逐 segment 创建任务。

SoulX `note_type` 约定：

| `note_type` | 含义 | 任务脚本处理 |
| --- | --- | --- |
| `1` | 静音或 `<SP>/<AP>` | 跳过 |
| `2` | 新歌词 token | 中文作为字，英文作为 word |
| `3` | 前一个 token 的延长或重复发声 | 合并到前一个 token 的结束时间 |

常用参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--audio-root` | 待对齐音频根目录 | 必填 |
| `--soulx-root` | SoulX-Singer 项目根目录 | 必填 |
| `--output-root` | 对齐输出目录 | 必填 |
| `--conda-env` | 对齐环境 | `align` |
| `--language` | SoulX 语言参数：中文 `Mandarin`，英文 `English` | `Mandarin` |
| `--device` | 推理设备 | `cuda` |
| `--max-merge-duration` | 最大合并 segment 时长，毫秒 | `30000` |
| `--extensions` | 扫描扩展名 | `.wav .flac .mp3 .m4a .ogg` |
| `--resume` | 跳过已经成功的条目 | 不开启 |

## 步骤 2：创建歌词修改任务

脚本：

```text
pipeline/scripts/create_lyric_edit_tasks.py
```

输入是 SoulX 对齐 manifest。脚本读取每条 `metadata_path`，按 SoulX segment 还原歌词和 token 时间戳，然后调用 DeepSeek 生成局部替换任务。

中文任务：

```bash
python pipeline/scripts/create_lyric_edit_tasks.py \
  --input-manifest pipeline/manifests/01_align.music_example_zh_15s.jsonl \
  --output pipeline/manifests/02_lyric_edit_tasks.music_example_zh_15s.jsonl \
  --env-file pipeline/.env \
  --model deepseek-chat \
  --language Chinese \
  --max-word-len 4 \
  --overwrite
```

英文任务：

```bash
python pipeline/scripts/create_lyric_edit_tasks.py \
  --input-manifest pipeline/manifests/01_align.music_example_en_15s.jsonl \
  --output pipeline/manifests/02_lyric_edit_tasks.music_example_en_15s.jsonl \
  --env-file pipeline/.env \
  --model deepseek-chat \
  --language English \
  --max-word-len 4 \
  --overwrite
```

只检查 metadata 解析，不调用 LLM：

```bash
python pipeline/scripts/create_lyric_edit_tasks.py \
  --input-manifest pipeline/manifests/01_align.music_example_en_15s.jsonl \
  --language auto \
  --dry-run \
  --limit 1
```

成功任务字段示例：

```json
{
  "id": "f3_xxx_seg001",
  "audio_path": ".../song.flac",
  "metadata_path": ".../metadata.json",
  "segment_index": 1,
  "segment_start_sec": 6.0,
  "segment_end_sec": 23.28,
  "language": "English",
  "original_lyrics": "We are not all right when we see young girls",
  "edited_lyrics": "We are not all right when we see old men",
  "original_word": "young girls",
  "replacement_word": "old men",
  "char_start": 33,
  "char_end": 44,
  "edited_char_end": 40,
  "token_start": 8,
  "token_end": 10,
  "edit_start_sec": 8.24,
  "edit_end_sec": 8.72,
  "local_edit_start_sec": 2.24,
  "local_edit_end_sec": 2.72,
  "status": "success"
}
```

字段说明：

- `char_start/char_end`：原歌词中的字符范围，`char_end` 是 exclusive。
- `edited_char_end`：修改后歌词中替换词结束位置。英文替换词字母数可能不同，Streamlit 高亮会优先使用它。
- `token_start/token_end`：英文任务的 word token 范围；中文任务为 `null`。
- `edit_start_sec/edit_end_sec`：原音频全局时间。
- `local_edit_start_sec/local_edit_end_sec`：segment 内局部时间。

常用参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--input-manifest` | SoulX 对齐 JSONL，需要包含 `metadata_path` | 必填/默认旧中文路径 |
| `--output` | 成功任务 JSONL | 必填/默认旧中文路径 |
| `--failed-output` | 失败任务 JSONL | `<output stem>.failed.jsonl` |
| `--env-file` | 包含 `deepseek_api_key` 的 env 文件 | `pipeline/.env` |
| `--model` | DeepSeek 模型名 | `deepseek-v4-flash` |
| `--base-url` | DeepSeek Chat Completions API 地址 | `https://api.deepseek.com/chat/completions` |
| `--language` | `auto`、`Chinese`、`English` | `auto` |
| `--max-word-len` | 中文为最大汉字数；英文为最大 word 数 | `4` |
| `--max-retries` | LLM 输出不合法时的重试次数 | `3` |
| `--limit` | 只处理前 N 条 | 不限制 |
| `--dry-run` | 只解析 metadata，不调用 LLM | 不开启 |
| `--overwrite` | 覆盖已有输出 | 不开启 |

## 步骤 3：YingMusic 局部歌词编辑

脚本：

```text
pipeline/scripts/run_yingmusic_lyric_edit_tasks.py
```

当前 YingMusic 修改逻辑：

- 不需要单独指定音色参考；`ref_audio_path=None` 时使用 melody audio 作为音色来源。
- 对每个 SoulX segment 任务，脚本会先从原音频裁出 segment 输入音频。
- 音频 prompt 的未编辑区域保留原始 latent，编辑区域 hard mask 后由模型生成。
- 可用 `--mask-start-offset-sec` 和 `--mask-end-offset-sec` 只扩大 mask 区域。
- 生成后未 mask latent 会替换回原始 latent，减少未编辑区域漂移。
- 歌词条件和 melody/MIDI 条件保持 YingMusic 原模型流程。

中文或英文任务都使用同一个推理脚本。示例：

```bash
HF_ENDPOINT=https://hf-mirror.com \
python pipeline/scripts/run_yingmusic_lyric_edit_tasks.py \
  --task-manifest pipeline/manifests/02_lyric_edit_tasks.music_example_en_15s.jsonl \
  --yingmusic-root YingMusic-Singer-Plus \
  --output-dir pipeline/runs/yingmusic_lyric_edit_music_example_en_15s \
  --conda-env ymsp \
  --ckpt-path ASLP-lab/YingMusic-Singer-Plus \
  --device cuda:0 \
  --mask-start-offset-sec 0.2 \
  --mask-end-offset-sec 0.2 \
  --overwrite
```

只跑一条 smoke test：

```bash
HF_ENDPOINT=https://hf-mirror.com \
python pipeline/scripts/run_yingmusic_lyric_edit_tasks.py \
  --task-manifest pipeline/manifests/02_lyric_edit_tasks.music_example_en_15s.jsonl \
  --output-dir pipeline/runs/yingmusic_lyric_edit_smoke \
  --limit 1 \
  --verbose \
  --overwrite
```

输出：

```text
pipeline/runs/yingmusic_lyric_edit_<name>/inference_results.jsonl
pipeline/runs/yingmusic_lyric_edit_<name>/wavs/<task_id>.wav
pipeline/runs/yingmusic_lyric_edit_<name>/segments/<task_id>.wav
pipeline/runs/yingmusic_lyric_edit_<name>/logs/<task_id>.log
```

常用参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--task-manifest` | 歌词修改任务 JSONL | 默认旧中文路径 |
| `--yingmusic-root` | YingMusic-Singer-Plus 根目录 | `YingMusic-Singer-Plus` |
| `--output-dir` | 推理输出目录 | 默认旧中文路径 |
| `--conda-env` | YingMusic 环境 | `ymsp` |
| `--ckpt-path` | HuggingFace repo id 或本地权重目录 | `ASLP-lab/YingMusic-Singer-Plus` |
| `--device` | 推理设备 | `cuda:0` |
| `--nfe-step` | 采样步数 | `32` |
| `--cfg-strength` | CFG 强度 | `3.0` |
| `--t-shift` | 采样时间偏移 | `0.5` |
| `--sil-len-to-end` | 音色 prompt 后静音秒数 | `0.5` |
| `--mask-start-offset-sec` | mask 起点提前秒数，只扩大 mask | `0.0` |
| `--mask-end-offset-sec` | mask 终点推迟秒数，只扩大 mask | `0.0` |
| `--seed` | 随机种子，每条任务使用 `seed + idx` | `20260706` |
| `--limit` | 只跑前 N 条 | 不限制 |
| `--overwrite` | 重新生成已有 wav | 不开启 |
| `--verbose` | 失败日志写完整 traceback | 不开启 |

## *步骤 4(optional)：SoulX 对照实验

脚本：

```text
pipeline/scripts/run_soulx_lyric_edit_tasks.py
```

这个对照会把 task 转成 SoulX target metadata，然后让 SoulX 重新合成整段音频。它不会像 YingMusic infilling 那样保留未编辑区域的原始 latent。

```bash
conda run -n align python pipeline/scripts/run_soulx_lyric_edit_tasks.py \
  --task-manifest pipeline/manifests/02_lyric_edit_tasks.gtsinger_mini_100_chinese.jsonl \
  --soulx-root SoulX-Singer \
  --output-dir pipeline/runs/soulx_lyric_edit_gtsinger_mini_100_chinese \
  --control melody \
  --device cuda \
  --fp16 \
  --overwrite
```

只生成 target metadata，不跑模型：

```bash
conda run -n align python pipeline/scripts/run_soulx_lyric_edit_tasks.py \
  --limit 1 \
  --prepare-only \
  --output-dir pipeline/runs/soulx_lyric_edit_prepare_smoke \
  --overwrite
```

注意：当前 SoulX 对照脚本主要按中文逐字任务设计。英文任务建议先以 YingMusic 主流程为准。

## 步骤 5：可视化检查

脚本：

```text
pipeline/scripts/view_lyric_edit_tasks.py
```

功能：

- 展示原歌词、修改歌词、修改词和时间戳。
- 支持中文和英文高亮；英文会使用 `edited_char_end` 或替换词长度修正绿色高亮范围。
- 类 DAW 三音轨波形：原始/segment 输入、YingMusic 输出、SoulX 对照输出。
- 红色播放头穿过所有音轨，空格播放/暂停。
- 点击音轨或按钮选择当前播放音轨。
- 结果 manifest 可以暂时为空或不存在，方便只检查 task。

启动：

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

在左侧 sidebar 修改：

- `Task manifest`
- `YingMusic result manifest`
- `SoulX result manifest`

## 常用检查命令

统计推理结果：

```bash
python - <<'EOF'
import json
from pathlib import Path
p = Path('pipeline/runs/yingmusic_lyric_edit_music_example_en_15s/inference_results.jsonl')
rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
print('rows', len(rows))
print('success', sum(r.get('status') == 'success' for r in rows))
print('failed', [r['id'] for r in rows if r.get('status') != 'success'])
EOF
```

查看失败日志：

```bash
ls pipeline/runs/yingmusic_lyric_edit_music_example_en_15s/logs
```

统计生成音频：

```bash
find pipeline/runs/yingmusic_lyric_edit_music_example_en_15s/wavs -name '*.wav' | wc -l
```

## 一键顺序示例：英文 music_example

```bash
python pipeline/scripts/run_soulx_alignment_batch.py \
  --audio-root ../music_example/en/audios \
  --soulx-root SoulX-Singer \
  --output-root pipeline/runs/soulx_align_music_example_en_15s \
  --conda-env align_en \
  --language English \
  --device cuda \
  --max-merge-duration 15000 \
  --extensions .wav .flac .mp3 .m4a .ogg \
  --resume

cp pipeline/runs/soulx_align_music_example_en_15s/alignment_results.jsonl \
  pipeline/manifests/01_align.music_example_en_15s.jsonl

python pipeline/scripts/create_lyric_edit_tasks.py \
  --input-manifest pipeline/manifests/01_align.music_example_en_15s.jsonl \
  --output pipeline/manifests/02_lyric_edit_tasks.music_example_en_15s.jsonl \
  --env-file pipeline/.env \
  --model deepseek-chat \
  --language English \
  --overwrite

HF_ENDPOINT=https://hf-mirror.com \
python pipeline/scripts/run_yingmusic_lyric_edit_tasks.py \
  --task-manifest pipeline/manifests/02_lyric_edit_tasks.music_example_en_15s.jsonl \
  --yingmusic-root YingMusic-Singer-Plus \
  --output-dir pipeline/runs/yingmusic_lyric_edit_music_example_en_15s \
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

## 一键顺序示例：中文 music_example

```bash
python pipeline/scripts/run_soulx_alignment_batch.py \
  --audio-root ../music_example/zh/audios \
  --soulx-root SoulX-Singer \
  --output-root pipeline/runs/soulx_align_music_example_zh_15s \
  --conda-env align \
  --language Mandarin \
  --device cuda \
  --max-merge-duration 15000 \
  --extensions .wav .flac .mp3 .m4a .ogg \
  --resume

cp pipeline/runs/soulx_align_music_example_zh_15s/alignment_results.jsonl \
  pipeline/manifests/01_align.music_example_zh_15s.jsonl

python pipeline/scripts/create_lyric_edit_tasks.py \
  --input-manifest pipeline/manifests/01_align.music_example_zh_15s.jsonl \
  --output pipeline/manifests/02_lyric_edit_tasks.music_example_zh_15s.jsonl \
  --env-file pipeline/.env \
  --model deepseek-chat \
  --language Chinese \
  --overwrite

HF_ENDPOINT=https://hf-mirror.com \
python pipeline/scripts/run_yingmusic_lyric_edit_tasks.py \
  --task-manifest pipeline/manifests/02_lyric_edit_tasks.music_example_zh_15s.jsonl \
  --yingmusic-root YingMusic-Singer-Plus \
  --output-dir pipeline/runs/yingmusic_lyric_edit_music_example_zh_15s \
  --conda-env ymsp \
  --ckpt-path ASLP-lab/YingMusic-Singer-Plus \
  --device cuda:0 \
  --mask-start-offset-sec 0.2 \
  --mask-end-offset-sec 0.3 \
  --overwrite
```

## 上游项目补丁

根仓库 `patches/` 目录保存了本工作流对两个上游项目的本地改动：

```text
patches/0001-Add-lyric-edit-audio-infilling-controls.patch
patches/0001-Patch-English-ASR-sampler-compatibility.patch
```

重新 clone 原始上游仓库后应用：

```bash
git -C YingMusic-Singer-Plus apply ../patches/0001-Add-lyric-edit-audio-infilling-controls.patch
git -C SoulX-Singer apply ../patches/0001-Patch-English-ASR-sampler-compatibility.patch
```

如果你已经把两个上游仓库也推到自己的 fork，可以直接 clone fork，不需要再 apply patch。

## 常见问题

### DeepSeek API 报错

检查 `pipeline/.env` 是否存在，且 key 名必须是：

```bash
deepseek_api_key=...
```

先用 `--dry-run` 验证 SoulX metadata 是否正常，排除对齐问题。

### HuggingFace 权重下载慢或失败

在推理命令前加：

```bash
HF_ENDPOINT=https://hf-mirror.com
```

如果机器上已有本地权重，直接用 `--ckpt-path /path/to/local/ckpt`。

### SoulX 对齐中断

重新运行时加 `--resume`。如果某条失败，查看：

```bash
pipeline/runs/<align_run>/logs/<item_id>.log
```

英文 NeMo 失败时，优先确认 `align_en` 环境和 SoulX 英文 ASR 补丁是否已应用。

### Streamlit 页面高亮不准

英文任务请确认 task 中有 `replacement_word`，最好有 `edited_char_end`。旧任务没有 `edited_char_end` 时，viewer 会用 `char_start + len(replacement_word)` 自动修正修改后歌词的绿色高亮范围。

### Streamlit 页面没有显示修改音频

确认左侧 `YingMusic result manifest` 指向推理输出目录下的 `inference_results.jsonl`，且其中 `output_path` 文件存在。
