#!/usr/bin/env python3
"""Streamlit viewer for YingMusic lyric edit tasks."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components


DEFAULT_TASK_MANIFEST = Path(
    "pipeline/manifests/02_lyric_edit_tasks.gtsinger_mini_40_chinese.jsonl"
)
DEFAULT_RESULT_MANIFEST = Path(
    "pipeline/runs/yingmusic_lyric_edit_gtsinger_mini_40_chinese_hardmask/inference_results.jsonl"
)


st.set_page_config(
    page_title="Lyric Edit Tasks",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
:root { color-scheme: light; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
[data-testid="stMetricValue"] { font-size: 1.05rem; }
.task-title { font-size: 1.05rem; font-weight: 650; margin-bottom: .15rem; }
.lyric-box {
  border: 1px solid #d8dde6;
  border-radius: 8px;
  padding: 12px 14px;
  background: #ffffff;
  min-height: 86px;
  font-size: 1.06rem;
  line-height: 1.9;
}
.lyric-label { color: #586174; font-size: .82rem; margin-bottom: 4px; }
mark.edit-original { background: #ffe0d4; color: #83230c; padding: 2px 4px; border-radius: 4px; }
mark.edit-new { background: #d8f4df; color: #165b2b; padding: 2px 4px; border-radius: 4px; }
.daw-wrap {
  border: 1px solid #ccd3de;
  border-radius: 8px;
  background: #f8fafc;
  padding: 12px;
}
.daw-head {
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap: 12px;
  margin-bottom: 8px;
  color:#485465;
  font-size: 13px;
}
.daw-grid { background: #ffffff; border:1px solid #d9e0ea; border-radius: 6px; overflow:hidden; }
.track-label { font-size: 12px; fill: #3f4a59; font-weight: 650; }
.time-label { font-size: 10px; fill: #6b7482; }
.region-label { font-size: 11px; fill: #91400c; font-weight: 650; }
.missing { color:#8a1f11; background:#fff0ed; border:1px solid #f3b7a8; padding:10px 12px; border-radius:8px; }
</style>
"""


@st.cache_data(show_spinner=False)
def load_jsonl(path_text: str) -> list[dict[str, Any]]:
    path = Path(path_text).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@st.cache_data(show_spinner=False)
def load_waveform(path_text: str, max_points: int = 1400) -> dict[str, Any]:
    path = Path(path_text).expanduser().resolve()
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    mono = data.mean(axis=1)
    duration = float(len(mono) / sr) if sr else 0.0
    if len(mono) == 0:
        peaks = np.zeros((max_points,), dtype=np.float32)
    else:
        frames = min(max_points, len(mono))
        edges = np.linspace(0, len(mono), frames + 1, dtype=np.int64)
        peaks = np.empty(frames, dtype=np.float32)
        for i in range(frames):
            chunk = mono[edges[i] : edges[i + 1]]
            peaks[i] = float(np.max(np.abs(chunk))) if len(chunk) else 0.0
        max_amp = float(peaks.max()) if len(peaks) else 0.0
        if max_amp > 0:
            peaks = peaks / max_amp
    return {"peaks": peaks.tolist(), "sr": sr, "duration": duration, "channels": data.shape[1]}


def merge_tasks(
    tasks: list[dict[str, Any]], results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result_by_id = {row.get("id"): row for row in results}
    merged = []
    for task in tasks:
        row = dict(task)
        result = result_by_id.get(task.get("id"), {})
        row["result"] = result
        row["output_path"] = result.get("output_path")
        row["result_status"] = result.get("status", "missing")
        merged.append(row)
    return merged


def highlight_text(text: str, start: int | None, end: int | None, css_class: str) -> str:
    if start is None or end is None or start < 0 or end <= start or start >= len(text):
        return html.escape(text)
    end = min(end, len(text))
    return (
        html.escape(text[:start])
        + f'<mark class="{css_class}">'
        + html.escape(text[start:end])
        + "</mark>"
        + html.escape(text[end:])
    )


def waveform_polyline(
    peaks: list[float], x0: float, y_mid: float, width: float, half_height: float
) -> str:
    if not peaks:
        return ""
    n = len(peaks)
    if n == 1:
        return f"M {x0:.2f} {y_mid:.2f}"
    top = []
    bottom = []
    for i, amp in enumerate(peaks):
        x = x0 + width * i / (n - 1)
        y_top = y_mid - half_height * float(amp)
        y_bottom = y_mid + half_height * float(amp)
        top.append(f"{x:.2f},{y_top:.2f}")
        bottom.append(f"{x:.2f},{y_bottom:.2f}")
    return "M " + " L ".join(top + bottom[::-1]) + " Z"


def audio_data_uri(path: Path | None) -> str:
    if not path:
        return ""
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def render_daw(
    original: dict[str, Any],
    edited: dict[str, Any] | None,
    edit_start: float,
    edit_end: float,
    task_id: str,
    original_audio_uri: str,
    edited_audio_uri: str,
) -> str:
    width = 1180
    left = 92
    right = 22
    top = 40
    track_h = 86
    gap = 18
    bottom = 34
    inner_w = width - left - right
    total_duration = max(
        float(original.get("duration") or 0.0),
        float((edited or {}).get("duration") or 0.0),
        float(edit_end or 0.0),
        0.1,
    )
    height = top + track_h * 2 + gap + bottom

    def x_for_time(t: float) -> float:
        t = max(0.0, min(float(t), total_duration))
        return left + inner_w * t / total_duration

    edit_x = x_for_time(edit_start)
    edit_w = max(1.0, x_for_time(edit_end) - edit_x)
    tick_count = min(12, max(4, int(total_duration) + 1))

    tick_svg = []
    for i in range(tick_count + 1):
        t = total_duration * i / tick_count
        x = x_for_time(t)
        tick_svg.append(
            f'<line x1="{x:.2f}" y1="24" x2="{x:.2f}" y2="{height - 18}" stroke="#e8edf3" stroke-width="1" />'
        )
        tick_svg.append(
            f'<text x="{x:.2f}" y="18" text-anchor="middle" class="time-label">{t:.1f}s</text>'
        )

    orig_path = waveform_polyline(
        original["peaks"], left, top + track_h / 2, inner_w, track_h * 0.36
    )
    edit_path = ""
    if edited:
        edit_path = waveform_polyline(
            edited["peaks"],
            left,
            top + track_h + gap + track_h / 2,
            inner_w,
            track_h * 0.36,
        )

    escaped_id = html.escape(task_id)
    edit_text = html.escape(f"edit {edit_start:.2f}s - {edit_end:.2f}s")
    original_audio_uri = html.escape(original_audio_uri, quote=True)
    edited_audio_uri = html.escape(edited_audio_uri, quote=True)
    has_edited = "true" if edited_audio_uri else "false"

    return f"""
<style>
  .daw-wrap {{
    border: 1px solid #ccd3de;
    border-radius: 8px;
    background: #f8fafc;
    padding: 12px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #1f2937;
    user-select: none;
  }}
  .daw-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
    color: #485465;
    font-size: 13px;
  }}
  .transport {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
  .transport button {{
    border: 1px solid #cbd5e1;
    background: #fff;
    color: #1f2937;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
    cursor: pointer;
  }}
  .transport button.active {{ border-color: #2563eb; background: #eaf2ff; color: #1d4ed8; }}
  .transport button:disabled {{ opacity: .45; cursor: not-allowed; }}
  .time-readout {{ font-variant-numeric: tabular-nums; min-width: 112px; text-align: right; }}
  .daw-grid {{ background: #ffffff; border: 1px solid #d9e0ea; border-radius: 6px; overflow: hidden; }}
  .track-label {{ font-size: 12px; fill: #3f4a59; font-weight: 650; pointer-events: none; }}
  .time-label {{ font-size: 10px; fill: #6b7482; pointer-events: none; }}
  .region-label {{ font-size: 11px; fill: #91400c; font-weight: 650; pointer-events: none; }}
  .track-bg {{ cursor: pointer; }}
  .track-bg.selected {{ stroke: #2563eb; stroke-width: 2.5; fill: #eff6ff; }}
  .wave {{ pointer-events: none; }}
  .playhead {{ cursor: ew-resize; }}
  .hint {{ color: #6b7280; font-size: 12px; }}
</style>
<div class="daw-wrap" id="dawRoot" tabindex="0">
  <audio id="originalAudio" src="{original_audio_uri}" preload="auto"></audio>
  <audio id="editedAudio" src="{edited_audio_uri}" preload="auto"></audio>
  <div class="daw-head">
    <div><strong>{escaped_id}</strong></div>
    <div class="transport">
      <button id="playBtn" type="button">Play</button>
      <button id="origBtn" type="button" class="active">Original</button>
      <button id="editBtn" type="button">Edited</button>
      <span class="time-readout"><span id="timeNow">0.00</span>s / {total_duration:.2f}s</span>
    </div>
  </div>
  <div class="daw-grid">
    <svg id="dawSvg" viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="waveform tracks">
      <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />
      {''.join(tick_svg)}
      <rect id="origTrack" class="track-bg selected" data-track="original" x="{left}" y="{top}" width="{inner_w}" height="{track_h}" rx="2" fill="#eff6ff" stroke="#2563eb" />
      <rect id="editTrack" class="track-bg" data-track="edited" x="{left}" y="{top + track_h + gap}" width="{inner_w}" height="{track_h}" rx="2" fill="#fbfdff" stroke="#d9e0ea" />
      <rect x="{edit_x:.2f}" y="{top}" width="{edit_w:.2f}" height="{track_h * 2 + gap}" fill="#f97316" opacity="0.18" pointer-events="none" />
      <text x="16" y="{top + track_h / 2 + 4:.2f}" class="track-label">Original</text>
      <text x="16" y="{top + track_h + gap + track_h / 2 + 4:.2f}" class="track-label">Edited</text>
      <path d="{orig_path}" class="wave" fill="#2563eb" opacity="0.72" />
      <path d="{edit_path}" class="wave" fill="#16a34a" opacity="0.72" />
      <line x1="{edit_x:.2f}" y1="{top}" x2="{edit_x:.2f}" y2="{top + track_h * 2 + gap}" stroke="#ea580c" stroke-width="2" pointer-events="none" />
      <line x1="{edit_x + edit_w:.2f}" y1="{top}" x2="{edit_x + edit_w:.2f}" y2="{top + track_h * 2 + gap}" stroke="#ea580c" stroke-width="2" pointer-events="none" />
      <text x="{edit_x + 6:.2f}" y="{top + 16}" class="region-label">{edit_text}</text>
      <g id="playhead" class="playhead" transform="translate({left},0)">
        <line x1="0" y1="24" x2="0" y2="{height - 8}" stroke="#dc2626" stroke-width="2.5" />
        <polygon points="-7,24 7,24 0,34" fill="#dc2626" />
        <rect x="-8" y="24" width="16" height="{height - 32}" fill="transparent" />
      </g>
    </svg>
  </div>
  <div class="hint">Space: play/pause selected track. Click a track to select it. Drag or click the red playhead area to seek.</div>
</div>
<script>
(() => {{
  const root = document.getElementById('dawRoot');
  const svg = document.getElementById('dawSvg');
  const playhead = document.getElementById('playhead');
  const originalAudio = document.getElementById('originalAudio');
  const editedAudio = document.getElementById('editedAudio');
  const origBtn = document.getElementById('origBtn');
  const editBtn = document.getElementById('editBtn');
  const playBtn = document.getElementById('playBtn');
  const timeNow = document.getElementById('timeNow');
  const origTrack = document.getElementById('origTrack');
  const editTrack = document.getElementById('editTrack');
  const left = {left};
  const innerW = {inner_w};
  const totalDuration = {total_duration:.8f};
  const hasEdited = {has_edited};
  let selected = 'original';
  let dragging = false;

  if (!hasEdited) editBtn.disabled = true;

  function audioFor(track) {{ return track === 'edited' ? editedAudio : originalAudio; }}
  function otherAudio(track) {{ return track === 'edited' ? originalAudio : editedAudio; }}
  function clampTime(t) {{ return Math.max(0, Math.min(totalDuration, t)); }}
  function xForTime(t) {{ return left + innerW * clampTime(t) / totalDuration; }}
  function timeForClientX(clientX) {{
    const pt = svg.createSVGPoint();
    pt.x = clientX;
    pt.y = 0;
    const svgP = pt.matrixTransform(svg.getScreenCTM().inverse());
    return clampTime((svgP.x - left) * totalDuration / innerW);
  }}
  function setPlayhead(t) {{
    const time = clampTime(t || 0);
    playhead.setAttribute('transform', `translate(${{xForTime(time)}},0)`);
    timeNow.textContent = time.toFixed(2);
  }}
  function setSelected(track) {{
    if (track === 'edited' && !hasEdited) return;
    const previous = audioFor(selected);
    selected = track;
    otherAudio(selected).pause();
    origBtn.classList.toggle('active', selected === 'original');
    editBtn.classList.toggle('active', selected === 'edited');
    origTrack.classList.toggle('selected', selected === 'original');
    editTrack.classList.toggle('selected', selected === 'edited');
    origTrack.setAttribute('fill', selected === 'original' ? '#eff6ff' : '#fbfdff');
    editTrack.setAttribute('fill', selected === 'edited' ? '#eff6ff' : '#fbfdff');
    const current = audioFor(selected);
    current.currentTime = clampTime(previous.currentTime || 0);
    setPlayhead(current.currentTime);
    updatePlayButton();
  }}
  function updatePlayButton() {{ playBtn.textContent = audioFor(selected).paused ? 'Play' : 'Pause'; }}
  async function togglePlay() {{
    const audio = audioFor(selected);
    otherAudio(selected).pause();
    if (audio.paused) {{
      if (audio.currentTime >= Math.min(audio.duration || totalDuration, totalDuration) - 0.02) audio.currentTime = 0;
      try {{ await audio.play(); }} catch (err) {{ console.warn(err); }}
    }} else {{
      audio.pause();
    }}
    updatePlayButton();
  }}
  function seekTo(t) {{
    const time = clampTime(t);
    originalAudio.currentTime = Math.min(time, originalAudio.duration || time);
    if (hasEdited) editedAudio.currentTime = Math.min(time, editedAudio.duration || time);
    setPlayhead(time);
  }}
  function onPointerSeek(event) {{
    seekTo(timeForClientX(event.clientX));
    root.focus();
  }}
  function animationLoop() {{
    setPlayhead(audioFor(selected).currentTime || 0);
    updatePlayButton();
    requestAnimationFrame(animationLoop);
  }}

  origBtn.addEventListener('click', () => setSelected('original'));
  editBtn.addEventListener('click', () => setSelected('edited'));
  origTrack.addEventListener('click', (event) => {{ setSelected('original'); onPointerSeek(event); }});
  editTrack.addEventListener('click', (event) => {{ setSelected('edited'); onPointerSeek(event); }});
  playBtn.addEventListener('click', togglePlay);
  svg.addEventListener('pointerdown', (event) => {{
    dragging = true;
    svg.setPointerCapture(event.pointerId);
    onPointerSeek(event);
  }});
  svg.addEventListener('pointermove', (event) => {{ if (dragging) onPointerSeek(event); }});
  svg.addEventListener('pointerup', (event) => {{ dragging = false; svg.releasePointerCapture(event.pointerId); }});
  svg.addEventListener('pointercancel', () => {{ dragging = false; }});
  root.addEventListener('keydown', (event) => {{
    if (event.code === 'Space') {{
      event.preventDefault();
      togglePlay();
    }}
  }});
  document.addEventListener('keydown', (event) => {{
    if (event.code === 'Space' && document.activeElement === document.body) {{
      event.preventDefault();
      root.focus();
      togglePlay();
    }}
  }});
  [originalAudio, editedAudio].forEach((audio) => {{
    audio.addEventListener('pause', updatePlayButton);
    audio.addEventListener('play', updatePlayButton);
    audio.addEventListener('ended', updatePlayButton);
  }});

  setSelected('original');
  setPlayhead(0);
  requestAnimationFrame(animationLoop);
}})();
</script>
"""

def existing_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    return path if path.exists() else None



def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Lyric Edit Task Viewer")

    with st.sidebar:
        st.header("Data")
        task_manifest = st.text_input("Task manifest", str(DEFAULT_TASK_MANIFEST))
        result_manifest = st.text_input("Result manifest", str(DEFAULT_RESULT_MANIFEST))
        st.divider()
        st.header("Task")

    try:
        tasks = load_jsonl(task_manifest)
        results = load_jsonl(result_manifest)
    except Exception as exc:
        st.error(f"Failed to load manifests: {exc}")
        return

    rows = merge_tasks(tasks, results)
    if not rows:
        st.warning("No tasks found.")
        return

    ids = [row["id"] for row in rows]
    with st.sidebar:
        selected_id = st.selectbox("Task id", ids, index=0)
        index = ids.index(selected_id)
        st.caption(f"{index + 1} / {len(rows)}")

    row = rows[index]
    original_path = existing_path(row.get("audio_path"))
    edited_path = existing_path(row.get("output_path"))
    edit_start = float(row.get("edit_start_sec", 0.0))
    edit_end = float(row.get("edit_end_sec", edit_start))

    if not original_path:
        st.error(f"Original audio missing: {row.get('audio_path')}")
        return

    original_wave = load_waveform(str(original_path))
    edited_wave = load_waveform(str(edited_path)) if edited_path else None
    total_duration = max(
        float(original_wave.get("duration") or 0.0),
        float((edited_wave or {}).get("duration") or 0.0),
        edit_end,
        0.1,
    )

    top_cols = st.columns([1.4, 1, 1, 1])
    top_cols[0].markdown(f'<div class="task-title">{html.escape(row["id"])}</div>', unsafe_allow_html=True)
    top_cols[1].metric("修改部分", f'{row.get("original_word", "")} → {row.get("replacement_word", "")}')
    top_cols[2].metric("起始", f"{edit_start:.3f}s")
    top_cols[3].metric("结束", f"{edit_end:.3f}s")

    lyric_cols = st.columns(2)
    char_start = row.get("char_start")
    char_end = row.get("char_end")
    original_html = highlight_text(
        row.get("original_lyrics", ""), char_start, char_end, "edit-original"
    )
    edited_html = highlight_text(
        row.get("edited_lyrics", ""), char_start, char_end, "edit-new"
    )
    lyric_cols[0].markdown(
        f'<div class="lyric-label">原歌词</div><div class="lyric-box">{original_html}</div>',
        unsafe_allow_html=True,
    )
    lyric_cols[1].markdown(
        f'<div class="lyric-label">修改歌词</div><div class="lyric-box">{edited_html}</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Tracks")
    components.html(
        render_daw(
            original_wave,
            edited_wave,
            edit_start,
            edit_end,
            row["id"],
            audio_data_uri(original_path),
            audio_data_uri(edited_path),
        ),
        height=330,
        scrolling=False,
    )

    path_cols = st.columns(2)
    path_cols[0].caption("原音频")
    path_cols[0].code(str(original_path), language=None)
    path_cols[1].caption("修改音频")
    path_cols[1].code(str(edited_path) if edited_path else "音频文件不存在", language=None)

    with st.expander("Task JSON"):
        st.json(row, expanded=False)


if __name__ == "__main__":
    main()
