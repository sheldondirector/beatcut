#!/usr/bin/env python3
from __future__ import annotations
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory, flash
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
import subprocess, shutil, json, os
import numpy as np

# Compatibility shims for deprecated numpy aliases used by some libs
if not hasattr(np, "complex"):
    np.complex = np.complex128  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = np.float64  # type: ignore[attr-defined]

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FRAME_HOP = 512

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")
BASE_DIR = Path(__file__).parent.resolve()
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

INDEX_HTML = """
<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
    <title>Flash-cut Builder</title>
    <link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css\">
    <style>
        body { padding-block: 1.5rem; background: #f8fafc; }
        .centered { max-width: 480px; margin: 2rem auto; background: #fff; border-radius: 16px; box-shadow: 0 2px 16px #0001; padding: 2rem; }
        .logo { display: block; margin: 0 auto 1.5rem; width: 64px; }
        .form-group { margin-bottom: 1.5rem; }
        .advanced { display: none; margin-top: 1.5rem; }
        .show-advanced .advanced { display: block; }
        .toggle-adv { color: #2563eb; background: none; border: none; cursor: pointer; font-size: 1rem; margin-bottom: 1rem; }
        .spinner-overlay { display: none; position: fixed; z-index: 9999; inset: 0; background: rgba(255,255,255,0.8); align-items: center; justify-content: center; flex-direction: column; }
        .spinner { border: 6px solid #eee; border-top: 6px solid #2563eb; border-radius: 50%; width: 48px; height: 48px; animation: spin 1s linear infinite; margin-bottom: 1.5rem; }
        .progress-msg { font-size: 1.2rem; font-weight: 500; color: #1e3a8a; text-align: center; max-width: 320px; }
        .progress-substep { font-size: 0.9rem; color: #64748b; margin-top: 0.5rem; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <main class=\"centered\">
        <img src=\"https://cdn.jsdelivr.net/gh/tabler/tabler-icons/icons/music.svg\" class=\"logo\" alt=\"logo\">
        <h2 style=\"text-align:center;\">Flash-cut Builder</h2>
        {% with messages = get_flashed_messages() %}
            {% if messages %}<article>{% for m in messages %}<p>{{m}}</p>{% endfor %}</article>{% endif %}
        {% endwith %}
        <form id=\"mainForm\" action=\"{{ url_for('analyze') }}\" method=\"post\" enctype=\"multipart/form-data\">
            <div class=\"form-group\">
                <label for=\"audio\"><b>Audio File</b></label>
                <input type=\"file\" id=\"audio\" name=\"audio\" accept=\".mp3,.wav,.flac,.ogg,.m4a,audio/*\" required>
            </div>
            <div class=\"form-group\">
                <label for=\"fps\"><b>Frames per Second (FPS)</b></label>
                <input type=\"number\" id=\"fps\" name=\"fps\" step=\"1\" min=\"10\" max=\"120\" value=\"30\" required>
            </div>
            <button type=\"button\" class=\"toggle-adv\" onclick=\"document.body.classList.toggle('show-advanced')\">Show advanced options</button>
            <div class=\"advanced\">
                <div class=\"form-group\">
                    <label>Aspect Ratio</label>
                    <select name=\"aspect_ratio\">
                        <option value=\"16:9\" selected>16:9 (widescreen)</option>
                        <option value=\"1:1\">1:1 (square)</option>
                        <option value=\"9:16\">9:16 (vertical)</option>
                        <option value=\"4:3\">4:3</option>
                    </select>
                </div>
                <div class=\"form-group\">
                    <label for=\"threshold\">Onset threshold</label>
                    <input type=\"number\" id=\"threshold\" name=\"threshold\" step=\"0.01\" min=\"0\" max=\"1\" value=\"0.30\">
                </div>
                <div class=\"form-group\">
                    <label for=\"max_gap\">Max gap (s)</label>
                    <input type=\"number\" id=\"max_gap\" name=\"max_gap\" step=\"0.05\" min=\"0.1\" max=\"10\" value=\"5.0\">
                </div>
                <div class=\"form-group\">
                    <label>Flash window</label>
                    <div style=\"display:flex;gap:0.5rem;\">
                        <input type=\"number\" name=\"flash_start\" step=\"0.1\" value=\"10\" placeholder=\"Start (s)\">
                        <input type=\"number\" name=\"flash_end\" step=\"0.1\" value=\"25\" placeholder=\"End (s)\">
                        <input type=\"number\" name=\"flash_gap\" step=\"0.01\" value=\"0.12\" placeholder=\"Min gap (s)\">
                    </div>
                </div>
                <div class=\"form-group\">
                    <label><input type=\"checkbox\" name=\"do_render\" value=\"1\" checked> Render video with ffmpeg</label>
                </div>
                <div class=\"form-group\">
                    <label>Video clips (multiple)</label>
                    <input type=\"file\" name=\"videos\" accept=\"video/*\" multiple>
                </div>
                <div class=\"form-group\">
                    <label>PNG images (multiple)</label>
                    <input type=\"file\" name=\"images\" accept=\"image/png\" multiple>
                </div>
                <div class=\"form-group\">
                    <label>Clip portion</label>
                    <select name=\"clip_mode\"><option value=\"head\" selected>Head (start)</option><option value=\"tail\">Tail (end)</option></select>
                </div>
                <div class=\"form-group\">
                    <label>Output file name</label>
                    <input type=\"text\" name=\"output_name\" value=\"final_video.mp4\">
                </div>
            </div>
            <button type=\"submit\" style=\"width:100%;font-size:1.2rem;margin-top:1.5rem;\">Analyze</button>
        </form>
        <div class=\"spinner-overlay\" id=\"spinnerOverlay\">
            <div class=\"spinner\"></div>
            <div class=\"progress-msg\" id=\"progressMsg\">Preparing to analyze...</div>
            <div class=\"progress-substep\" id=\"progressSubstep\"></div>
        </div>
    </main>
    <script>
        document.getElementById('mainForm').addEventListener('submit', function() {
            // Show spinner overlay
            document.getElementById('spinnerOverlay').style.display = 'flex';
            
            // Simulate progress updates
            const progressMsg = document.getElementById('progressMsg');
            const progressSubstep = document.getElementById('progressSubstep');
            const steps = [
                {msg: 'Processing audio file...', sub: 'Analyzing waveform', delay: 800},
                {msg: 'Detecting onsets...', sub: 'Finding beats and segments', delay: 1500},
                {msg: 'Creating segments...', sub: 'Mapping beats to time segments', delay: 1200},
                {msg: 'Generating flash cuts...', sub: 'Calculating optimal cuts', delay: 1500},
                {msg: 'Rendering media...', sub: 'Processing video clips', delay: 2000},
                {msg: 'Finalizing output...', sub: 'Muxing audio and video', delay: 1500},
            ];
            
            let stepIndex = 0;
            function updateProgress() {
                if (stepIndex < steps.length) {
                    progressMsg.textContent = steps[stepIndex].msg;
                    progressSubstep.textContent = steps[stepIndex].sub;
                    stepIndex++;
                    setTimeout(updateProgress, steps[stepIndex-1].delay);
                }
            }
            
            // Start progress simulation
            setTimeout(updateProgress, 400);
        });
    </script>
</body>
</html>
"""

RESULT_HTML = """
<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
    <title>Flash-cut Result</title>
    <link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css\">
    <style>
        body { background: #f8fafc; }
        .centered { max-width: 540px; margin: 2rem auto; background: #fff; border-radius: 16px; box-shadow: 0 2px 16px #0001; padding: 2rem; }
        .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
        .video-embed { width: 100%; max-width: 480px; aspect-ratio: 16/9; margin: 1.5rem auto; display: block; border-radius: 12px; box-shadow: 0 2px 8px #0002; }
        .summary-list { list-style: none; padding: 0; margin: 0 0 1.5rem 0; }
        .summary-list li { margin-bottom: 0.5rem; }
    </style>
</head>
<body>
    <main class=\"centered\">
        <h2 style=\"text-align:center;\">Flash-cut Result</h2>
        <ul class=\"summary-list\">
            <li><b>Onsets:</b> {{ num_onsets }}</li>
            <li><b>Segments:</b> {{ num_segments }}</li>
            <li><b>Flash cuts:</b> {{ num_flash }} ({{ flash_start }}–{{ flash_end }} s)</li>
            <li><b>FPS:</b> {{ fps }}</li>
        </ul>
        {% if rendered %}
            <video class=\"video-embed\" controls>
                <source src=\"{{ url_for('download', job_id=job_id, filename=output_name) }}\" type=\"video/mp4\">
                Your browser does not support the video tag.
            </video>
        {% else %}
            <p style=\"color:#f00;text-align:center;\">Video rendering failed or not available.</p>
        {% endif %}
        <p style=\"text-align:center;\"><a href=\"{{ url_for('index') }}\">← New analysis</a></p>
    </main>
</body>
</html>
"""
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB cap

# ---------------- helpers ----------------

def _ffmpeg_bin() -> str:
    # Try multiple common locations for FFmpeg
    ffmpeg_paths = [
        os.environ.get("FFMPEG"),
        shutil.which("ffmpeg"),
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/ffmpeg/bin/ffmpeg",
        "ffmpeg"
    ]
    
    # Filter out None values and return the first valid path
    for path in ffmpeg_paths:
        if path and (os.path.exists(path) or shutil.which(path)):
            return path
    
    # Default fallback
    return "ffmpeg"

def _ffprobe_bin() -> str:
    # Try multiple common locations for FFprobe
    ffprobe_paths = [
        os.environ.get("FFPROBE"),
        shutil.which("ffprobe"),
        "/usr/bin/ffprobe",
        "/usr/local/bin/ffprobe",
        "/opt/ffmpeg/bin/ffprobe",
        "ffprobe"
    ]
    
    # Filter out None values and return the first valid path
    for path in ffprobe_paths:
        if path and (os.path.exists(path) or shutil.which(path)):
            return path
    
    # Default fallback
    return "ffprobe"

def run_cmd(cmd: List[str], cwd: str | Path | None = None):
    """Run a subprocess and raise with captured logs on failure (friendlier Flask flash)."""
    try:
        # Print the command being executed for debugging
        print(f"Executing command: {' '.join(cmd)}")
        
        # Add PATH environment variable to help find executables
        env = os.environ.copy()
        extra_paths = ["/usr/local/bin", "/opt/ffmpeg/bin"]
        env["PATH"] = os.pathsep.join(extra_paths + [env.get("PATH", "")])
        
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True, env=env)
        
        if proc.returncode != 0:
            error_msg = (
                f"Command failed (exit {proc.returncode})\n"
                f"CMD: {' '.join(cmd)}\n"
                f"--- STDOUT ---\n{proc.stdout or ''}\n"
                f"--- STDERR ---\n{proc.stderr or ''}"
            )
            print(error_msg)  # Log to console/logs
            raise RuntimeError(error_msg)
        return proc
    except FileNotFoundError as e:
        error_msg = f"Command not found: {cmd[0]}\nMake sure FFmpeg is installed and in PATH.\nError: {str(e)}"
        print(error_msg)  # Log to console/logs
        raise RuntimeError(error_msg)

# ---------------- signal processing ----------------

def quantize_to_fps(times: List[float], fps: float) -> List[float]:
    return [round(t * fps) / fps for t in times]

def confidence_from_envelope(times, env, sr, hop):
    if len(times) == 0:
        return []
    frames = librosa.time_to_frames(times, sr=sr, hop_length=hop)
    frames = np.clip(frames, 0, len(env) - 1)
    vals = env[frames]
    scale = np.quantile(env, 0.98) or (env.max() or 1.0)
    return np.clip(vals / (scale if scale > 0 else 1.0), 0.0, 1.0).tolist()

def detect_onsets_flux(y: np.ndarray, sr: int, hop: int = FRAME_HOP, threshold: float = 0.30) -> List[dict]:
    _, y_perc = librosa.effects.hpss(y)
    env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop, aggregate=np.median)
    onset_times = librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=hop, units="time",
        backtrack=False, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.0
    )
    conf = confidence_from_envelope(onset_times, env, sr, hop)
    keep = [i for i, c in enumerate(conf) if c >= threshold]
    return [{"time": float(onset_times[i]), "confidence": float(conf[i])} for i in keep]

def detect_beats(audio_path: str, threshold: float):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))
    events = detect_onsets_flux(y, sr, hop=FRAME_HOP, threshold=threshold)
    return events, duration, sr, y

# ---------------- timeline building ----------------

def compute_intervals(beats: List[dict], duration: float, fps: float, max_gap: float):
    end = round(duration * fps) / fps
    if not beats:
        # chunk whole track into <= max_gap spans
        splits = [0.0]
        prev = 0.0
        while end - prev > max_gap:
            prev += max_gap
            splits.append(prev)
        splits.append(end)
        starts = [round(s, 3) for s in splits[:-1]]
        ends = [round(e, 3) for e in splits[1:]]
        return starts, ends
    beat_times = quantize_to_fps(sorted(float(b["time"]) for b in beats), fps)
    splits = [0.0]
    prev = 0.0
    first = beat_times[0]
    while first - prev > max_gap:
        prev += max_gap
        splits.append(prev)
    splits.append(first)
    for i in range(1, len(beat_times)):
        L, R = beat_times[i - 1], beat_times[i]
        prev = L
        while R - prev > max_gap:
            prev += max_gap
            splits.append(prev)
        splits.append(R)
    # ensure tail to full end
    if end > splits[-1]:
        prev = splits[-1]
        while end - prev > max_gap:
            prev += max_gap
            splits.append(prev)
        splits.append(end)
    starts = [round(s, 3) for s in splits[:-1]]
    ends = [round(e, 3) for e in splits[1:]]
    return starts, ends

# ---------------- flash window ----------------

def detect_flash_window(y: np.ndarray, sr: int, window: Tuple[float, float], min_gap: float, fps: float, threshold: float) -> List[float]:
    start_s, end_s = max(0.0, min(window)), max(0.0, max(window))
    i0, i1 = int(start_s * sr), int(end_s * sr)
    seg = y[i0:i1]
    if seg.size == 0:
        return []
    events = detect_onsets_flux(seg, sr, hop=FRAME_HOP, threshold=threshold)
    times = sorted([e["time"] + start_s for e in events])
    pruned, last = [], -1e9
    g = max(1.0 / fps, float(min_gap))
    for t in times:
        if t - last >= g:
            pruned.append(t)
            last = t
    return quantize_to_fps(pruned, fps)

# ---------------- rendering (PNG and VIDEO) ----------------

def render_from_images(pngs: List[str], starts: List[float], ends: List[float], audio: str, fps: float, out_path: str, aspect_ratio: str = "16:9") -> None:
    ffmpeg = _ffmpeg_bin()
    
    # Set dimensions based on aspect ratio
    aspect_map = {
        "16:9": (1280, 720),
        "1:1": (720, 720),
        "9:16": (720, 1280),
        "4:3": (960, 720),
    }
    target_w, target_h = aspect_map.get(aspect_ratio, (1280, 720))
    
    tmp = Path(out_path).parent / "_preconv"
    tmp.mkdir(parents=True, exist_ok=True)
    clip_paths = []
    for i, (s, e) in enumerate(zip(starts, ends), 1):
        length = max(1.0 / fps, e - s)
        src = pngs[(i - 1) % len(pngs)]
        out_i = tmp / f"seg_{i:04d}.mp4"
        cmd = [
            ffmpeg, "-y", "-loop", "1", "-t", f"{length:.3f}", "-i", src,
            "-vf", f"fps={int(fps)},scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            str(out_i),
        ]
        run_cmd(cmd)
        clip_paths.append(out_i)
    # concat list with explicit header + LF newlines (Windows friendly)
    list_file = tmp / "list.txt"
    list_text = "ffconcat version 1.0\n" + "\n".join(f"file '{p.name}'" for p in clip_paths) + "\n"
    list_file.write_text(list_text, encoding="utf-8", newline="\n")
    concat_out = tmp / "video.mp4"
    run_cmd([
        ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", "list.txt",
        "-fflags", "+genpts", "-r", str(int(fps)), "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart", str(concat_out),
    ], cwd=tmp)
    run_cmd([ffmpeg, "-y", "-i", str(concat_out), "-i", audio, "-c:v", "copy", "-c:a", "aac", "-shortest", out_path])


def probe_video_meta(path: str):
    ffprobe = _ffprobe_bin()
    try:
        proc = subprocess.run([
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration", "-show_entries", "format=duration",
            "-of", "json", path,
        ], check=True, capture_output=True, text=True)
        data = json.loads(proc.stdout or "{}")
        w = h = None
        dur = None
        if data.get("streams"):
            s0 = data["streams"][0]
            w = int(s0.get("width") or 0) or None
            h = int(s0.get("height") or 0) or None
            if s0.get("duration"):
                try: dur = float(s0.get("duration"))
                except Exception: pass
        if data.get("format", {}).get("duration"):
            try: dur = float(data["format"]["duration"]) or dur
            except Exception: pass
        if not w or not h: w, h = 1280, 720
        if dur is None: dur = 0.0
        return w, h, float(dur)
    except Exception:
        return 1280, 720, 0.0


def render_from_videos(videos: List[str], starts: List[float], ends: List[float], audio: str, fps: float, out_path: str, clip_mode: str = "head", aspect_ratio: str = "16:9") -> None:
    if not videos:
        raise RuntimeError("No video files provided")
    ffmpeg = _ffmpeg_bin()
    
    # Set dimensions based on aspect ratio
    aspect_map = {
        "16:9": (1280, 720),
        "1:1": (720, 720),
        "9:16": (720, 1280),
        "4:3": (960, 720),
    }
    target_w, target_h = aspect_map.get(aspect_ratio, (1280, 720))

    tmp = Path(out_path).parent / "_preconv"
    tmp.mkdir(parents=True, exist_ok=True)
    clip_paths: List[Path] = []

    for i, (s, e) in enumerate(zip(starts, ends), 1):
        length = max(1.0 / fps, e - s)
        src = videos[(i - 1) % len(videos)]
        _, _, dur = probe_video_meta(src)
        out_i = tmp / f"seg_{i:04d}.mp4"
        if dur > 0 and length <= dur:
            ss = max(dur - length, 0.0) if clip_mode == "tail" else 0.0
            cmd = [
                ffmpeg, "-y", "-ss", f"{ss:.3f}", "-t", f"{length:.3f}", "-i", src,
                "-vf", f"fps={int(fps)},scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black",
                "-an", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                str(out_i),
            ]
        else:
            cmd = [
                ffmpeg, "-y", "-stream_loop", "-1", "-t", f"{length:.3f}", "-i", src,
                "-vf", f"fps={int(fps)},scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black",
                "-an", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                str(out_i),
            ]
        run_cmd(cmd)
        clip_paths.append(out_i)

    # Concat (Windows-safe)
    list_file = tmp / "list.txt"
    list_text = "ffconcat version 1.0\n" + "\n".join(f"file '{p.name}'" for p in clip_paths) + "\n"
    list_file.write_text(list_text, encoding="utf-8", newline="\n")
    concat_out = tmp / "video.mp4"
    run_cmd([
        ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", "list.txt",
        "-fflags", "+genpts", "-r", str(int(fps)), "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart", str(concat_out),
    ], cwd=tmp)

    # Mux with audio
    run_cmd([ffmpeg, "-y", "-i", str(concat_out), "-i", audio, "-c:v", "copy", "-c:a", "aac", "-shortest", out_path])

# ---------------- routes ----------------
@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/analyze", methods=["POST"])
def analyze():
    audio_file = request.files.get("audio")
    if not audio_file or audio_file.filename == "":
        flash("Please upload an audio file.")
        return redirect(url_for("index"))

    fps = float(request.form.get("fps", 30))
    threshold = float(request.form.get("threshold", 0.30))
    max_gap = float(request.form.get("max_gap", 5.0))
    flash_start = float(request.form.get("flash_start", 10.0))
    flash_end = float(request.form.get("flash_end", 25.0))
    flash_gap = float(request.form.get("flash_gap", 0.12))
    do_render = request.form.get("do_render") == "1"
    clip_mode = (request.form.get("clip_mode", "head") or "head").lower()
    if clip_mode not in {"head", "tail"}:
        clip_mode = "head"
    output_name = (request.form.get("output_name", "final_video.mp4") or "final_video.mp4").strip()
    aspect_ratio = request.form.get("aspect_ratio", "16:9")

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    audio_path = job_dir / audio_file.filename
    audio_file.save(str(audio_path))

    try:
        events, duration, sr, y = detect_beats(str(audio_path), threshold=threshold)
    except Exception as e:
        flash("Failed to read audio. If you uploaded MP3, enable FFmpeg on Railway or upload a WAV/FLAC/OGG file.\n" + str(e))
        return redirect(url_for("index"))
    starts, ends = compute_intervals(events, duration, fps, max_gap)

    flash_times = detect_flash_window(y, sr, (flash_start, flash_end), flash_gap, fps, threshold) if flash_end > flash_start else []
    if flash_times:
        # If we later remove flash window UI, leave this here behind a feature flag
        starts, ends = inject_flash_splits(starts, ends, flash_times, fps)

    data = {
        "audio": audio_file.filename,
        "fps": fps,
        "max_gap": max_gap,
        "events_onsets": events,
        "segments": [{"start": s, "end": e} for s, e in zip(starts, ends)],
        "flash": flash_times,
        "flash_window": [flash_start, flash_end],
        "clip_mode": clip_mode,
    }
    (job_dir / "cuts.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    (job_dir / "cuts.csv").write_text(
        "index,start,end\n" + "\n".join(f"{i+1},{s:.3f},{e:.3f}" for i, (s, e) in enumerate(zip(starts, ends))),
        encoding="utf-8",
    )

    plot_waveform(job_dir / "waveform.png", y, sr, flash_times, (flash_start, flash_end))

    rendered = False
    # Always render video (default checked)
    do_render = True
    if do_render:
        saved_videos: List[str] = []
        for f in request.files.getlist("videos"):
            if f and f.filename:
                dst = job_dir / f.filename
                f.save(str(dst))
                saved_videos.append(str(dst))

        if saved_videos:
            out_path = job_dir / output_name
            try:
                render_from_videos(saved_videos, starts, ends, str(audio_path), fps, str(out_path), clip_mode=clip_mode, aspect_ratio=aspect_ratio)
                rendered = True
            except Exception as e:
                flash(f"ffmpeg video render failed:\n{e}")
        else:
            # PNG fallback
            images = request.files.getlist("images")
            pngs: List[str] = []
            for f in images:
                if f and f.filename.lower().endswith(".png"):
                    dst = job_dir / f.filename
                    f.save(str(dst))
                    pngs.append(str(dst))
            if pngs:
                out_path = job_dir / output_name
                try:
                    render_from_images(pngs, starts, ends, str(audio_path), fps, str(out_path), aspect_ratio=aspect_ratio)
                    rendered = True
                except Exception as e:
                    flash(f"ffmpeg image render failed:\n{e}")
            else:
                flash("No videos or PNGs were provided for rendering.")

    # No CSV, PNG, or JSON links, no waveform or segments preview
    return render_template_string(
        RESULT_HTML,
        job_id=job_id,
        fps=fps, threshold=threshold, max_gap=max_gap,
        flash_start=flash_start, flash_end=flash_end,
        num_onsets=len(events), num_segments=len(starts), num_flash=len(flash_times),
        rendered=rendered, output_name=output_name
    )

@app.route("/jobs/<job_id>/<path:filename>")
def download(job_id, filename):
    folder = JOBS_DIR / job_id
    if not folder.exists():
        return "Not found", 404
    return send_from_directory(folder, filename, as_attachment=True)

def inject_flash_splits(starts: List[float], ends: List[float], flash_times: List[float], fps: float):
    if not flash_times:
        return starts, ends
    flash = sorted(quantize_to_fps(flash_times, fps))
    out_s, out_e = [], []
    for s, e in zip(starts, ends):
        cuts = [t for t in flash if s < t < e]
        if not cuts:
            out_s.append(s)
            out_e.append(e)
            continue
        pts = [s] + cuts + [e]
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if b - a < 1.0 / fps:
                b = a + 1.0 / fps
            out_s.append(round(a, 3))
            out_e.append(round(b, 3))
    return out_s, out_e


def plot_waveform(png_path: Path, y: np.ndarray, sr: int, flash_times: List[float], window: Tuple[float, float]):
    t = np.linspace(0, librosa.get_duration(y=y, sr=sr), num=len(y), endpoint=True)
    plt.figure(figsize=(18, 4))
    plt.fill_between(t, y, -y, color="#f0b429", alpha=0.25)
    plt.plot(t, y, color="#f0b429", lw=0.7, alpha=0.8)
    lo, hi = min(window), max(window)
    plt.axvline(lo, color="red", ls="--", lw=2, dashes=(6, 6))
    plt.axvline(hi, color="red", ls="--", lw=2, dashes=(6, 6))
    for x in flash_times:
        plt.axvline(x, color="#22c55e", ls=(0, (3, 5)), lw=1.4, alpha=0.9)
    plt.title("Waveform with Flash Cut Points")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.25, ls="--")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

# --- keep everything above as-is ---


# --- Enhanced loading feedback (JS) ---
# (Already handled in INDEX_HTML with spinner overlay. For stepwise feedback, would require AJAX or WebSocket.)

@app.route("/health")
def health():
    return {"ok": True}

@app.route("/ffmpeg-check")
def ffmpeg_check():
    """Diagnostic endpoint to check FFmpeg availability and version."""
    ffmpeg = _ffmpeg_bin()
    ffprobe = _ffprobe_bin()
    
    results = {
        "ffmpeg_path": ffmpeg,
        "ffprobe_path": ffprobe,
        "environment": dict(os.environ),
        "path": os.environ.get("PATH", ""),
        "cwd": os.getcwd(),
        "timestamp": datetime.now().isoformat()
    }
    
    # Try to get versions
    try:
        proc = subprocess.run([ffmpeg, "-version"], text=True, capture_output=True)
        results["ffmpeg_version"] = proc.stdout.strip() if proc.returncode == 0 else "Error running ffmpeg"
        results["ffmpeg_available"] = proc.returncode == 0
    except Exception as e:
        results["ffmpeg_error"] = str(e)
        results["ffmpeg_available"] = False
    
    try:
        proc = subprocess.run([ffprobe, "-version"], text=True, capture_output=True)
        results["ffprobe_version"] = proc.stdout.strip() if proc.returncode == 0 else "Error running ffprobe"
        results["ffprobe_available"] = proc.returncode == 0
    except Exception as e:
        results["ffprobe_error"] = str(e)
        results["ffprobe_available"] = False
    
    return results

if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")  # headless on Railway
    
    # Print diagnostic info at startup
    ffmpeg = _ffmpeg_bin()
    ffprobe = _ffprobe_bin()
    print(f"Using FFmpeg: {ffmpeg}")
    print(f"Using FFprobe: {ffprobe}")
    print(f"Current PATH: {os.environ.get('PATH', '')}")
    
    try:
        # Test FFmpeg at startup
        proc = subprocess.run([ffmpeg, "-version"], text=True, capture_output=True)
        if proc.returncode == 0:
            print(f"FFmpeg version: {proc.stdout.splitlines()[0] if proc.stdout else 'Unknown'}")
        else:
            print(f"FFmpeg test failed with exit code {proc.returncode}")
            print(f"Error: {proc.stderr}")
    except Exception as e:
        print(f"Failed to run FFmpeg: {str(e)}")
    
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)

