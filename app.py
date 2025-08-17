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
        .spinner-overlay { display: none; position: fixed; z-index: 9999; inset: 0; background: rgba(255,255,255,0.8); align-items: center; justify-content: center; flex-direction: column; }
        .spinner { border: 6px solid #eee; border-top: 6px solid #2563eb; border-radius: 50%; width: 48px; height: 48px; animation: spin 1s linear infinite; margin-bottom: 1.5rem; }
        .progress-msg { font-size: 1.2rem; font-weight: 500; color: #1e3a8a; text-align: center; max-width: 320px; }
        .progress-substep { font-size: 0.9rem; color: #64748b; margin-top: 0.5rem; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
        .or-divider { text-align: center; position: relative; margin: 1.5rem 0; }
        .or-divider::before { content: ''; position: absolute; left: 0; top: 50%; width: 45%; height: 1px; background: #ddd; }
        .or-divider::after { content: ''; position: absolute; right: 0; top: 50%; width: 45%; height: 1px; background: #ddd; }
        .nav-buttons { display: flex; justify-content: center; gap: 1rem; margin-top: 2rem; }
        .nav-buttons a { flex: 1; text-align: center; }
    </style>
</head>
<body>
    <main class=\"centered\">
        <img src=\"https://cdn.jsdelivr.net/gh/tabler/tabler-icons/icons/music.svg\" class=\"logo\" alt=\"logo\">
        <h2 style=\"text-align:center;\">Flash-cut Builder</h2>
        {% with messages = get_flashed_messages() %}
            {% if messages %}<article>{% for m in messages %}<p>{{m}}</p>{% endfor %}</article>{% endif %}
        {% endwith %}
        
        <form id=\"audioForm\" action=\"{{ url_for('upload_audio') }}\" method=\"post\" enctype=\"multipart/form-data\">
            <div class=\"form-group\">
                <label for=\"audio\"><b>Upload Audio Track</b></label>
                <input type=\"file\" id=\"audio\" name=\"audio\" accept=\".mp3,.wav,.flac,.ogg,.m4a,audio/*\">
                <small>Supported formats: MP3, WAV, FLAC, OGG, M4A</small>
            </div>
            
            <div class=\"or-divider\">OR</div>
            
            <div class=\"form-group\">
                <label for=\"video_for_audio\"><b>Extract Audio from Video</b></label>
                <input type=\"file\" id=\"video_for_audio\" name=\"video_for_audio\" accept=\"video/*\">
                <small>Supported formats: MP4, MOV, etc. (any video with audio track)</small>
            </div>
            
            <button type=\"submit\" style=\"width:100%;font-size:1.2rem;margin-top:1rem;\">Analyze Audio</button>
        </form>
        
        <div class=\"spinner-overlay\" id=\"spinnerOverlay\">
            <div class=\"spinner\"></div>
            <div class=\"progress-msg\" id=\"progressMsg\">Processing audio file...</div>
            <div class=\"progress-substep\" id=\"progressSubstep\">Analyzing waveform</div>
        </div>
    </main>
    <script>
        // Form validation - ensure either audio or video is uploaded
        document.getElementById('audioForm').addEventListener('submit', function(e) {
            const audioFile = document.getElementById('audio').files.length;
            const videoFile = document.getElementById('video_for_audio').files.length;
            
            if (audioFile === 0 && videoFile === 0) {
                e.preventDefault();
                alert('Please upload either an audio file or a video to extract audio from.');
                return false;
            }
            
            // Show spinner during processing
            document.getElementById('spinnerOverlay').style.display = 'flex';
            
            // If video is selected, update the progress message
            if (videoFile > 0) {
                document.getElementById('progressMsg').textContent = 'Extracting audio from video...';
                document.getElementById('progressSubstep').textContent = 'This may take a moment';
            }
        });
        
        // When audio is selected, clear video selection and vice versa
        document.getElementById('audio').addEventListener('change', function() {
            if (this.files.length > 0) {
                document.getElementById('video_for_audio').value = '';
            }
        });
        
        document.getElementById('video_for_audio').addEventListener('change', function() {
            if (this.files.length > 0) {
                document.getElementById('audio').value = '';
            }
        });
    </script>
</body>
</html>
"""

ANALYSIS_RESULT_HTML = """
<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
    <title>Flash-cut - Analysis Results</title>
    <link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css\">
    <style>
        body { padding-block: 1.5rem; background: #f8fafc; }
        .centered { max-width: 480px; margin: 2rem auto; background: #fff; border-radius: 16px; box-shadow: 0 2px 16px #0001; padding: 2rem; }
        .logo { display: block; margin: 0 auto 1.5rem; width: 64px; }
        .form-group { margin-bottom: 1.5rem; }
        .audio-info { background: #f1f5f9; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem; }
        .audio-info h4 { margin-top: 0; }
        .step-indicator { display: flex; justify-content: center; margin-bottom: 2rem; }
        .step { width: 2rem; height: 2rem; border-radius: 50%; background: #e2e8f0; color: #64748b; display: flex; align-items: center; justify-content: center; font-weight: bold; margin: 0 0.5rem; }
        .step.active { background: #2563eb; color: white; }
        .step.completed { background: #10b981; color: white; }
        .waveform-img { width: 100%; height: auto; border-radius: 8px; margin: 1rem 0; }
        .next-btn { display: block; width: 100%; font-size: 1.2rem; margin-top: 1.5rem; }
    </style>
</head>
<body>
    <main class=\"centered\">
        <img src=\"https://cdn.jsdelivr.net/gh/tabler/tabler-icons/icons/chart-line.svg\" class=\"logo\" alt=\"logo\">
        <h2 style=\"text-align:center;\">Audio Analysis</h2>
        
        <div class=\"step-indicator\">
            <div class=\"step completed\">1</div>
            <div class=\"step active\">2</div>
            <div class=\"step\">3</div>
        </div>
        
        {% with messages = get_flashed_messages() %}
            {% if messages %}<article>{% for m in messages %}<p>{{m}}</p>{% endfor %}</article>{% endif %}
        {% endwith %}
        
        <div class=\"audio-info\">
            <h4>Analysis Complete</h4>
            <p><b>File:</b> {{ audio_filename }}</p>
            <p><b>Source:</b> {{ audio_source }}</p>
            <p><b>Duration:</b> {{ duration }} seconds</p>
            <p><b>Segments found:</b> {{ num_segments }}</p>
        </div>
        
        <h4>Waveform Preview</h4>
        <img src=\"{{ url_for('download', job_id=job_id, filename='waveform.png') }}\" alt=\"Audio waveform\" class=\"waveform-img\">
        
        <a href=\"{{ url_for('upload_media_page', job_id=job_id) }}\" class=\"next-btn\" role=\"button\">Continue to Upload Media</a>
    </main>
</body>
</html>
"""

UPLOAD_VISUAL_HTML = """
<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
    <title>Flash-cut - Upload Media</title>
    <link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css\">
    <style>
        body { padding-block: 1.5rem; background: #f8fafc; }
        .centered { max-width: 480px; margin: 2rem auto; background: #fff; border-radius: 16px; box-shadow: 0 2px 16px #0001; padding: 2rem; }
        .logo { display: block; margin: 0 auto 1.5rem; width: 64px; }
        .form-group { margin-bottom: 1.5rem; }
        .spinner-overlay { display: none; position: fixed; z-index: 9999; inset: 0; background: rgba(255,255,255,0.8); align-items: center; justify-content: center; flex-direction: column; }
        .spinner { border: 6px solid #eee; border-top: 6px solid #2563eb; border-radius: 50%; width: 48px; height: 48px; animation: spin 1s linear infinite; margin-bottom: 1.5rem; }
        .progress-msg { font-size: 1.2rem; font-weight: 500; color: #1e3a8a; text-align: center; max-width: 320px; }
        .progress-substep { font-size: 0.9rem; color: #64748b; margin-top: 0.5rem; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
        .audio-info { background: #f1f5f9; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem; }
        .audio-info h4 { margin-top: 0; }
        .step-indicator { display: flex; justify-content: center; margin-bottom: 2rem; }
        .step { width: 2rem; height: 2rem; border-radius: 50%; background: #e2e8f0; color: #64748b; display: flex; align-items: center; justify-content: center; font-weight: bold; margin: 0 0.5rem; }
        .step.active { background: #2563eb; color: white; }
        .step.completed { background: #10b981; color: white; }
        .media-preview { display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 10px; margin-top: 15px; }
        .media-thumbnail { width: 100%; height: 100px; object-fit: cover; border-radius: 6px; }
    </style>
</head>
<body>
    <main class=\"centered\">
        <img src=\"https://cdn.jsdelivr.net/gh/tabler/tabler-icons/icons/video.svg\" class=\"logo\" alt=\"logo\">
        <h2 style=\"text-align:center;\">Upload Media</h2>
        
        <div class=\"step-indicator\">
            <div class=\"step completed\">1</div>
            <div class=\"step completed\">2</div>
            <div class=\"step active\">3</div>
        </div>
        
        <div class=\"audio-info\">
            <h4>Audio Information</h4>
            <p><b>File:</b> {{ audio_filename }}</p>
            <p><b>Segments:</b> {{ num_segments }}</p>
        </div>
        
        {% with messages = get_flashed_messages() %}
            {% if messages %}<article>{% for m in messages %}<p>{{m}}</p>{% endfor %}</article>{% endif %}
        {% endwith %}
        
        <form id=\"visualForm\" action=\"{{ url_for('preview_media', job_id=job_id) }}\" method=\"post\" enctype=\"multipart/form-data\">
            <div class=\"form-group\">
                <label><b>Video Clips</b> (multiple allowed)</label>
                <input type=\"file\" id=\"videos\" name=\"videos\" accept=\"video/*\" multiple>
            </div>
            <div class=\"form-group\">
                <label><b>Or PNG Images</b> (multiple allowed)</label>
                <input type=\"file\" id=\"images\" name=\"images\" accept=\"image/png\" multiple>
            </div>
            
            <div class=\"form-group\">
                <label><b>Aspect Ratio</b></label>
                <select name=\"aspect_ratio\">
                    <option value=\"16:9\" selected>16:9 (widescreen)</option>
                    <option value=\"1:1\">1:1 (square)</option>
                    <option value=\"9:16\">9:16 (vertical)</option>
                    <option value=\"4:3\">4:3</option>
                </select>
            </div>
            
            <div id=\"mediaPreview\" class=\"media-preview\">
                <!-- Thumbnails will appear here -->
            </div>
            
            <button type=\"submit\" style=\"width:100%;font-size:1.2rem;margin-top:1.5rem;\">Preview Media</button>
        </form>
    </main>
    <script>
        // Function to handle image preview
        function handleImagePreview(event, fileInput) {
            const files = fileInput.files;
            const preview = document.getElementById('mediaPreview');
            
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                if (file.type.startsWith('image/')) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const img = document.createElement('img');
                        img.src = e.target.result;
                        img.className = 'media-thumbnail';
                        preview.appendChild(img);
                    }
                    reader.readAsDataURL(file);
                } else if (file.type.startsWith('video/')) {
                    // For videos, create a video element and get a thumbnail
                    const video = document.createElement('video');
                    video.preload = 'metadata';
                    video.onloadedmetadata = function() {
                        // Create a canvas to capture a frame
                        video.currentTime = 1; // Seek to 1 second
                        video.onseeked = function() {
                            const canvas = document.createElement('canvas');
                            canvas.width = video.videoWidth;
                            canvas.height = video.videoHeight;
                            const ctx = canvas.getContext('2d');
                            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                            
                            const img = document.createElement('img');
                            img.src = canvas.toDataURL();
                            img.className = 'media-thumbnail';
                            preview.appendChild(img);
                        }
                    }
                    video.src = URL.createObjectURL(file);
                }
            }
        }
        
        // Set up event listeners for both file inputs
        document.getElementById('videos').addEventListener('change', function(event) {
            handleImagePreview(event, this);
        });
        
        document.getElementById('images').addEventListener('change', function(event) {
            handleImagePreview(event, this);
        });
    </script>
</body>
</html>
"""

RENDER_OPTIONS_HTML = """
<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
    <title>Flash-cut - Rendering Options</title>
    <link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css\">
    <style>
        body { padding-block: 1.5rem; background: #f8fafc; }
        .centered { max-width: 480px; margin: 2rem auto; background: #fff; border-radius: 16px; box-shadow: 0 2px 16px #0001; padding: 2rem; }
        .logo { display: block; margin: 0 auto 1.5rem; width: 64px; }
        .form-group { margin-bottom: 1.5rem; }
        .spinner-overlay { display: none; position: fixed; z-index: 9999; inset: 0; background: rgba(255,255,255,0.8); align-items: center; justify-content: center; flex-direction: column; }
        .spinner { border: 6px solid #eee; border-top: 6px solid #2563eb; border-radius: 50%; width: 48px; height: 48px; animation: spin 1s linear infinite; margin-bottom: 1.5rem; }
        .progress-msg { font-size: 1.2rem; font-weight: 500; color: #1e3a8a; text-align: center; max-width: 320px; }
        .progress-substep { font-size: 0.9rem; color: #64748b; margin-top: 0.5rem; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
        .audio-info { background: #f1f5f9; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem; }
        .audio-info h4 { margin-top: 0; }
        .step-indicator { display: flex; justify-content: center; margin-bottom: 2rem; }
        .step { width: 2rem; height: 2rem; border-radius: 50%; background: #e2e8f0; color: #64748b; display: flex; align-items: center; justify-content: center; font-weight: bold; margin: 0 0.5rem; }
        .step.active { background: #2563eb; color: white; }
        .step.completed { background: #10b981; color: white; }
        .media-preview { display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 10px; margin: 15px 0; }
        .media-thumbnail { width: 100%; height: 100px; object-fit: cover; border-radius: 6px; }
        .progress-container { width: 100%; height: 24px; background: #e2e8f0; border-radius: 12px; overflow: hidden; margin-bottom: 1rem; }
        .progress-bar { height: 100%; width: 0%; background: #2563eb; transition: width 0.5s; }
    </style>
</head>
<body>
    <main class=\"centered\">
        <img src=\"https://cdn.jsdelivr.net/gh/tabler/tabler-icons/icons/player-play.svg\" class=\"logo\" alt=\"logo\">
        <h2 style=\"text-align:center;\">Ready to Render</h2>
        
        <div class=\"step-indicator\">
            <div class=\"step completed\">1</div>
            <div class=\"step completed\">2</div>
            <div class=\"step completed\">3</div>
        </div>
        
        {% with messages = get_flashed_messages() %}
            {% if messages %}<article>{% for m in messages %}<p>{{m}}</p>{% endfor %}</article>{% endif %}
        {% endwith %}
        
        <div class=\"audio-info\">
            <h4>Media Ready</h4>
            <p><b>Audio:</b> {{ audio_filename }}</p>
            <p><b>Segments:</b> {{ num_segments }}</p>
            <p><b>Aspect Ratio:</b> {{ aspect_ratio }}</p>
            <p><b>Media files:</b> {{ media_files|length }}</p>
        </div>
        
        <h4>Media Preview</h4>
        <div class=\"media-preview\">
            {% for file in media_files %}
                {% if file.type == 'image' %}
                    <img src=\"{{ url_for('download', job_id=job_id, filename=file.filename) }}\" alt=\"Image\" class=\"media-thumbnail\">
                {% else %}
                    <div class=\"media-thumbnail\" style=\"display: flex; justify-content: center; align-items: center; background: #000;\">
                        <span style=\"font-size: 24px; color: #fff;\">▶️</span>
                    </div>
                {% endif %}
            {% endfor %}
        </div>
        
        <form id=\"renderForm\" action=\"{{ url_for('render_video', job_id=job_id) }}\" method=\"post\" enctype=\"multipart/form-data\">
            <div class=\"form-group\">
                <label><b>Clip Portion</b></label>
                <select name=\"clip_mode\">
                    <option value=\"head\" selected>Head (start)</option>
                    <option value=\"tail\">Tail (end)</option>
                </select>
            </div>
            <div class=\"form-group\">
                <label><b>Output File Name</b></label>
                <input type=\"text\" name=\"output_name\" value=\"final_video.mp4\">
            </div>
            <button type=\"submit\" style=\"width:100%;font-size:1.2rem;margin-top:1rem;\">Render Final Video</button>
        </form>
        
        <div class=\"spinner-overlay\" id=\"spinnerOverlay\">
            <div class=\"spinner\"></div>
            <div class=\"progress-msg\" id=\"progressMsg\">Rendering video...</div>
            <div class=\"progress-substep\" id=\"progressSubstep\">Processing clips</div>
            <div class=\"progress-container\">
                <div class=\"progress-bar\" id=\"progressBar\"></div>
            </div>
        </div>
    </main>
    <script>
        document.getElementById('renderForm').addEventListener('submit', function() {
            // Show spinner overlay
            document.getElementById('spinnerOverlay').style.display = 'flex';
            
            // Simulate progress updates
            const progressMsg = document.getElementById('progressMsg');
            const progressSubstep = document.getElementById('progressSubstep');
            const progressBar = document.getElementById('progressBar');
            
            const steps = [
                {msg: 'Preparing media files...', sub: 'Processing uploaded content', delay: 1000, progress: 10},
                {msg: 'Rendering segments...', sub: 'Creating video clips', delay: 2000, progress: 30},
                {msg: 'Assembling video...', sub: 'Combining segments', delay: 1500, progress: 60},
                {msg: 'Finalizing output...', sub: 'Muxing audio and video', delay: 1500, progress: 90},
            ];
            
            let stepIndex = 0;
            function updateProgress() {
                if (stepIndex < steps.length) {
                    progressMsg.textContent = steps[stepIndex].msg;
                    progressSubstep.textContent = steps[stepIndex].sub;
                    progressBar.style.width = steps[stepIndex].progress + '%';
                    
                    stepIndex++;
                    setTimeout(updateProgress, steps[stepIndex-1].delay);
                } else {
                    // Final state
                    progressBar.style.width = '100%';
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
    <title>Flash-cut Complete</title>
    <link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css\">
    <style>
        body { background: #f8fafc; }
        .centered { max-width: 540px; margin: 2rem auto; background: #fff; border-radius: 16px; box-shadow: 0 2px 16px #0001; padding: 2rem; }
        .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
        .video-embed { width: 100%; max-width: 480px; aspect-ratio: 16/9; margin: 1.5rem auto; display: block; border-radius: 12px; box-shadow: 0 2px 8px #0002; }
        .summary-list { list-style: none; padding: 0; margin: 0 0 1.5rem 0; }
        .summary-list li { margin-bottom: 0.5rem; }
        .download-btn { display: block; text-align: center; margin-top: 1.5rem; text-decoration: none; }
        .step-indicator { display: flex; justify-content: center; margin-bottom: 2rem; }
        .step { width: 2rem; height: 2rem; border-radius: 50%; background: #e2e8f0; color: #64748b; display: flex; align-items: center; justify-content: center; font-weight: bold; margin: 0 0.5rem; }
        .step.completed { background: #10b981; color: white; }
        .confetti { position: fixed; z-index: -1; }
    </style>
</head>
<body>
    <main class=\"centered\">
        <h2 style=\"text-align:center;\">Video Complete! 🎉</h2>
        
        <div class=\"step-indicator\">
            <div class=\"step completed\">1</div>
            <div class=\"step completed\">2</div>
            <div class=\"step completed\">3</div>
        </div>
        
        {% with messages = get_flashed_messages() %}
            {% if messages %}<article>{% for m in messages %}<p>{{m}}</p>{% endfor %}</article>{% endif %}
        {% endwith %}
        
        <ul class=\"summary-list\">
            <li><b>Segments created:</b> {{ num_segments }}</li>
            <li><b>Video FPS:</b> {{ fps }}</li>
        </ul>
        
        {% if rendered %}
            <video class=\"video-embed\" controls>
                <source src=\"{{ url_for('download', job_id=job_id, filename=output_name) }}\" type=\"video/mp4\">
                Your browser does not support the video tag.
            </video>
            <a href=\"{{ url_for('download', job_id=job_id, filename=output_name) }}\" class=\"download-btn\" download>Download Video</a>
            <a href=\"{{ url_for('download', job_id=job_id, filename='cuts.csv') }}\" class=\"download-btn\" download>Download Segments CSV</a>
        {% else %}
            <p style=\"color:#f00;text-align:center;\">Video rendering failed. Please try again with different visual content.</p>
        {% endif %}
        
        <p style=\"text-align:center;margin-top:2rem;\"><a href=\"{{ url_for('index') }}\">← Create a new video</a></p>
    </main>
    
    <script>
        // Simple confetti effect for completion
        function createConfetti() {
            const colors = ['#f94144', '#f3722c', '#f8961e', '#f9c74f', '#90be6d', '#43aa8b', '#577590'];
            const confettiCount = 150;
            
            for (let i = 0; i < confettiCount; i++) {
                const confetti = document.createElement('div');
                confetti.className = 'confetti';
                confetti.style.left = Math.random() * 100 + 'vw';
                confetti.style.top = -20 + 'px';
                confetti.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
                confetti.style.width = Math.random() * 10 + 5 + 'px';
                confetti.style.height = Math.random() * 10 + 5 + 'px';
                confetti.style.opacity = Math.random() + 0.5;
                confetti.style.transform = 'rotate(' + Math.random() * 360 + 'deg)';
                
                document.body.appendChild(confetti);
                
                const fallDuration = Math.random() * 3 + 2;
                const swayDuration = Math.random() * 2 + 2;
                
                confetti.animate([
                    { transform: `translate3d(0, 0, 0) rotate(0deg)`, opacity: 1 },
                    { transform: `translate3d(${(Math.random() - 0.5) * 200}px, ${window.innerHeight}px, 0) rotate(${Math.random() * 360}deg)`, opacity: 0 }
                ], {
                    duration: fallDuration * 1000,
                    easing: 'cubic-bezier(0.4, 0.0, 0.2, 1)',
                    fill: 'forwards'
                });
                
                setTimeout(() => {
                    confetti.remove();
                }, fallDuration * 1000);
            }
        }
        
        // Only run confetti if the video rendered successfully
        {% if rendered %}
        window.addEventListener('load', createConfetti);
        {% endif %}
    </script>
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

def extract_audio_from_video(video_path, output_audio_path):
    """Extract audio track from a video file using FFmpeg."""
    ffmpeg = _ffmpeg_bin()
    cmd = [
        ffmpeg,
        "-i", str(video_path),
        "-vn",                 # No video
        "-acodec", "libmp3lame",  # MP3 codec
        "-q:a", "2",           # Quality (lower is better)
        "-y",                  # Overwrite output
        str(output_audio_path)
    ]
    
    try:
        print(f"Extracting audio from video: {video_path}")
        run_cmd(cmd)
        return True
    except Exception as e:
        print(f"Failed to extract audio: {e}")
        return False

# Helper function to run shell commands

def run_cmd(cmd, cwd=None):
    """Run a command using subprocess.run and raise an error if it fails."""
    import subprocess
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)

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

@app.route("/upload-audio", methods=["POST"])
def upload_audio():
    audio_file = request.files.get("audio")
    video_file = request.files.get("video_for_audio")
    
    if not audio_file or audio_file.filename == "":
        if not video_file or video_file.filename == "":
            flash("Please upload either an audio file or a video to extract audio from.")
            return redirect(url_for("index"))
    
    # Fixed values for simplified UX
    fps = 30  # Default fixed at 30 FPS
    threshold = 0.30
    max_gap = 5.0
    flash_start = 10.0
    flash_end = 25.0
    flash_gap = 0.12

    # Create job directory
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Process the uploaded files
    if audio_file and audio_file.filename != "":
        # Save audio file directly
        audio_filename = audio_file.filename
        audio_path = job_dir / audio_filename
        audio_file.save(str(audio_path))
    else:
        # Extract audio from video
        video_filename = video_file.filename
        video_path = job_dir / video_filename
        video_file.save(str(video_path))
        
        # Generate audio filename based on the video name
        audio_filename = os.path.splitext(video_filename)[0] + ".mp3"
        audio_path = job_dir / audio_filename
        
        # Extract the audio
        if not extract_audio_from_video(video_path, audio_path):
            flash("Failed to extract audio from the video. Please try a different file.")
            return redirect(url_for("index"))

    try:
        # Analyze audio
        events, duration, sr, y = detect_beats(str(audio_path), threshold=threshold)
        starts, ends = compute_intervals(events, duration, fps, max_gap)

        # Add flash times if needed
        flash_times = detect_flash_window(y, sr, (flash_start, flash_end), flash_gap, fps, threshold) if flash_end > flash_start else []
        if flash_times:
            starts, ends = inject_flash_splits(starts, ends, flash_times, fps)

        # Save data
        data = {
            "audio": audio_filename,
            "fps": fps,
            "max_gap": max_gap,
            "events_onsets": events,
            "segments": [{"start": s, "end": e} for s, e in zip(starts, ends)],
            "flash": flash_times,
            "flash_window": [flash_start, flash_end]
        }
        (job_dir / "cuts.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        
        # Create CSV file for download
        (job_dir / "cuts.csv").write_text(
            "index,start,end\n" + "\n".join(f"{i+1},{s:.3f},{e:.3f}" for i, (s, e) in enumerate(zip(starts, ends))),
            encoding="utf-8",
        )
        
        # Create waveform visualization
        plot_waveform(job_dir / "waveform.png", y, sr, flash_times, (flash_start, flash_end))
        
        # Record the source of the audio (direct upload or extracted from video)
        if video_file and video_file.filename != "":
            data["audio_source"] = "extracted_from_video"
            data["original_video"] = video_filename
        else:
            data["audio_source"] = "direct_upload"
        
        (job_dir / "cuts.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        
        # Redirect to analysis results page
        return render_template_string(
            ANALYSIS_RESULT_HTML,
            job_id=job_id,
            audio_filename=audio_filename,
            duration=round(duration, 1),
            num_segments=len(starts),
            audio_source="Video" if video_file and video_file.filename != "" else "Audio Upload"
        )
    except Exception as e:
        flash(f"Failed to analyze audio: {str(e)}")
        return redirect(url_for("index"))

@app.route("/upload-media/<job_id>", methods=["GET"])
def upload_media_page(job_id):
    job_dir = JOBS_DIR / job_id
    
    if not job_dir.exists() or not (job_dir / "cuts.json").exists():
        flash("Invalid job ID or session expired.")
        return redirect(url_for("index"))
    
    # Load the saved data
    data = json.loads((job_dir / "cuts.json").read_text(encoding="utf-8"))
    audio_filename = data["audio"]
    num_segments = len(data["segments"])
    
    return render_template_string(
        UPLOAD_VISUAL_HTML,
        job_id=job_id,
        audio_filename=audio_filename,
        num_segments=num_segments
    )

@app.route("/preview-media/<job_id>", methods=["POST"])
def preview_media(job_id):
    job_dir = JOBS_DIR / job_id
    
    if not job_dir.exists() or not (job_dir / "cuts.json").exists():
        flash("Invalid job ID or session expired.")
        return redirect(url_for("index"))
    
    # Load the saved data
    data = json.loads((job_dir / "cuts.json").read_text(encoding="utf-8"))
    audio_filename = data["audio"]
    num_segments = len(data["segments"])
    
    # Get aspect ratio from form
    aspect_ratio = request.form.get("aspect_ratio", "16:9")
    
    # Save uploaded media temporarily
    media_files = []
    
    # Process videos
    for f in request.files.getlist("videos"):
        if f and f.filename:
            dst = job_dir / f.filename
            f.save(str(dst))
            media_files.append({"type": "video", "filename": f.filename})
    
    # Process images
    for f in request.files.getlist("images"):
        if f and f.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
            dst = job_dir / f.filename
            f.save(str(dst))
            media_files.append({"type": "image", "filename": f.filename})
    
    if not media_files:
        flash("Please upload at least one video clip or PNG image.")
        return redirect(url_for("upload_media_page", job_id=job_id))
    
    # Save media info and aspect ratio to job
    data["media_files"] = media_files
    data["aspect_ratio"] = aspect_ratio
    (job_dir / "cuts.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    
    return render_template_string(
        RENDER_OPTIONS_HTML,
        job_id=job_id,
        audio_filename=audio_filename,
        num_segments=num_segments,
        aspect_ratio=aspect_ratio,
        media_files=media_files
    )

@app.route("/render/<job_id>", methods=["POST"])
def render_video(job_id):
    job_dir = JOBS_DIR / job_id
    
    if not job_dir.exists() or not (job_dir / "cuts.json").exists():
        flash("Invalid job ID or session expired.")
        return redirect(url_for("index"))
    
    # Load the saved data
    data = json.loads((job_dir / "cuts.json").read_text(encoding="utf-8"))
    audio_filename = data["audio"]
    audio_path = job_dir / audio_filename
    fps = data.get("fps", 30)
    starts = [s["start"] for s in data["segments"]]
    ends = [s["end"] for s in data["segments"]]
    aspect_ratio = data.get("aspect_ratio", "16:9")
    
    # Get form data
    clip_mode = (request.form.get("clip_mode", "head") or "head").lower()
    if clip_mode not in {"head", "tail"}:
        clip_mode = "head"
    output_name = (request.form.get("output_name", "final_video.mp4") or "final_video.mp4").strip()
    
    # Process saved media files
    rendered = False
    
    # Check for videos
    media_files = data.get("media_files", [])
    saved_videos = [str(job_dir / item["filename"]) for item in media_files if item["type"] == "video"]
    
    if saved_videos:
        out_path = job_dir / output_name
        try:
            render_from_videos(saved_videos, starts, ends, str(audio_path), fps, str(out_path), clip_mode=clip_mode, aspect_ratio=aspect_ratio)
            rendered = True
        except Exception as e:
            flash(f"FFmpeg video render failed: {str(e)}")
    else:
        # Try PNG fallback
        pngs = [str(job_dir / item["filename"]) for item in media_files if item["type"] == "image"]
        
        if pngs:
            out_path = job_dir / output_name
            try:
                render_from_images(pngs, starts, ends, str(audio_path), fps, str(out_path), aspect_ratio=aspect_ratio)
                rendered = True
            except Exception as e:
                flash(f"FFmpeg image render failed: {str(e)}")
        else:
            flash("No videos or PNGs were provided for rendering.")
            return redirect(url_for("upload_media_page", job_id=job_id))

    # Show the final result
    return render_template_string(
        RESULT_HTML,
        job_id=job_id,
        fps=fps,
        num_onsets=len(data["events_onsets"]),
        num_segments=len(starts),
        num_flash=len(data.get("flash", [])),
        flash_start=data.get("flash_window", [0, 0])[0],
        flash_end=data.get("flash_window", [0, 0])[1],
        rendered=rendered,
        output_name=output_name
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

