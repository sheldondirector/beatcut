# Flashcut

A simple Flask web app that analyzes an uploaded audio file for onsets and generates cut segments; optionally renders a video timeline using FFmpeg.

## Deploying to Railway

- This repo is ready for Railway out of the box (Nixpacks). It uses a Procfile and binds to `$PORT`.
- Python version is pinned via `runtime.txt`.
- Dependencies are pinned in `requirements.txt` for reliable builds on Python 3.11.

### FFmpeg (optional, for rendering)
Rendering videos/images requires FFmpeg. If it is not available, the app still works for analysis; rendering will be skipped with a friendly message.

To include FFmpeg on Railway, enable it in the Nixpacks settings in the Railway dashboard (Install Packages → `ffmpeg`).

### Environment

Optional variables:
- `FLASK_SECRET` — Flask session secret (defaults to "dev-secret").
 - Python version is pinned via `runtime.txt` (3.11.9). No extra nixpacks.toml is required.

### Health check

- `GET /health` → `{ "ok": true }`

### Notes

- Max upload size is 500 MB by default.
- Gunicorn is configured with a single threaded worker suitable for CPU-bound analysis; increase if you expect higher concurrency.
