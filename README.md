Advanced Candle Renderer (Python)

Overview

This is a minimal FastAPI service that renders candlestick charts with overlays (FVG boxes, Entry/SL/TP lines) and returns a PNG image. Use it with the Worker by setting RENDERER_URL in wrangler.toml.

Endpoints

- POST /render → image/png
  Body (JSON): { symbol, tf, data: [{ ts, open, high, low, close }...], overlays: any }

Quick start

1) Create venv and install deps
   python -m venv .venv
   . .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt

2) Run dev server
   uvicorn app:app --host 0.0.0.0 --port 8000

3) Configure Worker
   wrangler.toml → [vars] RENDERER_URL = "https://your-host/render"

Deploy ideas

- Render.com, Fly.io, Railway.app, Cloud Run, etc.
- If you want to return a URL instead of bytes, upload to R2/S3 in the handler and return { url }.

