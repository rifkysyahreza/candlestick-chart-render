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

Deploy

- Google Cloud Run (recommended free-tier):
  1) Make sure you have a GCP project and `gcloud` installed/authenticated.
  2) This repo includes a `Dockerfile` so Cloud Run will build from it.
  3) Deploy:
     gcloud run deploy candlestick-render \
       --source . \
       --region us-central1 \
       --allow-unauthenticated
  4) Copy the service URL and set it as `RENDERER_URL` in your Worker config.

- Railway (limited free credits):
  - Push this repo to GitHub and create a new Railway project from the repo.
  - Railway will auto-detect the `Dockerfile` and build/deploy the container.
  - Ensure the service listens on port `8000` (Railway sets `PORT`; our CMD reads it if provided by platform or defaults to 8000).

- Fly.io (low-cost, sometimes free usage):
  - Install `flyctl`, run `flyctl launch` in the repo, accept Dockerfile deploy, then `flyctl deploy`.
  - Expose internal port `8000`; Fly will map to HTTPS externally.

Notes

- This service uses matplotlib headless backend (`Agg`) and installs runtime libs (`libfreetype6`, `libpng16-16`) in the container for compatibility.
- Serverless buildpacks without a Dockerfile may miss these libs; prefer the provided container on platforms that support it.
- Free plans change frequently. As of 2025, Cloud Run has an Always Free tier suitable for low traffic; Railway often includes small monthly credits; Fly has low-cost pricing with occasional free allowances. Render no longer offers a true persistent free tier.
