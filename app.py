import os
from fastapi import FastAPI, Response, Depends, Header, HTTPException, Request
from pydantic import BaseModel
import io
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt

app = FastAPI()


class Candle(BaseModel):
    ts: int
    open: float
    high: float
    low: float
    close: float


class Payload(BaseModel):
    symbol: str
    tf: str
    data: list[Candle]
    overlays: dict | None = None

def verify_api_key(x_api_key: str | None = Header(default=None)):
    required = os.getenv("API_KEY")
    if required and x_api_key != required:
        raise HTTPException(status_code=401, detail="invalid api key")

@app.middleware("http")
async def limit_payload_size(request: Request, call_next):
    # Basic protection using Content-Length header; platforms usually set this.
    max_bytes = int(os.getenv("MAX_BODY_BYTES", "1048576"))  # 1 MiB default
    cl = request.headers.get("content-length")
    try:
        if cl is not None and int(cl) > max_bytes:
            return Response(status_code=413, content="payload too large")
    except ValueError:
        pass
    return await call_next(request)

@app.get("/healthz")
def healthz():
    return {"ok": True}

def draw_overlays(ax, overlays):
    if not overlays:
        return
    import matplotlib.patches as patches
    # FVG nearest zones
    nearest = overlays.get("nearest", {}) if isinstance(overlays, dict) else {}
    buy = nearest.get("buy") if isinstance(nearest, dict) else None
    sell = nearest.get("sell") if isinstance(nearest, dict) else None
    if buy:
        ax.axhspan(
            buy.get("low", 0),
            buy.get("high", 0),
            xmin=0.0,
            xmax=1.0,
            facecolor=(0.1, 0.73, 0.5, 0.08),
            edgecolor=(0.1, 0.73, 0.5, 0.25),
            linewidth=1,
            zorder=0.5,
        )
    if sell:
        ax.axhspan(
            sell.get("low", 0),
            sell.get("high", 0),
            xmin=0.0,
            xmax=1.0,
            facecolor=(0.94, 0.27, 0.27, 0.08),
            edgecolor=(0.94, 0.27, 0.27, 0.25),
            linewidth=1,
            zorder=0.5,
        )

    # Signals lines
    long_sig = overlays.get("long_signal") if isinstance(overlays, dict) else None
    short_sig = overlays.get("short_signal") if isinstance(overlays, dict) else None

    def hline(y, color, label=None):
        ax.axhline(y=y, color=color, linewidth=1, linestyle="-")
        if label:
            ax.text(ax.get_xlim()[0], y, f" {label}", color=color, va="bottom", fontsize=7)

    if long_sig:
        hline(long_sig.get("entry"), "#10b981", "Entry")
        hline(long_sig.get("sl"), "#ef4444", "SL")
        hline(long_sig.get("tp1"), "#22c55e", "TP1")
        hline(long_sig.get("tp2"), "#16a34a", "TP2")
    if short_sig:
        hline(short_sig.get("entry"), "#ef4444", "Entry")
        hline(short_sig.get("sl"), "#10b981", "SL")
        hline(short_sig.get("tp1"), "#f97316", "TP1")
        hline(short_sig.get("tp2"), "#ea580c", "TP2")


@app.post("/render", dependencies=[Depends(verify_api_key)])
def render(payload: Payload):
    if not payload.data:
        return Response(status_code=400, content="no data")
    max_candles = int(os.getenv("MAX_CANDLES", "1000"))
    if len(payload.data) > max_candles:
        return Response(status_code=413, content="too many candles")
    # Build dataframe for mplfinance
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime([c.ts for c in payload.data], unit="ms"),
            "Open": [c.open for c in payload.data],
            "High": [c.high for c in payload.data],
            "Low": [c.low for c in payload.data],
            "Close": [c.close for c in payload.data],
        }
    )
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    # Style
    mc = mpf.make_marketcolors(up="#10b981", down="#ef4444", inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridcolor="#e5e7eb", gridstyle="-.")

    fig, ax = mpf.plot(
        df,
        type="candle",
        style=s,
        volume=False,
        returnfig=True,
        figsize=(12, 6),
        datetime_format="%m-%d %H:%M",
        tight_layout=True,
        update_width_config=dict(candle_linewidth=0.6, candle_width=0.6),
    )
    try:
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
    except Exception:
        pass
    try:
        draw_overlays(ax, payload.overlays)
    except Exception:
        pass

    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=160)
        buf.seek(0)
        png = buf.getvalue()
        return Response(content=png, media_type="image/png")
    finally:
        plt.close(fig)
