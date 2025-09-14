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


def draw_legend(ax, text_lines: list[str], loc: str = "upper left"):
    x, y = {
        "upper left": (0.02, 0.98),
        "upper right": (0.98, 0.98),
        "lower left": (0.02, 0.02),
        "lower right": (0.98, 0.02),
    }.get(loc, (0.02, 0.98))
    txt = "\n".join(text_lines)
    ax.text(
        x,
        y,
        txt,
        transform=ax.transAxes,
        ha="left" if "left" in loc else "right",
        va="top" if "upper" in loc else "bottom",
        fontsize=8,
        color="#e5e7eb",
        bbox=dict(boxstyle="round,pad=0.4", fc=(0, 0, 0, 0.55), ec=(1, 1, 1, 0.2), lw=0.5),
    )


def draw_watermark(ax, text: str):
    ax.text(
        0.5,
        -0.08,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#cbd5e1",
        alpha=0.85,
        bbox=dict(boxstyle="round,pad=0.3", fc=(0, 0, 0, 0.25), ec=(1, 1, 1, 0.15), lw=0.5),
    )


def top_fvg_zones(overlays, kind: str, px: float, limit: int = 3):
    zones = []
    try:
        for z in overlays.get("fvg_active", []) or []:
            if z.get("type") == kind:
                low = float(z.get("low"))
                high = float(z.get("high"))
                center = (low + high) / 2.0
                dist = abs(center - px)
                zones.append((dist, {"low": low, "high": high}))
    except Exception:
        pass
    zones.sort(key=lambda t: t[0])
    # buy zones ideally below px, sell zones above px; keep sort but prioritize side
    preferred = [z for d, z in zones if (z["high"] <= px if kind == "bullish" else z["low"] >= px)]
    rest = [z for d, z in zones if z not in preferred]
    return (preferred + rest)[:limit]


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

    # Style (dark background for readability)
    mc = mpf.make_marketcolors(up="#10b981", down="#ef4444", inherit=True)
    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridcolor="#334155",
        facecolor="#0f172a",
        edgecolor="#0f172a",
        figcolor="#0f172a",
        gridstyle="-.",
        rc={"axes.labelcolor": "#cbd5e1", "xtick.color": "#94a3b8", "ytick.color": "#94a3b8"},
    )

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
    # mpf.plot may return a single Axes or a list/tuple of Axes. Use the price Axes.
    try:
        ax_main = ax[0] if isinstance(ax, (list, tuple, np.ndarray)) else ax
    except Exception:
        ax_main = ax
    try:
        fig.patch.set_facecolor("white")
        ax_main.set_facecolor("white")
    except Exception:
        pass
    # Draw FVG zones: top 3 each side
    try:
        px = float(df["Close"].iloc[-1])
        buys = top_fvg_zones(payload.overlays or {}, "bullish", px, limit=3)
        sells = top_fvg_zones(payload.overlays or {}, "bearish", px, limit=3)
        import matplotlib.patches as patches
        for idx, z in enumerate(buys, start=1):
            ax_main.axhspan(z["low"], z["high"], xmin=0.0, xmax=1.0, facecolor=(0.09, 0.78, 0.69, 0.12), edgecolor=(0.09, 0.78, 0.69, 0.35), linestyle=":", linewidth=0.8, zorder=0.3)
            ax_main.text(0.99, (z["low"] + z["high"]) / 2, f" {idx} ", color="#083344", fontsize=8, va="center", ha="right", transform=ax_main.get_yaxis_transform(), bbox=dict(boxstyle="round", fc="#99f6e4", ec="#0e7490", lw=0.6, alpha=0.9))
        for idx, z in enumerate(sells, start=1):
            ax_main.axhspan(z["low"], z["high"], xmin=0.0, xmax=1.0, facecolor=(0.94, 0.27, 0.27, 0.10), edgecolor=(0.94, 0.27, 0.27, 0.35), linestyle=":", linewidth=0.8, zorder=0.3)
            ax_main.text(0.99, (z["low"] + z["high"]) / 2, f" {idx} ", color="#450a0a", fontsize=8, va="center", ha="right", transform=ax_main.get_yaxis_transform(), bbox=dict(boxstyle="round", fc="#fecaca", ec="#b91c1c", lw=0.6, alpha=0.9))
    except Exception:
        pass

    # Signals and lines
    try:
        draw_overlays(ax_main, payload.overlays)
    except Exception:
        pass

    # Legends and watermark
    last_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
    chg = last_close - prev_close
    pct = (chg / prev_close * 100) if prev_close else 0.0
    from datetime import datetime, timezone, timedelta
    wib = datetime.now(timezone.utc) + timedelta(hours=7)
    info_lines = [
        f"{payload.symbol} {payload.tf}",
        f"$ {last_close:.4f}",
        f"{chg:+.4f} ({pct:+.2f}%)",
        f"Time: {wib.strftime('%H:%M')} WIB",
    ]
    draw_legend(ax_main, info_lines, loc="upper left")

    # Footer stats
    ov = payload.overlays or {}
    footer = f"CHoCH: {ov.get('choch_count', 0)} | BOS: {ov.get('bos_bull',0)+ov.get('bos_bear',0)} | {ov.get('trend','')}"
    draw_watermark(ax_main, footer)

    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=160)
        buf.seek(0)
        png = buf.getvalue()
        return Response(content=png, media_type="image/png")
    finally:
        plt.close(fig)
