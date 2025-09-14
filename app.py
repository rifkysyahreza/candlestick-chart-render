from fastapi import FastAPI, Response
from pydantic import BaseModel
import io
import pandas as pd
import numpy as np
import mplfinance as mpf

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


def draw_overlays(ax, overlays):
    if not overlays:
        return
    import matplotlib.patches as patches
    # FVG nearest zones
    nearest = overlays.get("nearest", {}) if isinstance(overlays, dict) else {}
    buy = nearest.get("buy") if isinstance(nearest, dict) else None
    sell = nearest.get("sell") if isinstance(nearest, dict) else None
    if buy:
        ax.add_patch(
            patches.Rectangle(
                (0, buy.get("low", 0)),
                width=ax.get_xlim()[1] - ax.get_xlim()[0],
                height=buy.get("high", 0) - buy.get("low", 0),
                facecolor=(0.1, 0.73, 0.5, 0.08),
                edgecolor=(0.1, 0.73, 0.5, 0.25),
                linewidth=1,
            )
        )
    if sell:
        ax.add_patch(
            patches.Rectangle(
                (0, sell.get("low", 0)),
                width=ax.get_xlim()[1] - ax.get_xlim()[0],
                height=sell.get("high", 0) - sell.get("low", 0),
                facecolor=(0.94, 0.27, 0.27, 0.08),
                edgecolor=(0.94, 0.27, 0.27, 0.25),
                linewidth=1,
            )
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


@app.post("/render")
def render(payload: Payload):
    if not payload.data:
        return Response(status_code=400, content="no data")
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
        draw_overlays(ax, payload.overlays)
    except Exception:
        pass

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    buf.seek(0)
    png = buf.getvalue()
    return Response(content=png, media_type="image/png")

