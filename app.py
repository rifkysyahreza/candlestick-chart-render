import os
from fastapi import FastAPI, Response, Depends, Header, HTTPException, Request
from pydantic import BaseModel
import io
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import matplotlib.patheffects as pe

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
        ax.axhline(y=y, color=color, linewidth=1.2, linestyle="-")
        if label:
            ax.text(ax.get_xlim()[0], y, f" {label}", color=color, va="bottom", fontsize=8)

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


def draw_legend(ax, text_lines: list[str], loc: str = "upper left", use_figure: bool = False):
    # Choose coordinate system
    trans = ax.figure.transFigure if use_figure else ax.transAxes
    # Default anchors
    if use_figure:
        # place inside figure margins; assumes left<=0.12, right<=0.82
        pos_map = {
            "upper left": (0.06, 0.97),
            "upper right": (0.94, 0.97),
            "lower left": (0.06, 0.06),
            "lower right": (0.94, 0.06),
        }
    else:
        pos_map = {
            "upper left": (0.02, 0.98),
            "upper right": (0.98, 0.98),
            "lower left": (0.02, 0.02),
            "lower right": (0.98, 0.02),
        }
    x, y = pos_map.get(loc, (0.02, 0.98))
    txt = "\n".join(text_lines)
    ax.text(
        x,
        y,
        txt,
        transform=trans,
        ha="left" if "left" in loc else "right",
        va="top" if "upper" in loc else "bottom",
        fontsize=11,
        color="#e5e7eb",
        bbox=dict(boxstyle="round,pad=0.5", fc=(0, 0, 0, 0.6), ec=(1, 1, 1, 0.25), lw=0.6),
    )


def draw_watermark(ax, text: str):
    ax.text(
        0.5,
        -0.08,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        color="#e5e7eb",
        alpha=0.9,
        bbox=dict(boxstyle="round,pad=0.35", fc=(0, 0, 0, 0.35), ec=(1, 1, 1, 0.2), lw=0.6),
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


def draw_right_legend(ax):
    # Draw a compact legend block at top-right with color keys; background auto-fits contents
    fig = ax.figure
    trans = fig.transFigure
    x0, y0 = 0.965, 0.965  # top-right anchor in figure coords
    dy = 0.04
    items = [
        ("Buy FVG", (0.09, 0.78, 0.69, 0.8), None, "box"),
        ("Sell FVG", (0.94, 0.27, 0.27, 0.8), None, "box"),
        ("Entry Area", "#10b981", None, "line"),
        ("Touch", "#fde047", None, "tri"),
        ("CHoCH", "#14b8a6", "C", "label"),
        ("BOS", "#f59e0b", "B", "label"),
    ]
    # Draw items first to measure their extents
    artists = []
    x_text = x0 - 0.008
    x_sample = x_text - 0.06
    for i, (label, color, txt, kind) in enumerate(items):
        y = y0 - i * dy
        artists.append(ax.text(x_text, y, label, transform=trans, fontsize=9, color="#e5e7eb", ha="right", va="top", zorder=10))
        if kind == "box":
            rect = patches.Rectangle((x_sample, y - 0.022), 0.028, 0.014, transform=trans, fc=color, ec=color, lw=1, zorder=10)
            ax.add_artist(rect); artists.append(rect)
        elif kind == "line":
            ln = Line2D([x_sample, x_sample + 0.028], [y - 0.017, y - 0.017], transform=trans, color=color, lw=2, zorder=10)
            ax.add_line(ln); artists.append(ln)
        elif kind == "label":
            artists.append(ax.text(x_sample + 0.014, y - 0.017, txt or "", transform=trans, fontsize=8.5, color="#0f172a",
                                   ha="center", va="center", bbox=dict(boxstyle="round,pad=0.18", fc=color, ec=(1,1,1,0.25), lw=0.5), zorder=10))
        elif kind == "tri":
            sc = ax.scatter([x_sample + 0.014], [y - 0.017], transform=trans, marker="v", color=color, s=22, zorder=10)
            artists.append(sc)
    # Force a draw to get a renderer and compute extents
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        from matplotlib.transforms import Bbox
        bbs = []
        for ar in artists:
            try:
                bb_disp = ar.get_window_extent(renderer=renderer)
                bb_fig = fig.transFigure.inverted().transform_bbox(bb_disp)
                bbs.append(bb_fig)
            except Exception:
                pass
        if bbs:
            minx = min(bb.x0 for bb in bbs); miny = min(bb.y0 for bb in bbs)
            maxx = max(bb.x1 for bb in bbs); maxy = max(bb.y1 for bb in bbs)
            pad_x = 0.006; pad_y = 0.008
            panel = patches.FancyBboxPatch((minx - pad_x, miny - pad_y), (maxx - minx) + 2*pad_x, (maxy - miny) + 2*pad_y,
                                           boxstyle="round,pad=0.35", transform=trans,
                                           facecolor=(0,0,0,0.70), edgecolor=(1,1,1,0.18), linewidth=0.6, zorder=5)
            ax.add_artist(panel)
    except Exception:
        pass


def compute_markers(df):
    # Lightweight swings + BOS/CHoCH markers
    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    n = len(df)
    LEFT = RIGHT = 2
    swings_high = []
    swings_low = []
    for i in range(LEFT, n - RIGHT):
        is_h = all(h[i] > h[i - k] for k in range(1, LEFT + 1)) and all(h[i] >= h[i + k] for k in range(1, RIGHT + 1))
        is_l = all(l[i] < l[i - k] for k in range(1, LEFT + 1)) and all(l[i] <= l[i + k] for k in range(1, RIGHT + 1))
        if is_h:
            swings_high.append(i)
        if is_l:
            swings_low.append(i)
    markers = []
    regime = None
    last_h = max(swings_high) if swings_high else None
    last_l = max(swings_low) if swings_low else None
    for i in range(n):
        if last_h is not None and c[i] > h[last_h]:
            markers.append((i, c[i], "BOS", "up"))
            if regime == "down":
                markers.append((i, c[i], "CHoCH", "up"))
            regime = "up"
            # BSL near the broken low if available
            if last_l is not None:
                markers.append((i, l[last_l], "BSL", "low"))
            # update last_h to next swing after i
            last_h = next((idx for idx in swings_high if idx > i), last_h)
        if last_l is not None and c[i] < l[last_l]:
            markers.append((i, c[i], "BOS", "down"))
            if regime == "up":
                markers.append((i, c[i], "CHoCH", "down"))
            regime = "down"
            if last_h is not None:
                markers.append((i, h[last_h], "BSL", "high"))
            last_l = next((idx for idx in swings_low if idx > i), last_l)
        # roll swing references forward
        if last_h is None or (swings_high and i >= swings_high[-1]):
            last_h = next((idx for idx in swings_high if idx > i), last_h)
        if last_l is None or (swings_low and i >= swings_low[-1]):
            last_l = next((idx for idx in swings_low if idx > i), last_l)
    return markers


def draw_markers(ax, df):
    markers = compute_markers(df)
    idx = df.index.to_numpy()
    for i, y, kind, direction in markers[:200]:  # cap to avoid clutter
        x = idx[i]
        if kind == "BOS":
            ax.axhline(y, color="#f59e0b", linestyle="-", linewidth=1.0, alpha=0.9)
            ax.text(x, y, " BOS ", color="#7c2d12", fontsize=8, va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.2", fc="#fed7aa", ec="#f59e0b", lw=0.4, alpha=0.95))
        elif kind == "CHoCH":
            ax.axhline(y, color="#14b8a6", linestyle="-", linewidth=1.0, alpha=0.9)
            ax.text(x, y, " CH ", color="#064e3b", fontsize=8, va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.2", fc="#99f6e4", ec="#14b8a6", lw=0.4, alpha=0.95))
        elif kind == "BSL":
            ax.axhline(y, color="#a78bfa", linestyle=(0, (3, 3)), linewidth=1.0, alpha=0.9)
            ax.text(x, y, " BSL ", color="#312e81", fontsize=8, va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.2", fc="#ddd6fe", ec="#4f46e5", lw=0.4, alpha=0.95))


def draw_events(ax, df, overlays):
    try:
        events = (overlays or {}).get("events")
        if not events:
            return
        # map ts to index datetime
        for ev in events[:300]:
            ts = pd.to_datetime(ev.get("ts", None), unit="ms", utc=True)
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
            price = float(ev.get("price", 0))
            kind = ev.get("kind")
            direction = ev.get("dir")
            x = ts
            if kind == "BOS":
                ax.axhline(price, color="#f59e0b", linestyle="-", linewidth=1.0, alpha=0.9)
                ax.text(x, price, " BOS ", color="#7c2d12", fontsize=8, va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.2", fc="#fed7aa", ec="#f59e0b", lw=0.4, alpha=0.95))
            elif kind == "CHoCH":
                ax.axhline(price, color="#14b8a6", linestyle="-", linewidth=1.0, alpha=0.9)
                ax.text(x, price, " CH ", color="#064e3b", fontsize=8, va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.2", fc="#99f6e4", ec="#14b8a6", lw=0.4, alpha=0.95))
            elif kind == "BSL":
                ax.axhline(price, color="#a78bfa", linestyle=(0, (3, 3)), linewidth=1.0, alpha=0.9)
                ax.text(x, price, " BSL ", color="#312e81", fontsize=8, va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.2", fc="#ddd6fe", ec="#4f46e5", lw=0.4, alpha=0.95))
    except Exception:
        pass


def draw_short_lines(ax, df, overlays):
    """Draw horizontal arrows per BOS/CHoCH, dynamically extending to the next event.
    If there is no later event, extend to the last candle.
    """
    try:
        events = (overlays or {}).get("events") or []
        if not events:
            return
        # sort by time
        events = sorted(events, key=lambda e: e.get("ts", 0))
        def dt(ms):
            t = pd.to_datetime(ms, unit="ms", utc=True)
            return t.tz_convert("UTC").tz_localize(None)
        last_x = df.index[-1]
        prev_y = None
        for i, ev in enumerate(events):
            kind = ev.get("kind")
            if kind not in ("BOS", "CHoCH"):
                continue
            y = float(ev.get("price", 0))
            # de-duplicate if same level repeats
            if prev_y is not None and abs(prev_y - y) < 1e-9:
                continue
            prev_y = y
            x0 = dt(ev.get("ts"))
            x1 = dt(events[i + 1]["ts"]) if i + 1 < len(events) else last_x
            # guard for ordering
            if x1 <= x0:
                x1 = last_x
            color = "#f59e0b" if kind == "BOS" else "#14b8a6"
            ax.annotate(
                "",
                xy=(x1, y),
                xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, shrinkA=0, shrinkB=0, capstyle="round"),
            )
    except Exception:
        pass


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
            "Date": pd.to_datetime([c.ts for c in payload.data], unit="ms", utc=True).tz_convert("UTC").tz_localize(None),
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
        figsize=(16, 8),
        datetime_format="%m-%d %H:%M",
        tight_layout=True,
        update_width_config=dict(candle_linewidth=0.6, candle_width=0.6),
    )
    # top-center title
    try:
        fig.suptitle(f"{payload.symbol} {payload.tf} — Systematic Smart Money by Nca", color="#e5e7eb", fontsize=14, fontweight="bold")
    except Exception:
        pass

    # Axis labels
    try:
        ax_main.set_ylabel("Price (USD)", color="#cbd5e1")
        ax_main.set_xlabel("Time (WIB)", color="#cbd5e1")
        tz_wib = pd.Timestamp.utcnow().tz_localize("UTC").tz_convert("Asia/Jakarta").tz
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M WIB\n%d-%m", tz=tz_wib))
    except Exception:
        pass
    # Add extra padding: leave space at top and right/left for legends
    try:
        fig.subplots_adjust(left=0.12, right=0.80, top=0.78, bottom=0.26)
    except Exception:
        pass
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
    # Clean FVG representation: only nearest BUY/SELL as dashed midlines with labels
    try:
        ov = payload.overlays or {}
        nb = ((ov.get("nearest") or {}).get("buy"))
        ns = ((ov.get("nearest") or {}).get("sell"))
        if nb:
            y = (float(nb.get("low", 0)) + float(nb.get("high", 0))) / 2.0
            ax_main.axhline(y, color="#22d3ee", linestyle=(0, (6, 3)), linewidth=1.2, alpha=0.9)
            ax_main.text(ax_main.get_xlim()[1], y, " BUY FVG ", color="#022c22", fontsize=9, va="bottom", ha="right",
                         bbox=dict(boxstyle="round,pad=0.2", fc="#99f6e4", ec="#0e7490", lw=0.6, alpha=0.95))
        if ns:
            y = (float(ns.get("low", 0)) + float(ns.get("high", 0))) / 2.0
            ax_main.axhline(y, color="#ef4444", linestyle=(0, (6, 3)), linewidth=1.2, alpha=0.9)
            ax_main.text(ax_main.get_xlim()[1], y, " SELL FVG ", color="#450a0a", fontsize=9, va="top", ha="right",
                         bbox=dict(boxstyle="round,pad=0.2", fc="#fecaca", ec="#b91c1c", lw=0.6, alpha=0.95))
    except Exception:
        pass

    # Signals and lines
    try:
        draw_overlays(ax_main, payload.overlays)
    except Exception:
        pass

    # Minimal entry/SL/TP lines from signals
    try:
        sigL = (payload.overlays or {}).get("long_signal")
        sigS = (payload.overlays or {}).get("short_signal")
        def label_line(y, text, color, va="bottom"):
            ax_main.axhline(y, color=color, linewidth=1.2)
            ax_main.text(ax_main.get_xlim()[0], y, f" {text} ", color="#0f172a", fontsize=9, va=va, ha="left",
                         bbox=dict(boxstyle="round,pad=0.2", fc=color, ec=color, lw=0.4, alpha=0.9))
        if sigL:
            label_line(float(sigL.get("entry")), "Entry", "#10b981")
            ax_main.axhline(float(sigL.get("sl")), color="#ef4444", linestyle=(0,(6,3)), linewidth=1.1)
            ax_main.text(ax_main.get_xlim()[0], float(sigL.get("sl")), " SL ", color="#7f1d1d", va="top", ha="left",
                         fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="#fecaca", ec="#ef4444", lw=0.4, alpha=0.9))
            ax_main.axhline(float(sigL.get("tp1")), color="#22c55e", linestyle=(0,(2,2)), linewidth=1.0, alpha=0.9)
            ax_main.text(ax_main.get_xlim()[0], float(sigL.get("tp1")), " TP1 ", color="#064e3b", va="bottom", ha="left",
                         fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="#bbf7d0", ec="#22c55e", lw=0.4, alpha=0.9))
        if sigS:
            label_line(float(sigS.get("entry")), "Entry", "#ef4444")
            ax_main.axhline(float(sigS.get("sl")), color="#10b981", linestyle=(0,(6,3)), linewidth=1.1)
            ax_main.text(ax_main.get_xlim()[0], float(sigS.get("sl")), " SL ", color="#064e3b", va="top", ha="left",
                         fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="#a7f3d0", ec="#10b981", lw=0.4, alpha=0.9))
            ax_main.axhline(float(sigS.get("tp1")), color="#f59e0b", linestyle=(0,(2,2)), linewidth=1.0, alpha=0.9)
            ax_main.text(ax_main.get_xlim()[0], float(sigS.get("tp1")), " TP1 ", color="#7c2d12", va="bottom", ha="left",
                         fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="#fde68a", ec="#f59e0b", lw=0.4, alpha=0.9))
    except Exception:
        pass

    # Keep chart clean: skip additional touch markers

    # Legends and watermark
    last_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
    chg = last_close - prev_close
    pct = (chg / prev_close * 100) if prev_close else 0.0
    from datetime import datetime, timezone, timedelta
    wib = datetime.now(timezone.utc) + timedelta(hours=7)
    ov = payload.overlays or {}
    bos_total = (ov.get('bos_bull', 0) or 0) + (ov.get('bos_bear', 0) or 0)
    choch_cnt = ov.get('choch_count', 0) or 0
    entry_cnt = len(ov.get('entry_areas') or [])
    info_lines = [
        f"{payload.symbol} {payload.tf}",
        f"$ {last_close:.4f}",
        f"{chg:+.4f} ({pct:+.2f}%)",
        f"Time: {wib.strftime('%H:%M')} WIB",
        f"BOS/CHoCH: {bos_total}/{choch_cnt}",
        f"Entry Areas: {entry_cnt}",
    ]
    draw_legend(ax_main, info_lines, loc="upper left", use_figure=True)
    draw_right_legend(ax_main)
    # Prefer exact events from overlays; fallback to computed markers
    if isinstance(payload.overlays, dict) and payload.overlays.get("events"):
        draw_events(ax_main, df, payload.overlays)
    else:
        draw_markers(ax_main, df)
    # Add short horizontal arrows from each BOS/CHoCH forward a few bars
    draw_short_lines(ax_main, df, payload.overlays)

    # Footer stats
    ov = payload.overlays or {}
    bsl_count = sum(1 for e in (ov.get('events') or []) if e.get('kind') == 'BSL')
    footer = f"CHoCH: {choch_cnt} | BOS: {bos_total} | BSL: {bsl_count} | {ov.get('trend','')}"
    draw_watermark(ax_main, footer)

    # Bottom-left legend removed for a cleaner look

    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=160)
        buf.seek(0)
        png = buf.getvalue()
        return Response(content=png, media_type="image/png")
    finally:
        plt.close(fig)
