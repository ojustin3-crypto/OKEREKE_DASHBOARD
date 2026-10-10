import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from datetime import datetime, timezone, date, timedelta
import pytz
import base64
import requests

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="OCAP Evergreen (BETA)",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto"
)

# ── Custom styling ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IM+Fell+English:ital@0;1&display=swap');

    /* Base */
    .stApp {
        background-color: #080808;
        background-image:
            linear-gradient(rgba(0,184,156,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,184,156,0.03) 1px, transparent 1px);
        background-size: 40px 40px;
    }
    .block-container { padding-top: 0rem !important; }
    section[data-testid="stSidebar"] { background-color: #0d0d0d; }

    html, body, [class*="css"], p, div, span, label {
        font-family: 'Times New Roman', Times, serif !important;
        color: #e0e0e0;
    }

    /* Inputs and selects */
    .stSelectbox > div, .stCheckbox, .stButton {
        font-family: 'Times New Roman', Times, serif !important;
    }

    /* Stat cards */
    .stat-card {
        background: #0f0f0f;
        border: 1px solid #1a1a1a;
        border-left: 3px solid #00b89c;
        border-radius: 0;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .stat-label {
        font-size: 10px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-family: 'Times New Roman', Times, serif !important;
        font-style: italic;
    }
    .stat-value {
        font-size: 26px;
        font-weight: bold;
        color: #ffffff;
        margin-top: 6px;
        font-family: 'Times New Roman', Times, serif !important;
    }

    /* Status card */
    .status-card {
        border-radius: 0;
        padding: 14px 20px;
        margin-bottom: 12px;
        border: 1px solid #1a1a1a;
        border-left: 3px solid #00b89c;
    }

    /* Session badge */
    .session-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 0;
        font-size: 11px;
        font-weight: bold;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-family: 'Times New Roman', Times, serif !important;
    }

    /* Live indicator */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
    .live-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 0;
        background: #ff4444;
        animation: pulse 1.5s infinite;
        margin-right: 8px;
    }

    hr { border-color: #1a1a1a !important; margin: 24px 0 !important; }

    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stSelectbox label, .stCheckbox label {
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 13px;
        color: #888 !important;
        font-style: italic;
    }

    button[kind="secondary"], button[kind="primary"] {
        border-radius: 0 !important;
        font-family: 'Times New Roman', Times, serif !important;
        font-weight: bold;
        border: 1px solid #333 !important;
        background: #111 !important;
        color: #fff !important;
    }

    .stCaption {
        font-family: 'Times New Roman', Times, serif !important;
        color: #555 !important;
        font-style: italic;
    }

    .stSelectbox > div > div {
        border-radius: 0 !important;
        border: 1px solid #333 !important;
        background-color: #0f0f0f !important;
        font-family: 'Times New Roman', Times, serif !important;
    }
    .stSelectbox > div > div:hover { border-color: #00b89c !important; }
    [data-baseweb="select"] { border-radius: 0 !important; }
    [data-baseweb="popover"] { border-radius: 0 !important; }
    [data-baseweb="menu"] {
        border-radius: 0 !important;
        background-color: #0f0f0f !important;
        border: 1px solid #222 !important;
    }
    [data-baseweb="option"] {
        background-color: #0f0f0f !important;
        font-family: 'Times New Roman', Times, serif !important;
    }
    [data-baseweb="option"]:hover { background-color: #1a1a1a !important; }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #080808 !important;
        border-bottom: 1px solid #1a1a1a !important;
        gap: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #080808 !important;
        border-radius: 0 !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        color: #444 !important;
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
        padding: 10px 24px !important;
        font-style: italic !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #080808 !important;
        border-bottom: 2px solid #00b89c !important;
        color: #ffffff !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #00b89c !important;
        background-color: #080808 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 28px !important;
        background-color: #080808 !important;
    }

    /* Strength table rows */
    .pair-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 16px;
        border-bottom: 1px solid #111;
        font-family: 'Times New Roman', Times, serif;
    }
    .pair-row:hover { background: #0f0f0f; }
</style>
""", unsafe_allow_html=True)

# ── Watchlist ─────────────────────────────────────────────────
WATCHLIST = {
    "Gold":     {"ticker": "GC=F",      "pip": 10,    "threshold": 30},
    "US Oil":   {"ticker": "CL=F",      "pip": 100,   "threshold": 50},
    "S&P 500":  {"ticker": "ES=F",      "pip": 1,     "threshold": 5},
    "NAS100":   {"ticker": "NQ=F",      "pip": 1,     "threshold": 50},
    "US30":     {"ticker": "YM=F",      "pip": 1,     "threshold": 100},
    "Bitcoin":  {"ticker": "BTC-USD",   "pip": 1,     "threshold": 500},
    "EUR/USD":  {"ticker": "EURUSD=X",  "pip": 10000, "threshold": 10},
    "EUR/GBP":  {"ticker": "EURGBP=X",  "pip": 10000, "threshold": 10},
    "EUR/JPY":  {"ticker": "EURJPY=X",  "pip": 100,   "threshold": 10},
    "EUR/CHF":  {"ticker": "EURCHF=X",  "pip": 10000, "threshold": 10},
    "EUR/CAD":  {"ticker": "EURCAD=X",  "pip": 10000, "threshold": 10},
    "EUR/AUD":  {"ticker": "EURAUD=X",  "pip": 10000, "threshold": 10},
    "EUR/NZD":  {"ticker": "EURNZD=X",  "pip": 10000, "threshold": 10},
    "GBP/USD":  {"ticker": "GBPUSD=X",  "pip": 10000, "threshold": 10},
    "GBP/JPY":  {"ticker": "GBPJPY=X",  "pip": 100,   "threshold": 10},
    "GBP/CHF":  {"ticker": "GBPCHF=X",  "pip": 10000, "threshold": 10},
    "GBP/CAD":  {"ticker": "GBPCAD=X",  "pip": 10000, "threshold": 10},
    "GBP/AUD":  {"ticker": "GBPAUD=X",  "pip": 10000, "threshold": 10},
    "GBP/NZD":  {"ticker": "GBPNZD=X",  "pip": 10000, "threshold": 10},
    "CHF/JPY":  {"ticker": "CHFJPY=X",  "pip": 100,   "threshold": 10},
    "USD/JPY":  {"ticker": "USDJPY=X",  "pip": 100,   "threshold": 10},
    "USD/CHF":  {"ticker": "USDCHF=X",  "pip": 10000, "threshold": 10},
    "USD/CAD":  {"ticker": "USDCAD=X",  "pip": 10000, "threshold": 10},
    "CAD/JPY":  {"ticker": "CADJPY=X",  "pip": 100,   "threshold": 10},
    "CAD/CHF":  {"ticker": "CADCHF=X",  "pip": 10000, "threshold": 10},
    "AUD/USD":  {"ticker": "AUDUSD=X",  "pip": 10000, "threshold": 10},
    "AUD/JPY":  {"ticker": "AUDJPY=X",  "pip": 100,   "threshold": 10},
    "AUD/CHF":  {"ticker": "AUDCHF=X",  "pip": 10000, "threshold": 10},
    "AUD/CAD":  {"ticker": "AUDCAD=X",  "pip": 10000, "threshold": 10},
    "AUD/NZD":  {"ticker": "AUDNZD=X",  "pip": 10000, "threshold": 10},
    "NZD/USD":  {"ticker": "NZDUSD=X",  "pip": 10000, "threshold": 10},
    "NZD/JPY":  {"ticker": "NZDJPY=X",  "pip": 100,   "threshold": 10},
    "NZD/CHF":  {"ticker": "NZDCHF=X",  "pip": 10000, "threshold": 10},
    "NZD/CAD":  {"ticker": "NZDCAD=X",  "pip": 10000, "threshold": 10},
}

# ── Session detector ──────────────────────────────────────────
def get_current_session():
    utc_hour = datetime.now(timezone.utc).hour
    if 21 <= utc_hour or utc_hour < 0:
        return "Sydney", "#4A9EE0"
    elif 0 <= utc_hour < 7:
        return "Tokyo", "#E0B44A"
    elif 7 <= utc_hour < 12:
        return "London", "#E07A4A"
    elif 12 <= utc_hour < 17:
        return "New York", "#7A4AE0"
    else:
        return "Off Hours", "#444444"

# ── Top header ────────────────────────────────────────────────
session_name, session_color = get_current_session()
utc_now = datetime.now(timezone.utc)

st.markdown(f"""
<div style="
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding: 20px 0 20px 0;
    border-bottom: 1px solid #1a1a1a;
    margin-bottom: 28px;
">
    <div style="
        font-size: 26px;
        font-weight: bold;
        color: #ffffff;
        font-family: 'Times New Roman', Times, serif;
        letter-spacing: 0.06em;
        text-transform: none;
    ">Okereke Capital | OCAP Evergreen 0.1 (BETA)</div>
    <div style="display:flex; align-items:center; gap:20px;">
        <span class="session-badge" style="
            background: transparent;
            color: {session_color};
            border: 1px solid {session_color};
        ">{session_name} Session</span>
        <div style="
            font-size: 11px;
            color: #444;
            font-family: 'Times New Roman', Times, serif;
            font-style: italic;
            letter-spacing: 0.05em;
        ">{utc_now.strftime('%H:%M UTC')} &nbsp;·&nbsp; Market Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# BACKTEST — FVG first fill + EMA reclaim
# Engine (OANDA download + strategy port), then the tab that draws it.
#
# Rules (same as the Pine strategy):
#   - Standard 3-bar FVG on the chart timeframe (middle bar must also close beyond bar A's extreme).
#   - The gap must form inside the window (bar C opens inside it).
#   - First fill only: a bar's wick reaches the far edge without closing through it.
#     If that first fill happens outside the window, the gap is dead.
#   - After the fill, on a LATER bar, the first bar that closes above BOTH EMAs (long) or below BOTH (short),
#     gap still intact, triggers an entry at the next bar's open, if that open is inside the window
#     and before the Friday flat time.
#   - Stop = sl_pips beyond the fill wick. Target = tp_pips from the actual fill price.
#   - Same-direction trades stack. Opposite-direction signals are skipped while a position is open.
#     A long and a short signal on the same bar are both skipped.
#   - Everything is closed at the Friday flat time (Chicago clock).
# Window = Chicago clock before end_hr OR Tokyo clock at/after tokyo_hr (each in its own timezone,
# so daylight saving is handled automatically).
# ════════════════════════════════════════════════════════════════

GRAN_SECONDS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800}

DEFAULTS = dict(
    tokyo_hr=9,        # Tokyo open hour (JST)
    end_hr=12,         # window end hour (Chicago)
    flat_hr=15,        # Friday flat hour (Chicago)
    flat_min=45,       # Friday flat minute
    ema_fast=9,
    ema_slow=21,
    sl_pips=2.5,       # stop beyond the fill wick
    tp_pips=10.0,      # target from the fill price
    min_gap_pips=0.0,  # 0 = no minimum
    max_gaps=200,      # max gaps tracked at once (oldest dropped)
    max_age=0,         # max gap age in bars, 0 = no limit
    skip_breaks=True,  # ignore gaps that span a weekend/market break
    cost_pips=1.0,     # spread + slippage deducted from every trade
    both_hit="stop",   # if stop and target are both inside one bar: "stop" (conservative) or "target"
    max_open=50,       # max stacked trades
)

TRADE_COLS = [
    "id", "side", "form_i", "fill_i", "signal_i", "entry_i", "exit_i",
    "entry_px", "stop", "tp", "exit_px", "reason", "pips", "gap_top", "gap_bot",
    "form_time", "fill_time", "entry_time", "exit_time", "hours_fill_to_entry",
]


# ───────────────────────────── Data ─────────────────────────────

def oanda_instrument(ticker):
    """'GBPJPY=X' -> 'GBP_JPY'"""
    base = ticker.replace("=X", "")
    return base[:3] + "_" + base[3:6]


def fetch_oanda(instrument, granularity, start, end, token, env="practice"):
    """Download mid-price candles from OANDA, paging 5000 at a time.
    Returns a DataFrame indexed by UTC bar-open time with open/high/low/close."""
    host = "https://api-fxpractice.oanda.com" if env == "practice" else "https://api-fxtrade.oanda.com"
    url = f"{host}/v3/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {token}"}
    step = pd.Timedelta(seconds=GRAN_SECONDS[granularity])

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    end = min(end, pd.Timestamp.now(tz="UTC"))

    cur = start
    rows = []
    while cur < end:
        params = {
            "granularity": granularity,
            "price": "M",
            "from": cur.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count": 5000,
        }
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
        except requests.RequestException as e:
            raise RuntimeError(f"Could not reach OANDA: {e}")
        if r.status_code == 401:
            raise RuntimeError(
                "OANDA rejected the token (401). The token may have expired, or OANDA_ENV does not "
                "match the account type (practice token with practice, live token with live)."
            )
        if r.status_code != 200:
            raise RuntimeError(f"OANDA error {r.status_code}: {r.text[:200]}")

        candles = r.json().get("candles", [])
        if not candles:
            break
        last_t = None
        for cd in candles:
            t = pd.Timestamp(cd["time"])
            last_t = t
            if t >= end or not cd.get("complete", False):
                continue
            m = cd["mid"]
            rows.append((t, float(m["o"]), float(m["h"]), float(m["l"]), float(m["c"])))
        if last_t is None or last_t >= end:
            break
        nxt = last_t + step
        if nxt <= cur:
            break
        cur = nxt

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close"],
                            index=pd.DatetimeIndex([], tz="UTC"))
    df = (pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])
          .drop_duplicates("time").set_index("time").sort_index())
    df.index = pd.DatetimeIndex(df.index)
    return df


# ───────────────────────────── Helpers ─────────────────────────────

def _in_win(index, p):
    chi = index.tz_convert("America/Chicago")
    tk = index.tz_convert("Asia/Tokyo")
    chi_min = np.asarray(chi.hour) * 60 + np.asarray(chi.minute)
    tk_min = np.asarray(tk.hour) * 60 + np.asarray(tk.minute)
    return (chi_min < p["end_hr"] * 60) | (tk_min >= p["tokyo_hr"] * 60)


def _fri_flat(index, p):
    chi = index.tz_convert("America/Chicago")
    chi_min = np.asarray(chi.hour) * 60 + np.asarray(chi.minute)
    return (np.asarray(chi.dayofweek) == 4) & (chi_min >= p["flat_hr"] * 60 + p["flat_min"])


def _check_exit(tr, o, h, l, both_hit):
    """Return (exit_price, reason) if the trade exits during this bar, else None."""
    if tr["dir"] > 0:
        sl_hit = l <= tr["stop"]
        tp_hit = h >= tr["tp"]
        sl_px = min(tr["stop"], o)   # opens beyond the stop -> filled at the open
        tp_px = max(tr["tp"], o)
    else:
        sl_hit = h >= tr["stop"]
        tp_hit = l <= tr["tp"]
        sl_px = max(tr["stop"], o)
        tp_px = min(tr["tp"], o)
    if sl_hit and tp_hit:
        return (sl_px, "SL") if both_hit == "stop" else (tp_px, "TP")
    if sl_hit:
        return (sl_px, "SL")
    if tp_hit:
        return (tp_px, "TP")
    return None


# ───────────────────────────── Engine ─────────────────────────────

def run_backtest(df, pip, tf_seconds, **overrides):
    """df: DataFrame indexed by UTC bar-open time with open/high/low/close.
    pip: pip size (0.0001, or 0.01 for JPY pairs).
    Returns {"trades": DataFrame, "counts": dict, "bars": int}."""
    p = {**DEFAULTS, **overrides}
    n = len(df)
    counts = dict(formed=0, filled=0, dead_outside=0, closed_through=0,
                  signals=0, skipped=0, entries=0)
    if n < 50:
        return {"trades": pd.DataFrame(columns=TRADE_COLS), "counts": counts, "bars": n}

    idx = df.index
    tf = pd.Timedelta(seconds=tf_seconds)
    close_idx = idx + tf
    win_open = _in_win(idx, p)
    next_ok = _in_win(close_idx, p) & ~_fri_flat(close_idx, p)
    flat_close = _fri_flat(close_idx, p)

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    ema_f = df["close"].ewm(span=p["ema_fast"], adjust=False).mean().to_numpy()
    ema_s = df["close"].ewm(span=p["ema_slow"], adjust=False).mean().to_numpy()
    tsec = (idx - idx[0]).total_seconds().to_numpy()

    sl_d = p["sl_pips"] * pip
    tp_d = p["tp_pips"] * pip
    min_gap = p["min_gap_pips"] * pip

    # gap = [bull, top, bot, filled, ext, age, fill_i, form_i]
    gaps = []
    open_tr = []
    pending = []
    pending_flat = False
    trades = []
    tid = 0

    def close_trade(tr, i, px, reason):
        g = tr["gap"]
        pips = tr["dir"] * (px - tr["entry_px"]) / pip - p["cost_pips"]
        trades.append(dict(
            id=tr["id"], side="Long" if tr["dir"] > 0 else "Short",
            form_i=g["form_i"], fill_i=g["fill_i"], signal_i=g["signal_i"],
            entry_i=tr["entry_i"], exit_i=i,
            entry_px=tr["entry_px"], stop=tr["stop"], tp=tr["tp"], exit_px=px,
            reason=reason, pips=pips, gap_top=g["top"], gap_bot=g["bot"],
        ))

    for i in range(n):
        # 1) Orders placed on the previous bar fill at this bar's open
        if pending_flat:
            for tr in open_tr:
                close_trade(tr, i, o[i], "WEEKEND")
            open_tr = []
            pending_flat = False
        if pending:
            for s in pending:
                open_tr.append(dict(id=s["id"], dir=s["dir"], entry_i=i, entry_px=o[i],
                                    stop=s["stop"], tp=o[i] + s["dir"] * tp_d, gap=s["gap"]))
                counts["entries"] += 1
            pending = []

        # 2) Stops and targets during this bar (including trades that just filled)
        still = []
        for tr in open_tr:
            res = _check_exit(tr, o[i], h[i], l[i], p["both_hit"])
            if res is None:
                still.append(tr)
            else:
                close_trade(tr, i, res[0], res[1])
        open_tr = still

        # 3) Update tracked gaps (newest to oldest)
        sigs = []
        for gi in range(len(gaps) - 1, -1, -1):
            g = gaps[gi]
            bull, top, bot = g[0], g[1], g[2]
            g[5] += 1
            drop = False
            closed_through = (c[i] < bot) if bull else (c[i] > top)
            if closed_through:
                drop = True
                counts["closed_through"] += 1
            elif p["max_age"] > 0 and g[5] > p["max_age"]:
                drop = True
            elif not g[3]:
                reached = (l[i] <= bot) if bull else (h[i] >= top)
                if reached:
                    if win_open[i]:
                        g[3] = True
                        g[4] = l[i] if bull else h[i]
                        g[6] = i
                        counts["filled"] += 1
                    else:
                        drop = True          # first fill outside the window: gap is dead
                        counts["dead_outside"] += 1
            elif i > g[6]:
                if bull:
                    reclaimed = c[i] > ema_f[i] and c[i] > ema_s[i]
                else:
                    reclaimed = c[i] < ema_f[i] and c[i] < ema_s[i]
                if reclaimed and next_ok[i]:
                    ext = g[4]
                    stop = ext - sl_d if bull else ext + sl_d
                    sigs.append((bull, stop, g))
                    drop = True              # one trade per gap
            if drop:
                gaps.pop(gi)

        # 4) New gap from bars A, B, C (added after the checks, so it can't fill on its own bar)
        if i >= 2 and win_open[i] and (not p["skip_breaks"] or tsec[i] - tsec[i - 2] <= tf_seconds * 2.5):
            if l[i] > h[i - 2] and c[i - 1] > h[i - 2] and (l[i] - h[i - 2]) >= min_gap:
                gaps.append([True, l[i], h[i - 2], False, np.nan, 0, -1, i])
                counts["formed"] += 1
            if h[i] < l[i - 2] and c[i - 1] < l[i - 2] and (l[i - 2] - h[i]) >= min_gap:
                gaps.append([False, l[i - 2], h[i], False, np.nan, 0, -1, i])
                counts["formed"] += 1
        while len(gaps) > p["max_gaps"]:
            gaps.pop(0)

        # 5) Signals -> orders for the next bar's open
        if sigs:
            has_long = any(s[0] for s in sigs)
            has_short = any(not s[0] for s in sigs)
            conflict = has_long and has_short
            pos = sum(1 if t["dir"] > 0 else -1 for t in open_tr)
            for bull, stop, g in sigs:
                counts["signals"] += 1
                d = 1 if bull else -1
                allowed = (
                    not conflict
                    and ((d > 0 and pos >= 0) or (d < 0 and pos <= 0))
                    and len(open_tr) + len(pending) < p["max_open"]
                    and i + 1 < n
                )
                if allowed:
                    tid += 1
                    pending.append(dict(id=tid, dir=d, stop=stop,
                                        gap=dict(form_i=g[7], fill_i=g[6], signal_i=i, top=g[1], bot=g[2])))
                else:
                    counts["skipped"] += 1

        # 6) Friday flat: close everything at the next bar's open
        if flat_close[i]:
            pending_flat = True

    for tr in open_tr:  # still open when the data ends
        close_trade(tr, n - 1, c[-1], "END")

    if not trades:
        return {"trades": pd.DataFrame(columns=TRADE_COLS), "counts": counts, "bars": n}

    tdf = pd.DataFrame(trades)
    tdf["form_time"] = idx[tdf["form_i"].to_numpy()]
    tdf["fill_time"] = idx[tdf["fill_i"].to_numpy()]
    tdf["entry_time"] = idx[tdf["entry_i"].to_numpy()]
    tdf["exit_time"] = idx[tdf["exit_i"].to_numpy()]
    tdf["hours_fill_to_entry"] = (tdf["entry_time"] - tdf["fill_time"]).dt.total_seconds() / 3600.0
    tdf = tdf.sort_values(["exit_i", "id"]).reset_index(drop=True)
    return {"trades": tdf[TRADE_COLS], "counts": counts, "bars": n}


# ───────────────────────────── Stats ─────────────────────────────

def summarize(trades):
    if trades is None or len(trades) == 0:
        return dict(trades=0, win_rate=0.0, avg_pips=0.0, total_pips=0.0,
                    profit_factor=0.0, max_dd=0.0, med_hours=0.0)
    pips = trades["pips"].to_numpy(float)
    gross_win = pips[pips > 0].sum()
    gross_loss = -pips[pips < 0].sum()
    cum = np.cumsum(pips)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return dict(
        trades=len(pips),
        win_rate=float((pips > 0).mean() * 100),
        avg_pips=float(pips.mean()),
        total_pips=float(pips.sum()),
        profit_factor=float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        max_dd=float((peak - cum).max()),
        med_hours=float(np.nanmedian(trades["hours_fill_to_entry"].to_numpy(float))),
    )


FONT = "Times New Roman, Times, serif"
TEAL = "#00b89c"
RED = "#cc4422"


# ───────────────────────────── Helpers ─────────────────────────────

def _get_secrets():
    try:
        token = st.secrets["OANDA_TOKEN"]
    except Exception:
        return None, "practice"
    try:
        env = st.secrets["OANDA_ENV"]
    except Exception:
        env = "practice"
    return token, env


@st.cache_data(ttl=21600, show_spinner=False)
def _load(instrument, gran, start_iso, end_iso, env, _token):
    return fetch_oanda(instrument, gran, pd.Timestamp(start_iso), pd.Timestamp(end_iso), _token, env)


def _chi(x):
    """UTC -> Chicago wall time (tz removed). Works on a single Timestamp, a DatetimeIndex, or a Series."""
    if isinstance(x, pd.Series):
        return x.dt.tz_convert("America/Chicago").dt.tz_localize(None)
    return x.tz_convert("America/Chicago").tz_localize(None)


def _card(label, value):
    return (f'<div class="stat-card"><div class="stat-label">{label}</div>'
            f'<div class="stat-value">{value}</div></div>')


def _fmt_pf(v):
    return "inf" if np.isinf(v) else f"{v:.2f}"


def _layout(fig, height=360):
    fig.update_layout(
        paper_bgcolor="#080808", plot_bgcolor="#080808",
        font=dict(color="#e0e0e0", family=FONT),
        xaxis=dict(gridcolor="#111111", linecolor="#222", showline=True, tickfont=dict(color="#666")),
        yaxis=dict(gridcolor="#111111", linecolor="#222", showline=True, tickfont=dict(color="#666")),
        legend=dict(bgcolor="#0f0f0f", bordercolor="#222", borderwidth=1),
        margin=dict(l=40, r=40, t=20, b=40), height=height,
    )
    return fig


def _session_of(utc_hour):
    if utc_hour >= 21:
        return "Sydney"
    if utc_hour < 7:
        return "Tokyo"
    if utc_hour < 12:
        return "London"
    if utc_hour < 17:
        return "New York"
    return "Off Hours"


def _group_table(t, key):
    g = t.groupby(key)["pips"]
    out = pd.DataFrame({
        "Trades": g.size(),
        "Win %": g.apply(lambda s: (s > 0).mean() * 100).round(1),
        "Avg pips": g.mean().round(2),
        "Total pips": g.sum().round(1),
    })
    out.index.name = None
    return out


# ───────────────────────────── Trade inspector ─────────────────────────────

def _trade_chart(row, res, token):
    m = res["meta"][row["pair"]]
    p = res["params"]
    df = _load(m["instr"], m["gran"], m["start"], m["end"], m["env"], token)

    form_i, fill_i = int(row["form_i"]), int(row["fill_i"])
    entry_i, exit_i = int(row["entry_i"]), int(row["exit_i"])
    lo = max(0, form_i - 20)
    hi = min(len(df) - 1, exit_i + 20)
    seg = df.iloc[lo:hi + 1]
    x = _chi(seg.index)

    ef = df["close"].ewm(span=p["ema_fast"], adjust=False).mean().iloc[lo:hi + 1]
    es = df["close"].ewm(span=p["ema_slow"], adjust=False).mean().iloc[lo:hi + 1]

    is_long = row["side"] == "Long"
    x_a = _chi(df.index[max(0, form_i - 2)])
    x_fill = _chi(df.index[fill_i])
    x_entry = _chi(df.index[entry_i])
    x_exit = _chi(df.index[exit_i])
    fill_y = df["low"].iloc[fill_i] if is_long else df["high"].iloc[fill_i]

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x, open=seg["open"], high=seg["high"], low=seg["low"], close=seg["close"],
        increasing_line_color=TEAL, decreasing_line_color=RED, name="Price", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=ef, mode="lines", name=f"EMA {p['ema_fast']}",
                             line=dict(color="#e0b44a", width=1)))
    fig.add_trace(go.Scatter(x=x, y=es, mode="lines", name=f"EMA {p['ema_slow']}",
                             line=dict(color="#4a9ee0", width=1)))
    fig.add_shape(type="rect", x0=x_a, x1=x_entry, y0=row["gap_bot"], y1=row["gap_top"],
                  fillcolor="rgba(0,184,156,0.15)" if is_long else "rgba(204,68,34,0.15)",
                  line=dict(color=TEAL if is_long else RED, width=1))
    fig.add_shape(type="line", x0=x_entry, x1=x_exit, y0=row["stop"], y1=row["stop"],
                  line=dict(color=RED, width=1, dash="dot"))
    fig.add_shape(type="line", x0=x_entry, x1=x_exit, y0=row["tp"], y1=row["tp"],
                  line=dict(color=TEAL, width=1, dash="dot"))
    fig.add_trace(go.Scatter(x=[x_fill], y=[fill_y], mode="markers+text", text=["first fill"],
                             textposition="bottom center" if is_long else "top center",
                             marker=dict(color="#ffffff", size=8, symbol="diamond"), name="First fill"))
    fig.add_trace(go.Scatter(x=[x_entry], y=[row["entry_px"]], mode="markers",
                             marker=dict(color="#ffd400", size=11,
                                         symbol="triangle-up" if is_long else "triangle-down"),
                             name="Entry"))
    fig.add_trace(go.Scatter(x=[x_exit], y=[row["exit_px"]], mode="markers",
                             marker=dict(color="#ffffff", size=10, symbol="x"), name=f"Exit ({row['reason']})"))
    _layout(fig, 520)
    fig.update_layout(xaxis_rangeslider_visible=False)
    return fig


# ───────────────────────────── Results ─────────────────────────────

def _show_results(res, token):
    t = res["trades"]
    s = summarize(t)
    cnt = res["counts"]

    if t.empty:
        st.info("The rules produced no trades for this selection. Check the setup counts below, "
                "or widen the date range.")
    else:
        cols = st.columns(6)
        cards = [
            ("Trades", f"{s['trades']}"),
            ("Win rate", f"{s['win_rate']:.1f}%"),
            ("Avg pips / trade", f"{s['avg_pips']:+.2f}"),
            ("Total pips", f"{s['total_pips']:+.1f}"),
            ("Profit factor", _fmt_pf(s["profit_factor"])),
            ("Max drawdown (pips)", f"{s['max_dd']:.1f}"),
        ]
        for col, (lab, val) in zip(cols, cards):
            col.markdown(_card(lab, val), unsafe_allow_html=True)

    cols = st.columns(4)
    setup = [
        ("Gaps formed in window", cnt["formed"]),
        ("First fills in window", cnt["filled"]),
        ("Entries", cnt["entries"]),
        ("Triggers skipped", cnt["skipped"]),
    ]
    for col, (lab, val) in zip(cols, setup):
        col.markdown(_card(lab, f"{val}"), unsafe_allow_html=True)

    if t.empty:
        return

    st.caption(
        f"Pips are after a {res['params']['cost_pips']:.1f}-pip cost per trade. Stacked trades and trades from "
        "the same move are counted separately, so the effective sample is smaller than the trade count."
    )

    # Per-pair table
    if t["pair"].nunique() > 1:
        rows = []
        for name, g in t.groupby("pair"):
            r = summarize(g)
            rows.append({"Pair": name, "Trades": r["trades"], "Win %": round(r["win_rate"], 1),
                         "Avg pips": round(r["avg_pips"], 2), "Total pips": round(r["total_pips"], 1),
                         "PF": round(r["profit_factor"], 2) if np.isfinite(r["profit_factor"]) else None,
                         "Max DD": round(r["max_dd"], 1)})
        st.markdown("**By pair**")
        st.dataframe(pd.DataFrame(rows).set_index("Pair"), use_container_width=True)

    # Equity curve
    ts = t.sort_values("exit_time").reset_index(drop=True)
    fig = go.Figure(go.Scatter(x=_chi(ts["exit_time"]), y=ts["pips"].cumsum(), mode="lines",
                               line=dict(color=TEAL, width=2, shape="hv"), name="Cumulative pips"))
    _layout(fig, 320)
    fig.update_yaxes(title_text="Cumulative pips")
    st.markdown("**Equity curve (pips)**")
    st.plotly_chart(fig, use_container_width=True)

    # Breakdowns
    c1, c2 = st.columns(2)
    tt = t.copy()
    tt["Session (entry, UTC)"] = tt["entry_time"].dt.tz_convert("UTC").dt.hour.map(_session_of)
    tt["Weekday (Chicago)"] = tt["entry_time"].dt.tz_convert("America/Chicago").dt.day_name()
    with c1:
        st.markdown("**By session**")
        st.dataframe(_group_table(tt, "Session (entry, UTC)"), use_container_width=True)
    with c2:
        st.markdown("**By weekday**")
        st.dataframe(_group_table(tt, "Weekday (Chicago)"), use_container_width=True)

    # Fill-to-entry delay
    hist = go.Figure(go.Histogram(x=t["hours_fill_to_entry"], nbinsx=40, marker_color=TEAL))
    _layout(hist, 260)
    hist.update_xaxes(title_text="Hours from first fill to entry")
    hist.update_yaxes(title_text="Trades")
    st.markdown(f"**How long after the fill the entry came** (median {s['med_hours']:.1f} hours)")
    st.plotly_chart(hist, use_container_width=True)

    # Trade list + inspector
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Trades (times in Chicago)**")
    disp = t.copy()
    for col in ["form_time", "fill_time", "entry_time", "exit_time"]:
        disp[col] = _chi(disp[col]).dt.strftime("%Y-%m-%d %H:%M")
    show = disp[["pair", "id", "side", "form_time", "fill_time", "entry_time", "exit_time",
                 "entry_px", "stop", "tp", "exit_px", "reason", "pips", "hours_fill_to_entry"]].copy()
    show["pips"] = show["pips"].round(2)
    show["hours_fill_to_entry"] = show["hours_fill_to_entry"].round(1)
    st.dataframe(show, use_container_width=True, height=300)

    st.markdown("**Inspect a trade**")
    st.caption("Shows the gap box (from bar A to the entry), the first fill, the EMAs, the stop and target, and the exit. "
               "Use it to check that each trade matches the rules.")
    labels = [f"{r.pair} · #{r.id} · {r.side} · {r.entry_time_chi} · {r.pips:+.1f} pips · {r.reason}"
              for r in disp.assign(entry_time_chi=disp["entry_time"], pips=t["pips"]).itertuples()]
    choice = st.selectbox("Trade", labels, label_visibility="collapsed")
    row = t.iloc[labels.index(choice)]
    try:
        st.plotly_chart(_trade_chart(row, res, token), use_container_width=True)
    except Exception as e:
        st.warning(f"Could not draw this trade: {e}")


# ───────────────────────────── Tab ─────────────────────────────

def render_backtest_tab(watchlist):
    pair_options = [n for n, a in watchlist.items() if a["ticker"].endswith("=X")]
    token, env = _get_secrets()

    st.markdown("""
    <div style="margin-bottom:6px;">
        <span style="font-size:20px; font-weight:bold; color:#ffffff; font-family:'Times New Roman',Times,serif;
            text-transform:uppercase; letter-spacing:0.05em;">FVG First Fill + EMA Reclaim &mdash; Backtest</span>
    </div>
    <div style="font-size:13px; color:#555; font-family:'Times New Roman',Times,serif;
        font-style:italic; margin-bottom:20px;">
        Python version of the TradingView strategy, run on OANDA candles. Window: Tokyo open to noon Chicago.
    </div>
    """, unsafe_allow_html=True)

    if not token:
        st.warning("OANDA token not found. Add OANDA_TOKEN to your Streamlit secrets, then reload the app.")

    with st.form("bt_form"):
        c1, c2, c3 = st.columns([3, 1, 2])
        pairs = c1.multiselect("Pairs", pair_options, default=["GBP/JPY"] if "GBP/JPY" in pair_options else None)
        gran = c2.selectbox("Timeframe", list(GRAN_SECONDS.keys()), index=1)
        dr = c3.date_input("Date range", value=(date.today() - timedelta(days=30), date.today()),
                           max_value=date.today())

        with st.expander("Strategy settings", expanded=True):
            r1 = st.columns(5)
            sl = r1[0].number_input("Stop beyond fill wick (pips)", 0.0, 100.0, 2.5, 0.5)
            tp = r1[1].number_input("Target (pips)", 0.5, 500.0, 10.0, 0.5)
            ef = r1[2].number_input("Fast EMA", 1, 500, 9, 1)
            es = r1[3].number_input("Slow EMA", 1, 500, 21, 1)
            cost = r1[4].number_input("Cost per trade (pips)", 0.0, 20.0, 1.0, 0.1,
                                      help="Spread + slippage, deducted from every trade.")
            r2 = st.columns(4)
            tokyo = r2[0].number_input("Tokyo open hour (JST)", 0, 23, 9, 1)
            endh = r2[1].number_input("Window end hour (Chicago)", 1, 23, 12, 1)
            flath = r2[2].number_input("Friday flat hour (Chicago)", 0, 23, 15, 1)
            flatm = r2[3].number_input("Friday flat minute", 0, 59, 45, 1)
            r3 = st.columns(4)
            min_gap = r3[0].number_input("Min gap size (pips, 0 = none)", 0.0, 100.0, 0.0, 0.5)
            max_gaps = r3[1].number_input("Max gaps tracked", 1, 1000, 200, 10)
            max_age = r3[2].number_input("Max gap age (bars, 0 = none)", 0, 100000, 0, 10)
            both = r3[3].selectbox("Stop and target in one bar", ["Stop first (conservative)", "Target first"])
            skip_breaks = st.checkbox("Ignore gaps spanning a weekend or other market break", value=True)

        submitted = st.form_submit_button("Run backtest")

    if submitted:
        if not token:
            st.error("No OANDA token, so no data can be loaded.")
        elif not pairs:
            st.error("Pick at least one pair.")
        elif not isinstance(dr, (tuple, list)) or len(dr) < 2:
            st.error("Pick both a start and an end date.")
        else:
            params = dict(
                sl_pips=sl, tp_pips=tp, ema_fast=int(ef), ema_slow=int(es), cost_pips=cost,
                tokyo_hr=int(tokyo), end_hr=int(endh), flat_hr=int(flath), flat_min=int(flatm),
                min_gap_pips=min_gap, max_gaps=int(max_gaps), max_age=int(max_age),
                skip_breaks=skip_breaks, both_hit="stop" if both.startswith("Stop") else "target",
            )
            start = pd.Timestamp(dr[0], tz="UTC")
            end = pd.Timestamp(dr[1], tz="UTC") + pd.Timedelta(days=1)
            if gran == "M1" and (dr[1] - dr[0]).days > 90:
                st.warning("A 1-minute range this long can be slow to download and run. Consider fewer pairs or days.")

            all_tr, meta = [], {}
            counts = dict(formed=0, filled=0, dead_outside=0, closed_through=0, signals=0, skipped=0, entries=0)
            bar = st.progress(0.0)
            for k, name in enumerate(pairs):
                a = watchlist[name]
                instr = oanda_instrument(a["ticker"])
                pip = 1.0 / a["pip"]
                try:
                    with st.spinner(f"{name}: loading candles..."):
                        df = _load(instr, gran, start.isoformat(), end.isoformat(), env, token)
                except Exception as e:
                    st.error(f"{name}: {e}")
                    bar.progress((k + 1) / len(pairs))
                    continue
                if len(df) < 100:
                    st.warning(f"{name}: not enough candles returned ({len(df)}).")
                    bar.progress((k + 1) / len(pairs))
                    continue
                with st.spinner(f"{name}: running backtest on {len(df):,} bars..."):
                    out = run_backtest(df, pip, GRAN_SECONDS[gran], **params)
                tr = out["trades"].copy()
                tr.insert(0, "pair", name)
                all_tr.append(tr)
                for key in counts:
                    counts[key] += out["counts"][key]
                meta[name] = dict(instr=instr, gran=gran, start=start.isoformat(), end=end.isoformat(), env=env)
                bar.progress((k + 1) / len(pairs))
            bar.empty()

            if all_tr:
                trades = pd.concat(all_tr, ignore_index=True)
                st.session_state["bt"] = dict(trades=trades, counts=counts, meta=meta, params={**DEFAULTS, **params})
            else:
                st.session_state.pop("bt", None)

    res = st.session_state.get("bt")
    if res:
        st.markdown("<hr>", unsafe_allow_html=True)
        _show_results(res, token)


# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Daily High / Low Formation", "Backtest"])

# The Backtest tab is drawn before tab 1's code so that st.stop() inside tab 1 can't block it.
with tab2:
    render_backtest_tab(WATCHLIST)



# ════════════════════════════════════════════════════════════════
# TAB 1 — Daily High / Low Formation
# ════════════════════════════════════════════════════════════════
with tab1:

    col_a, col_b, col_c, col_d = st.columns([4, 2, 1, 1])

    with col_a:
        selected_name = st.selectbox(
            "Asset",
            options=list(WATCHLIST.keys()),
            label_visibility="collapsed"
        )
    with col_b:
        period = st.selectbox(
            "Period",
            ["30d", "60d", "90d"],
            index=1,
            label_visibility="collapsed"
        )
    with col_c:
        show_high = st.checkbox("High", value=True)
        show_low  = st.checkbox("Low",  value=True)
    with col_d:
        if st.button("Refresh"):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"{utc_now.strftime('%H:%M')} UTC")

    @st.cache_data(ttl=3600)
    def fetch_data(ticker, period):
        df = yf.download(ticker, period=period, interval="1h", auto_adjust=True)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("UTC")
        return df

    @st.cache_data(ttl=300)
    def fetch_today(ticker):
        df = yf.download(ticker, period="2d", interval="1h", auto_adjust=True)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("UTC")
        today = datetime.now(timezone.utc).date()
        df = df[df.index.date == today]
        return df

    def get_high_low_times(df):
        df = df.copy()
        df["date"] = df.index.date
        results = []
        for date, group in df.groupby("date"):
            if len(group) < 4:
                continue
            high_idx = group["High"].idxmax()
            low_idx  = group["Low"].idxmin()
            if hasattr(high_idx, 'iloc'):
                high_idx = high_idx.iloc[0]
            if hasattr(low_idx, 'iloc'):
                low_idx = low_idx.iloc[0]
            results.append({
                "date":      date,
                "high_hour": high_idx.hour + high_idx.minute / 60,
                "low_hour":  low_idx.hour  + low_idx.minute  / 60,
            })
        return pd.DataFrame(results)

    asset     = WATCHLIST[selected_name]
    ticker    = asset["ticker"]
    pip       = asset["pip"]
    threshold = asset["threshold"]

    st.markdown(f"""
    <div style="margin-bottom:6px;">
        <span style="font-size:20px; font-weight:bold; color:#ffffff;
            font-family:'Times New Roman',Times,serif; text-transform:uppercase; letter-spacing:0.05em;">
            {selected_name} &mdash; Daily High / Low Formation
        </span>
    </div>
    <div style="font-size:13px; color:#555; font-family:'Times New Roman',Times,serif;
        font-style:italic; margin-bottom:20px;">
        Distribution of when the daily high and low most frequently form &mdash; last {period}
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading data..."):
        df       = fetch_data(ticker, period)
        hl_df    = get_high_low_times(df)
        today_df = fetch_today(ticker)

    if hl_df.empty:
        st.error("No data returned for this asset. Try a different period.")
        st.stop()

    today_high    = float(today_df["High"].max().iloc[0])     if not today_df.empty else None
    today_low     = float(today_df["Low"].min().iloc[0])      if not today_df.empty else None
    current_price = float(today_df["Close"].iloc[-1].iloc[0]) if not today_df.empty else None

    high_peak  = int(round(hl_df["high_hour"].value_counts(bins=24).idxmax().mid))
    low_peak   = int(round(hl_df["low_hour"].value_counts(bins=24).idxmax().mid))
    total_days = len(hl_df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">High Peak Window</div>
            <div class="stat-value">{high_peak:02d}:00 UTC</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Low Peak Window</div>
            <div class="stat-value">{low_peak:02d}:00 UTC</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Days Analyzed</div>
            <div class="stat-value">{total_days}</div>
        </div>""", unsafe_allow_html=True)

    if current_price and today_high and today_low:
        pips_from_high = abs(current_price - today_high) * pip
        pips_from_low  = abs(current_price - today_low)  * pip

        if pips_from_high <= threshold:
            status_color  = "#ff4444"
            status_label  = "Near Daily High"
            status_bg     = "rgba(255,68,68,0.06)"
            status_border = "#ff4444"
        elif pips_from_low <= threshold:
            status_color  = "#00b89c"
            status_label  = "Near Daily Low"
            status_bg     = "rgba(0,184,156,0.06)"
            status_border = "#00b89c"
        else:
            status_color  = "#888"
            status_label  = "Mid Range"
            status_bg     = "rgba(255,255,255,0.02)"
            status_border = "#333"

        daily_range     = (today_high - today_low) * pip
        range_completed = ((current_price - today_low) / (today_high - today_low) * 100) if today_high != today_low else 0

        st.markdown(f"""
        <div class="status-card" style="background:{status_bg}; border-left-color:{status_border};">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
                <div style="display:flex; align-items:center;">
                    <span class="live-dot"></span>
                    <span style="font-size:12px; font-weight:bold; color:{status_color};
                        font-family:'Times New Roman',Times,serif; text-transform:uppercase;
                        letter-spacing:0.1em;">{status_label}</span>
                </div>
                <div style="display:flex; gap:36px; flex-wrap:wrap;">
                    <div>
                        <div class="stat-label">Current Price</div>
                        <div style="font-size:15px; font-weight:bold; color:#fff;
                            font-family:'Times New Roman',Times,serif;">{current_price:.4f}</div>
                    </div>
                    <div>
                        <div class="stat-label">Today's High</div>
                        <div style="font-size:15px; font-weight:bold; color:#ff4444;
                            font-family:'Times New Roman',Times,serif;">
                            {today_high:.4f}
                            <span style="font-size:11px; color:#555; font-style:italic;">
                                {pips_from_high:.1f} pips away</span>
                        </div>
                    </div>
                    <div>
                        <div class="stat-label">Today's Low</div>
                        <div style="font-size:15px; font-weight:bold; color:#00b89c;
                            font-family:'Times New Roman',Times,serif;">
                            {today_low:.4f}
                            <span style="font-size:11px; color:#555; font-style:italic;">
                                {pips_from_low:.1f} pips away</span>
                        </div>
                    </div>
                    <div>
                        <div class="stat-label">Day Range</div>
                        <div style="font-size:15px; font-weight:bold; color:#fff;
                            font-family:'Times New Roman',Times,serif;">{daily_range:.1f} pips</div>
                    </div>
                    <div>
                        <div class="stat-label">Range Position</div>
                        <div style="font-size:15px; font-weight:bold; color:#fff;
                            font-family:'Times New Roman',Times,serif;">{range_completed:.0f}%</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    x   = np.linspace(0, 24, 500)
    fig = go.Figure()

    if show_high:
        kde_high = gaussian_kde(hl_df["high_hour"], bw_method=0.3)
        fig.add_trace(go.Scatter(
            x=x, y=kde_high(x),
            mode="lines",
            name="Historical High",
            line=dict(color="#00b89c", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,184,156,0.08)",
            yaxis="y1"
        ))

    if show_low:
        kde_low = gaussian_kde(hl_df["low_hour"], bw_method=0.3)
        fig.add_trace(go.Scatter(
            x=x, y=kde_low(x),
            mode="lines",
            name="Historical Low",
            line=dict(color="#cc4422", width=2, dash="dash"),
            fill="tozeroy",
            fillcolor="rgba(204,68,34,0.08)",
            yaxis="y1"
        ))

    sessions = [
        (0,  7,  "rgba(255,255,255,0.015)", "Tokyo"),
        (7,  12, "rgba(255,160,40,0.04)",   "London"),
        (12, 17, "rgba(90,70,200,0.04)",    "New York"),
        (21, 24, "rgba(255,255,255,0.015)", "Sydney"),
    ]
    for start, end, color, label in sessions:
        fig.add_vrect(x0=start, x1=end, fillcolor=color, line_width=0,
                      annotation_text=label, annotation_position="top left",
                      annotation=dict(font=dict(size=10, color="#444",
                                                family="Times New Roman, Times, serif")))

    fig.add_vline(
        x=datetime.now(timezone.utc).hour + datetime.now(timezone.utc).minute / 60,
        line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.15)",
        annotation_text="Now", annotation_position="top",
        annotation=dict(font=dict(size=10, color="rgba(255,255,255,0.25)",
                                   family="Times New Roman, Times, serif"))
    )

    fig.update_layout(
        paper_bgcolor="#080808", plot_bgcolor="#080808",
        font=dict(color="#e0e0e0", family="Times New Roman, Times, serif"),
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(0, 25)),
            ticktext=[f"{h:02d}:00" for h in range(0, 25)],
            gridcolor="#111111",
            title=dict(text="Time of Day (UTC)",
                       font=dict(color="#555", family="Times New Roman, Times, serif")),
            tickfont=dict(color="#666", family="Times New Roman, Times, serif"),
            linecolor="#222", showline=True
        ),
        yaxis=dict(
            gridcolor="#111111",
            title=dict(text="Density",
                       font=dict(color="#555", family="Times New Roman, Times, serif")),
            tickfont=dict(color="#666", family="Times New Roman, Times, serif"),
            linecolor="#222", showline=True
        ),
        legend=dict(bgcolor="#0f0f0f", bordercolor="#222", borderwidth=1,
                    font=dict(family="Times New Roman, Times, serif", size=12)),
        margin=dict(l=40, r=40, t=20, b=60),
        height=440
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:13px; font-weight:bold; color:#888; font-family:'Times New Roman',Times,serif;
         text-transform:uppercase; letter-spacing:0.1em; margin-bottom:12px;">
        UTC &rarr; Chicago Time
    </div>
    """, unsafe_allow_html=True)

    utc_hours = [f"{h:02d}:00" for h in range(24)]
    col_utc, col_arrow, col_chi = st.columns([2, 1, 2])

    with col_utc:
        selected_utc = st.selectbox("UTC Time", utc_hours, label_visibility="collapsed")
    with col_arrow:
        st.markdown("""
        <div style='text-align:center; font-size:20px; padding-top:8px; color:#444;
             font-family:"Times New Roman",Times,serif;'>&rarr;</div>
        """, unsafe_allow_html=True)
    with col_chi:
        utc_hour = int(selected_utc.split(":")[0])
        utc_time = datetime.now(pytz.utc).replace(hour=utc_hour, minute=0, second=0, microsecond=0)
        chi_time = utc_time.astimezone(pytz.timezone("America/Chicago"))
        st.markdown(f"""
        <div style='font-size:20px; font-weight:bold; color:#00b89c; padding-top:6px;
             font-family:"Times New Roman",Times,serif;'>{chi_time.strftime('%I:%M %p')} Chicago</div>
        """, unsafe_allow_html=True)
