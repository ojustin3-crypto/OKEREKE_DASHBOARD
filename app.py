import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from datetime import datetime, timezone
import pytz
import base64

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="OCAP Dash#1",
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

# ── All 28 forex pairs for strength meter ─────────────────────
FOREX_PAIRS = {
    "EUR/USD": {"ticker": "EURUSD=X", "base": "EUR", "quote": "USD"},
    "EUR/GBP": {"ticker": "EURGBP=X", "base": "EUR", "quote": "GBP"},
    "EUR/JPY": {"ticker": "EURJPY=X", "base": "EUR", "quote": "JPY"},
    "EUR/CHF": {"ticker": "EURCHF=X", "base": "EUR", "quote": "CHF"},
    "EUR/CAD": {"ticker": "EURCAD=X", "base": "EUR", "quote": "CAD"},
    "EUR/AUD": {"ticker": "EURAUD=X", "base": "EUR", "quote": "AUD"},
    "EUR/NZD": {"ticker": "EURNZD=X", "base": "EUR", "quote": "NZD"},
    "GBP/USD": {"ticker": "GBPUSD=X", "base": "GBP", "quote": "USD"},
    "GBP/JPY": {"ticker": "GBPJPY=X", "base": "GBP", "quote": "JPY"},
    "GBP/CHF": {"ticker": "GBPCHF=X", "base": "GBP", "quote": "CHF"},
    "GBP/CAD": {"ticker": "GBPCAD=X", "base": "GBP", "quote": "CAD"},
    "GBP/AUD": {"ticker": "GBPAUD=X", "base": "GBP", "quote": "AUD"},
    "GBP/NZD": {"ticker": "GBPNZD=X", "base": "GBP", "quote": "NZD"},
    "CHF/JPY": {"ticker": "CHFJPY=X", "base": "CHF", "quote": "JPY"},
    "USD/JPY": {"ticker": "USDJPY=X", "base": "USD", "quote": "JPY"},
    "USD/CHF": {"ticker": "USDCHF=X", "base": "USD", "quote": "CHF"},
    "USD/CAD": {"ticker": "USDCAD=X", "base": "USD", "quote": "CAD"},
    "CAD/JPY": {"ticker": "CADJPY=X", "base": "CAD", "quote": "JPY"},
    "CAD/CHF": {"ticker": "CADCHF=X", "base": "CAD", "quote": "CHF"},
    "AUD/USD": {"ticker": "AUDUSD=X", "base": "AUD", "quote": "USD"},
    "AUD/JPY": {"ticker": "AUDJPY=X", "base": "AUD", "quote": "JPY"},
    "AUD/CHF": {"ticker": "AUDCHF=X", "base": "AUD", "quote": "CHF"},
    "AUD/CAD": {"ticker": "AUDCAD=X", "base": "AUD", "quote": "CAD"},
    "AUD/NZD": {"ticker": "AUDNZD=X", "base": "AUD", "quote": "NZD"},
    "NZD/USD": {"ticker": "NZDUSD=X", "base": "NZD", "quote": "USD"},
    "NZD/JPY": {"ticker": "NZDJPY=X", "base": "NZD", "quote": "JPY"},
    "NZD/CHF": {"ticker": "NZDCHF=X", "base": "NZD", "quote": "CHF"},
    "NZD/CAD": {"ticker": "NZDCAD=X", "base": "NZD", "quote": "CAD"},
}

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF"]

# ── Timeframe config ──────────────────────────────────────────
TF_CONFIG = {
    "30m":    {"interval": "30m",  "period": "7d",  "resample": None},
    "1H":     {"interval": "1h",   "period": "30d", "resample": None},
    "4H":     {"interval": "1h",   "period": "60d", "resample": "4h"},
    "Daily":  {"interval": "1d",   "period": "1y",  "resample": None},
    "Weekly": {"interval": "1wk",  "period": "2y",  "resample": None},
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
        text-transform: uppercase;
    ">Dashboard1</div>
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

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Daily High / Low Formation", "Currency Strength"])


# ════════════════════════════════════════════════════════════════
# TAB 1 — existing dashboard
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


# ════════════════════════════════════════════════════════════════
# TAB 2 — Currency Strength Meter
# ════════════════════════════════════════════════════════════════
with tab2:

    # ── Controls ──────────────────────────────────────────────
    cs_col1, cs_col2 = st.columns([2, 6])

    with cs_col1:
        timeframe = st.selectbox(
            "Timeframe",
            options=["30m", "1H", "4H", "Daily", "Weekly"],
            index=1,
            label_visibility="collapsed"
        )
    with cs_col2:
        if st.button("Refresh ", key="cs_refresh"):
            st.cache_data.clear()
            st.rerun()

    st.markdown(f"""
    <div style="margin-bottom:6px; margin-top:8px;">
        <span style="font-size:20px; font-weight:bold; color:#ffffff;
            font-family:'Times New Roman',Times,serif; text-transform:uppercase; letter-spacing:0.05em;">
            Currency Strength &mdash; {timeframe}
        </span>
    </div>
    <div style="font-size:13px; color:#555; font-family:'Times New Roman',Times,serif;
        font-style:italic; margin-bottom:24px;">
        Last closed {timeframe} candle scored across all 28 major pairs &mdash;
        bullish candle: base +1 / quote &minus;1 &nbsp;&middot;&nbsp;
        bearish candle: base &minus;1 / quote +1
    </div>
    """, unsafe_allow_html=True)

    # ── Fetch & score ─────────────────────────────────────────
    @st.cache_data(ttl=300)
    def fetch_strength_data(timeframe):
        cfg = TF_CONFIG[timeframe]
        scores  = {c: 0 for c in CURRENCIES}
        results = {}
        failed  = []

        tickers = [v["ticker"] for v in FOREX_PAIRS.values()]
        pair_list = list(FOREX_PAIRS.keys())

        try:
            raw = yf.download(
                tickers,
                period=cfg["period"],
                interval=cfg["interval"],
                auto_adjust=True,
                group_by="ticker",
                progress=False
            )
        except Exception:
            return scores, {}, []

        for pair_name in pair_list:
            pair_info = FOREX_PAIRS[pair_name]
            tkr       = pair_info["ticker"]
            base      = pair_info["base"]
            quote     = pair_info["quote"]

            try:
                if len(tickers) == 1:
                    df = raw.copy()
                else:
                    df = raw[tkr].copy()

                df = df.dropna(subset=["Open", "Close"])

                # Resample to 4H if needed
                if cfg["resample"]:
                    df = df.resample(cfg["resample"]).agg({
                        "Open":   "first",
                        "High":   "max",
                        "Low":    "min",
                        "Close":  "last",
                        "Volume": "sum"
                    }).dropna(subset=["Open", "Close"])

                if len(df) < 2:
                    failed.append(pair_name)
                    continue

                # Last fully closed candle = second to last row
                candle = df.iloc[-2]
                o = float(candle["Open"])
                c = float(candle["Close"])

                direction = 1 if c > o else -1  # +1 bullish, -1 bearish

                scores[base]  += direction
                scores[quote] -= direction

                results[pair_name] = {
                    "base":      base,
                    "quote":     quote,
                    "direction": direction,
                    "open":      o,
                    "close":     c,
                }

            except Exception:
                failed.append(pair_name)
                continue

        return scores, results, failed

    with st.spinner("Fetching candle data..."):
        scores, pair_results, failed_pairs = fetch_strength_data(timeframe)

    if not pair_results:
        st.error("Could not retrieve data. Check your connection or try a different timeframe.")
        st.stop()

    # ── Currency strength bar chart ───────────────────────────
    sorted_currencies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    currencies_ordered = [c for c, _ in sorted_currencies]
    score_values       = [s for _, s in sorted_currencies]
    bar_colors         = ["#00b89c" if s > 0 else "#cc4422" if s < 0 else "#333333"
                          for s in score_values]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=currencies_ordered,
        y=score_values,
        marker_color=bar_colors,
        marker_line_width=0,
        text=[f"{s:+d}" for s in score_values],
        textposition="outside",
        textfont=dict(
            color=bar_colors,
            size=13,
            family="Times New Roman, Times, serif"
        ),
    ))

    fig2.update_layout(
        paper_bgcolor="#080808",
        plot_bgcolor="#080808",
        font=dict(color="#e0e0e0", family="Times New Roman, Times, serif"),
        xaxis=dict(
            gridcolor="#111",
            tickfont=dict(color="#aaa", size=13, family="Times New Roman, Times, serif"),
            linecolor="#222",
            showline=True,
        ),
        yaxis=dict(
            gridcolor="#111111",
            tickfont=dict(color="#666", family="Times New Roman, Times, serif"),
            linecolor="#222",
            showline=True,
            zeroline=True,
            zerolinecolor="#2a2a2a",
            zerolinewidth=1,
        ),
        margin=dict(l=40, r=40, t=20, b=40),
        height=320,
        showlegend=False,
        bargap=0.35,
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ── Pair imbalance table ──────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:13px; font-weight:bold; color:#888; font-family:'Times New Roman',Times,serif;
         text-transform:uppercase; letter-spacing:0.1em; margin-bottom:16px;">
        Pair Imbalance &mdash; Ranked by Spread
    </div>
    """, unsafe_allow_html=True)

    # Build imbalance rows
    imbalance_rows = []
    for pair_name, info in pair_results.items():
        base_score  = scores[info["base"]]
        quote_score = scores[info["quote"]]
        spread      = base_score - quote_score
        imbalance_rows.append({
            "pair":        pair_name,
            "base":        info["base"],
            "quote":       info["quote"],
            "base_score":  base_score,
            "quote_score": quote_score,
            "spread":      spread,
            "direction":   info["direction"],
        })

    imbalance_rows.sort(key=lambda x: abs(x["spread"]), reverse=True)

    # Table header
    st.markdown("""
    <div style="display:flex; align-items:center; justify-content:space-between;
         padding: 6px 16px; border-bottom: 1px solid #222; margin-bottom:4px;">
        <span style="font-size:10px; color:#444; text-transform:uppercase;
            letter-spacing:0.12em; font-family:'Times New Roman',Times,serif;
            font-style:italic; width:120px;">Pair</span>
        <span style="font-size:10px; color:#444; text-transform:uppercase;
            letter-spacing:0.12em; font-family:'Times New Roman',Times,serif;
            font-style:italic; width:80px; text-align:center;">Candle</span>
        <span style="font-size:10px; color:#444; text-transform:uppercase;
            letter-spacing:0.12em; font-family:'Times New Roman',Times,serif;
            font-style:italic; width:80px; text-align:center;">Base Score</span>
        <span style="font-size:10px; color:#444; text-transform:uppercase;
            letter-spacing:0.12em; font-family:'Times New Roman',Times,serif;
            font-style:italic; width:80px; text-align:center;">Quote Score</span>
        <span style="font-size:10px; color:#444; text-transform:uppercase;
            letter-spacing:0.12em; font-family:'Times New Roman',Times,serif;
            font-style:italic; width:80px; text-align:right;">Spread</span>
    </div>
    """, unsafe_allow_html=True)

    for row in imbalance_rows:
        candle_color = "#00b89c" if row["direction"] == 1 else "#cc4422"
        candle_label = "▲ Bull" if row["direction"] == 1 else "▼ Bear"
        spread_color = "#00b89c" if row["spread"] > 0 else "#cc4422" if row["spread"] < 0 else "#555"

        # Highlight rows with high absolute spread
        abs_spread  = abs(row["spread"])
        row_bg      = "rgba(0,184,156,0.04)" if abs_spread >= 8 else "transparent"
        row_border  = "border-left: 3px solid #00b89c;" if abs_spread >= 8 else "border-left: 3px solid transparent;"

        st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:space-between;
             padding: 9px 16px; border-bottom: 1px solid #111;
             background:{row_bg}; {row_border}">
            <span style="font-size:14px; font-weight:bold; color:#fff;
                font-family:'Times New Roman',Times,serif; width:120px;">
                {row['pair']}
            </span>
            <span style="font-size:12px; color:{candle_color}; font-weight:bold;
                font-family:'Times New Roman',Times,serif; width:80px; text-align:center;">
                {candle_label}
            </span>
            <span style="font-size:13px; color:#ccc;
                font-family:'Times New Roman',Times,serif; width:80px; text-align:center;">
                {row['base']} &nbsp;
                <span style="color:{'#00b89c' if row['base_score'] > 0 else '#cc4422' if row['base_score'] < 0 else '#555'}; font-weight:bold;">
                    {row['base_score']:+d}
                </span>
            </span>
            <span style="font-size:13px; color:#ccc;
                font-family:'Times New Roman',Times,serif; width:80px; text-align:center;">
                {row['quote']} &nbsp;
                <span style="color:{'#00b89c' if row['quote_score'] > 0 else '#cc4422' if row['quote_score'] < 0 else '#555'}; font-weight:bold;">
                    {row['quote_score']:+d}
                </span>
            </span>
            <span style="font-size:14px; font-weight:bold; color:{spread_color};
                font-family:'Times New Roman',Times,serif; width:80px; text-align:right;">
                {row['spread']:+d}
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Footer notes ──────────────────────────────────────────
    if failed_pairs:
        st.markdown(f"""
        <div style="margin-top:16px; font-size:11px; color:#333;
             font-family:'Times New Roman',Times,serif; font-style:italic;">
            Could not fetch: {', '.join(failed_pairs)}
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:20px; font-size:11px; color:#333;
         font-family:'Times New Roman',Times,serif; font-style:italic;">
        Scores based on last closed {timeframe} candle &nbsp;&middot;&nbsp;
        Rows highlighted teal where |spread| &ge; 8 &nbsp;&middot;&nbsp;
        Data via yfinance &nbsp;&middot;&nbsp; Refreshed {utc_now.strftime('%H:%M UTC')}
    </div>
    """, unsafe_allow_html=True)
