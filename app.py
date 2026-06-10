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
    page_title="Okereke Capital | Market Analytics",
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

    /* Stat cards — sharp, left teal border */
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

    /* Session badge — sharp */
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

    /* Divider */
    hr {
        border-color: #1a1a1a !important;
        margin: 24px 0 !important;
    }

    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Streamlit widget font override */
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

    /* Caption */
    .stCaption {
        font-family: 'Times New Roman', Times, serif !important;
        color: #555 !important;
        font-style: italic;
    }
    /* Sharp dropdown corners */
    .stSelectbox > div > div {
        border-radius: 0 !important;
        border: 1px solid #333 !important;
        background-color: #0f0f0f !important;
        font-family: 'Times New Roman', Times, serif !important;
    }
    .stSelectbox > div > div:hover {
        border-color: #00b89c !important;
    }
    [data-baseweb="select"] {
        border-radius: 0 !important;
    }
    [data-baseweb="popover"] {
        border-radius: 0 !important;
    }
    [data-baseweb="menu"] {
        border-radius: 0 !important;
        background-color: #0f0f0f !important;
        border: 1px solid #222 !important;
    }
    [data-baseweb="option"] {
        background-color: #0f0f0f !important;
        font-family: 'Times New Roman', Times, serif !important;
    }
    [data-baseweb="option"]:hover {
        background-color: #1a1a1a !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Watchlist ─────────────────────────────────────────────────
WATCHLIST = {
    "GBP/JPY":  {"ticker": "GBPJPY=X",  "pip": 100,   "threshold": 10},
    "AUD/USD":  {"ticker": "AUDUSD=X",  "pip": 10000, "threshold": 10},
    "EUR/USD":  {"ticker": "EURUSD=X",  "pip": 10000, "threshold": 10},
    "USD/JPY":  {"ticker": "USDJPY=X",  "pip": 100,   "threshold": 10},
    "Ethereum": {"ticker": "ETH-USD",   "pip": 1,     "threshold": 50},
    "Bitcoin":  {"ticker": "BTC-USD",   "pip": 1,     "threshold": 500},
    "Gold":     {"ticker": "GC=F",      "pip": 10,    "threshold": 30},
    "S&P 500":  {"ticker": "ES=F",      "pip": 1,     "threshold": 5},
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
    ">Okereke Capital</div>
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

# ── Controls row ──────────────────────────────────────────────
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

# ── Data fetching ─────────────────────────────────────────────
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

# ── Main content ──────────────────────────────────────────────
asset     = WATCHLIST[selected_name]
ticker    = asset["ticker"]
pip       = asset["pip"]
threshold = asset["threshold"]

st.markdown(f"""
<div style="margin-bottom:6px;">
    <span style="font-size:20px; font-weight:bold; color:#ffffff; font-family:'Times New Roman',Times,serif; text-transform:uppercase; letter-spacing:0.05em;">
        {selected_name} &mdash; Daily High / Low Formation
    </span>
</div>
<div style="font-size:13px; color:#555; font-family:'Times New Roman',Times,serif; font-style:italic; margin-bottom:20px;">
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

# ── Today's live data ─────────────────────────────────────────
today_high    = float(today_df["High"].max().iloc[0])     if not today_df.empty else None
today_low     = float(today_df["Low"].min().iloc[0])      if not today_df.empty else None
current_price = float(today_df["Close"].iloc[-1].iloc[0]) if not today_df.empty else None

# ── Stat cards ────────────────────────────────────────────────
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

# ── Live price status ─────────────────────────────────────────
if current_price and today_high and today_low:
    pips_from_high = abs(current_price - today_high) * pip
    pips_from_low  = abs(current_price - today_low)  * pip

    if pips_from_high <= threshold:
        status_color = "#ff4444"
        status_label = "Near Daily High"
        status_bg    = "rgba(255,68,68,0.06)"
        status_border = "#ff4444"
    elif pips_from_low <= threshold:
        status_color = "#00b89c"
        status_label = "Near Daily Low"
        status_bg    = "rgba(0,184,156,0.06)"
        status_border = "#00b89c"
    else:
        status_color = "#888"
        status_label = "Mid Range"
        status_bg    = "rgba(255,255,255,0.02)"
        status_border = "#333"

    daily_range     = (today_high - today_low) * pip
    range_completed = ((current_price - today_low) / (today_high - today_low) * 100) if today_high != today_low else 0

    st.markdown(f"""
    <div class="status-card" style="background:{status_bg}; border-left-color:{status_border};">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
            <div style="display:flex; align-items:center;">
                <span class="live-dot"></span>
                <span style="
                    font-size:12px;
                    font-weight:bold;
                    color:{status_color};
                    font-family:'Times New Roman',Times,serif;
                    text-transform:uppercase;
                    letter-spacing:0.1em;
                ">{status_label}</span>
            </div>
            <div style="display:flex; gap:36px; flex-wrap:wrap;">
                <div>
                    <div class="stat-label">Current Price</div>
                    <div style="font-size:15px; font-weight:bold; color:#fff; font-family:'Times New Roman',Times,serif;">{current_price:.4f}</div>
                </div>
                <div>
                    <div class="stat-label">Today's High</div>
                    <div style="font-size:15px; font-weight:bold; color:#ff4444; font-family:'Times New Roman',Times,serif;">
                        {today_high:.4f}
                        <span style="font-size:11px; color:#555; font-style:italic;"> {pips_from_high:.1f} pips away</span>
                    </div>
                </div>
                <div>
                    <div class="stat-label">Today's Low</div>
                    <div style="font-size:15px; font-weight:bold; color:#00b89c; font-family:'Times New Roman',Times,serif;">
                        {today_low:.4f}
                        <span style="font-size:11px; color:#555; font-style:italic;"> {pips_from_low:.1f} pips away</span>
                    </div>
                </div>
                <div>
                    <div class="stat-label">Day Range</div>
                    <div style="font-size:15px; font-weight:bold; color:#fff; font-family:'Times New Roman',Times,serif;">{daily_range:.1f} pips</div>
                </div>
                <div>
                    <div class="stat-label">Range Position</div>
                    <div style="font-size:15px; font-weight:bold; color:#fff; font-family:'Times New Roman',Times,serif;">{range_completed:.0f}%</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Distribution chart ────────────────────────────────────────
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


# Session shading
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

# Current time line
fig.add_vline(
    x=datetime.now(timezone.utc).hour + datetime.now(timezone.utc).minute / 60,
    line_width=1,
    line_dash="dot",
    line_color="rgba(255,255,255,0.15)",
    annotation_text="Now",
    annotation_position="top",
    annotation=dict(font=dict(size=10, color="rgba(255,255,255,0.25)",
                               family="Times New Roman, Times, serif"))
)

fig.update_layout(
    paper_bgcolor="#080808",
    plot_bgcolor="#080808",
    font=dict(color="#e0e0e0", family="Times New Roman, Times, serif"),
    xaxis=dict(
        tickmode="array",
        tickvals=list(range(0, 25)),
        ticktext=[f"{h:02d}:00" for h in range(0, 25)],
        gridcolor="#111111",
        title=dict(text="Time of Day (UTC)", font=dict(color="#555",
                   family="Times New Roman, Times, serif")),
        tickfont=dict(color="#666", family="Times New Roman, Times, serif"),
        linecolor="#222",
        showline=True
    ),
    yaxis=dict(
        gridcolor="#111111",
        title=dict(text="Density", font=dict(color="#555",
                   family="Times New Roman, Times, serif")),
        tickfont=dict(color="#666", family="Times New Roman, Times, serif"),
        linecolor="#222",
        showline=True
    ),
    legend=dict(
        bgcolor="#0f0f0f",
        bordercolor="#222",
        borderwidth=1,
        font=dict(family="Times New Roman, Times, serif", size=12)
    ),
    margin=dict(l=40, r=40, t=20, b=60),
    height=440
)

st.plotly_chart(fig, use_container_width=True)

# ── UTC to Chicago Time Converter ────────────────────────────
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