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
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto"
)

# ── Custom styling ────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0d0d0d; }
    .block-container { padding-top: 0rem !important; }
    section[data-testid="stSidebar"] { background-color: #111111; }
    html, body, [class*="css"] { color: #e0e0e0; font-family: 'Inter', sans-serif; }
    .stat-card {
        background: #161616;
        border: 1px solid #222;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .stat-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.08em; }
    .stat-value { font-size: 24px; font-weight: 600; color: #ffffff; margin-top: 4px; }
    .status-card {
        border-radius: 10px;
        padding: 14px 20px;
        margin-bottom: 12px;
        border: 1px solid #222;
    }
    .session-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.4; }
        100% { opacity: 1; }
    }
    .live-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #ff4444;
        animation: pulse 1.5s infinite;
        margin-right: 6px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
    "Gold":     {"ticker": "GC=F",      "pip": 10,    "threshold": 2},
    "S&P 500":  {"ticker": "ES=F",      "pip": 1,     "threshold": 5},
}

# ── Session detector ──────────────────────────────────────────
def get_current_session():
    utc_hour = datetime.now(timezone.utc).hour
    if 22 <= utc_hour or utc_hour < 2:
        return "Sydney", "#4A9EE0"
    elif 2 <= utc_hour < 8:
        return "Tokyo", "#E0B44A"
    elif 8 <= utc_hour < 13:
        return "London", "#E07A4A"
    elif 13 <= utc_hour < 17:
        return "New York", "#7A4AE0"
    else:
        return "Off Hours", "#555555"

# ── Top header ────────────────────────────────────────────────
session_name, session_color = get_current_session()
utc_now = datetime.now(timezone.utc)

st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:space-between; 
     padding: 8px 0 20px 0; border-bottom: 1px solid #222; margin-bottom: 24px;">
    <div style="font-size:22px; font-weight:700; color:#ffffff; font-family:'Georgia', serif;">Okereke Capital</div>
    <div style="display:flex; align-items:center; gap:16px;">
        <span class="session-badge" style="background:{session_color}22; color:{session_color}; border:1px solid {session_color}44;">
            ● {session_name} Session
        </span>
        <div style="font-size:12px; color:#555;">{utc_now.strftime('%H:%M UTC')} · Market Analytics</div>
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
    if st.button("🔄 Refresh"):
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
asset    = WATCHLIST[selected_name]
ticker   = asset["ticker"]
pip      = asset["pip"]
threshold = asset["threshold"]

st.markdown(f"## {selected_name} — Daily High/Low Formation")
st.markdown(f"*Distribution of when the daily high and low most frequently form — last {period}*")

with st.spinner("Loading data..."):
    df       = fetch_data(ticker, period)
    hl_df    = get_high_low_times(df)
    today_df = fetch_today(ticker)

if hl_df.empty:
    st.error("No data returned for this asset. Try a different period.")
    st.stop()

# ── Today's live data ─────────────────────────────────────────
today_high    = float(today_df["High"].max().iloc[0])  if not today_df.empty else None
today_low     = float(today_df["Low"].min().iloc[0])   if not today_df.empty else None
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
        status_label = "⚠️ Near Daily High"
        status_bg    = "rgba(255,68,68,0.08)"
    elif pips_from_low <= threshold:
        status_color = "#00b89c"
        status_label = "⚠️ Near Daily Low"
        status_bg    = "rgba(0,184,156,0.08)"
    else:
        status_color = "#888"
        status_label = "Mid Range"
        status_bg    = "rgba(255,255,255,0.03)"

    daily_range     = (today_high - today_low) * pip
    range_completed = ((current_price - today_low) / (today_high - today_low) * 100) if today_high != today_low else 0

    st.markdown(f"""
    <div class="status-card" style="background:{status_bg}; border-color:{status_color}44;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
            <div>
                <span class="live-dot"></span>
                <span style="font-size:13px; font-weight:600; color:{status_color};">{status_label}</span>
            </div>
            <div style="display:flex; gap:32px; flex-wrap:wrap;">
                <div>
                    <div class="stat-label">Current Price</div>
                    <div style="font-size:16px; font-weight:600; color:#fff;">{current_price:.4f}</div>
                </div>
                <div>
                    <div class="stat-label">Today's High</div>
                    <div style="font-size:16px; font-weight:600; color:#ff4444;">{today_high:.4f} <span style="font-size:11px; color:#888;">({pips_from_high:.1f} pips away)</span></div>
                </div>
                <div>
                    <div class="stat-label">Today's Low</div>
                    <div style="font-size:16px; font-weight:600; color:#00b89c;">{today_low:.4f} <span style="font-size:11px; color:#888;">({pips_from_low:.1f} pips away)</span></div>
                </div>
                <div>
                    <div class="stat-label">Day Range</div>
                    <div style="font-size:16px; font-weight:600; color:#fff;">{daily_range:.1f} pips</div>
                </div>
                <div>
                    <div class="stat-label">Range Position</div>
                    <div style="font-size:16px; font-weight:600; color:#fff;">{range_completed:.0f}%</div>
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
        line=dict(color="#D85A30", width=2, dash="dash"),
        fill="tozeroy",
        fillcolor="rgba(216,90,48,0.08)",
        yaxis="y1"
    ))

# ── Today's live overlay ──────────────────────────────────────
if not today_df.empty:
    today_hours  = [t.hour + t.minute / 60 for t in today_df.index]
    today_highs  = today_df["High"].iloc[:, 0].cummax().tolist()
    today_lows   = today_df["Low"].iloc[:, 0].cummin().tolist()
    current_hour = datetime.now(timezone.utc).hour + datetime.now(timezone.utc).minute / 60

    fig.add_trace(go.Scatter(
        x=today_hours, y=today_highs,
        mode="lines",
        name="Today's High",
        line=dict(color="#ff4444", width=1.5),
        yaxis="y2",
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=today_hours, y=today_lows,
        mode="lines",
        name="Today's Low",
        line=dict(color="#00eeff", width=1.5),
        yaxis="y2",
        showlegend=False
    ))

# Session shading
sessions = [
    (0,  2,  "rgba(255,255,255,0.02)", "Sydney"),
    (2,  8,  "rgba(255,255,255,0.02)", "Tokyo"),
    (8,  13, "rgba(255,180,50,0.05)",  "London"),
    (13, 17, "rgba(100,80,220,0.05)",  "New York"),
]
for start, end, color, label in sessions:
    fig.add_vrect(x0=start, x1=end, fillcolor=color, line_width=0,
                  annotation_text=label, annotation_position="top left",
                  annotation=dict(font=dict(size=10, color="#555")))

# current time vertical line
fig.add_vline(
    x=current_hour if not today_df.empty else datetime.now(timezone.utc).hour,
    line_width=1,
    line_dash="dot",
    line_color="rgba(255,255,255,0.2)",
    annotation_text="Now",
    annotation_position="top",
    annotation=dict(font=dict(size=10, color="rgba(255,255,255,0.3)"))
)

fig.update_layout(
    paper_bgcolor="#0d0d0d",
    plot_bgcolor="#0d0d0d",
    font=dict(color="#e0e0e0"),
    xaxis=dict(
        tickmode="array",
        tickvals=list(range(0, 25)),
        ticktext=[f"{h:02d}:00" for h in range(0, 25)],
        gridcolor="#1a1a1a",
        title=dict(text="Time of Day (UTC)", font=dict(color="#888")),
        tickfont=dict(color="#888")
    ),
    yaxis=dict(
        gridcolor="#1a1a1a",
        title=dict(text="Density (Historical)", font=dict(color="#888")),
        tickfont=dict(color="#888")
    ),
    yaxis2=dict(
        title=dict(text="", font=dict(color="#aaa")),
        tickfont=dict(color="#aaa"),
        overlaying="y",
        side="right",
        showgrid=False,
        showticklabels=False
    ),
    legend=dict(bgcolor="#111", bordercolor="#222", borderwidth=1),
    margin=dict(l=40, r=60, t=20, b=60),
    height=440
)

st.plotly_chart(fig, use_container_width=True)

# ── UTC to Chicago Time Converter ────────────────────────────
st.markdown("---")
st.markdown("##### 🕐 UTC → Chicago Time Converter")

utc_hours = [f"{h:02d}:00" for h in range(24)]
col_utc, col_arrow, col_chi = st.columns([2, 1, 2])

with col_utc:
    selected_utc = st.selectbox("UTC Time", utc_hours, label_visibility="collapsed")
with col_arrow:
    st.markdown("<div style='text-align:center; font-size:24px; padding-top:4px;'>→</div>", unsafe_allow_html=True)
with col_chi:
    utc_hour   = int(selected_utc.split(":")[0])
    utc_time   = datetime.now(pytz.utc).replace(hour=utc_hour, minute=0, second=0, microsecond=0)
    chi_time   = utc_time.astimezone(pytz.timezone("America/Chicago"))
    st.markdown(f"<div style='font-size:20px; font-weight:600; color:#00b89c; padding-top:6px;'>{chi_time.strftime('%I:%M %p')} Chicago</div>", unsafe_allow_html=True)