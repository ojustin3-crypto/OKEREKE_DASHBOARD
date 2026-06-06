import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from datetime import datetime
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
    /* Background */
    .stApp { background-color: #0d0d0d; }
    section[data-testid="stSidebar"] { background-color: #111111; }

    /* Text */
    html, body, [class*="css"] { color: #e0e0e0; font-family: 'Inter', sans-serif; }

    /* Sidebar title */
    .sidebar-title {
        font-size: 20px;
        font-weight: 700;
        color: #00b89c;
        padding-bottom: 8px;
        border-bottom: 1px solid #222;
        margin-bottom: 16px;
    }

    /* Stat cards */
    .stat-card {
        background: #161616;
        border: 1px solid #222;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .stat-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.08em; }
    .stat-value { font-size: 24px; font-weight: 600; color: #ffffff; margin-top: 4px; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Watchlist ─────────────────────────────────────────────────
WATCHLIST = {
    "GBP/JPY":       "GBPJPY=X",
    "AUD/USD":       "AUDUSD=X",
    "EUR/USD":       "EURUSD=X",
    "USD/JPY":       "USDJPY=X",
    "Ethereum":      "ETH-USD",
    "Bitcoin":       "BTC-USD",
    "Gold":          "GC=F",
    "S&P 500":       "ES=F",
}

# ── Top header ────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; justify-content:space-between; 
     padding: 8px 0 20px 0; border-bottom: 1px solid #222; margin-bottom: 24px;">
    <div style="font-size:22px; font-weight:700; color:#ffffff; font-family:'Georgia', serif;">Okereke Capital</div>
    <div style="font-size:12px; color:#555;">Market Analytics Dashboard</div>
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
    st.caption(f"{datetime.utcnow().strftime('%H:%M')} UTC")

# ── Data fetching ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_data(ticker, period):
    df = yf.download(ticker, period=period, interval="1h", auto_adjust=True)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("UTC")
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
ticker = WATCHLIST[selected_name]
st.markdown(f"## {selected_name} — Daily High/Low Formation")
st.markdown(f"*Distribution of when the daily high and low most frequently form — last {period}*")

with st.spinner("Loading data..."):
    df = fetch_data(ticker, period)
    hl_df = get_high_low_times(df)

if hl_df.empty:
    st.error("No data returned for this asset. Try a different period.")
    st.stop()

# ── Stat cards ────────────────────────────────────────────────
high_peak = int(round(hl_df["high_hour"].value_counts(bins=24).idxmax().mid))
low_peak  = int(round(hl_df["low_hour"].value_counts(bins=24).idxmax().mid))
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

# ── Distribution chart ────────────────────────────────────────
x = np.linspace(0, 24, 500)
fig = go.Figure()

if show_high:
    kde_high = gaussian_kde(hl_df["high_hour"], bw_method=0.3)
    fig.add_trace(go.Scatter(
        x=x, y=kde_high(x),
        mode="lines",
        name="Daily High",
        line=dict(color="#00b89c", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,184,156,0.1)"
    ))

if show_low:
    kde_low = gaussian_kde(hl_df["low_hour"], bw_method=0.3)
    fig.add_trace(go.Scatter(
        x=x, y=kde_low(x),
        mode="lines",
        name="Daily Low",
        line=dict(color="#D85A30", width=2, dash="dash"),
        fill="tozeroy",
        fillcolor="rgba(216,90,48,0.1)"
    ))

# Session shading
sessions = [
    (0,  2,  "rgba(255,255,255,0.02)", "Sydney"),
    (2,  8,  "rgba(255,255,255,0.02)", "Tokyo"),
    (8,  13, "rgba(255,180,50,0.05)",  "London"),
    (13, 17, "rgba(100,80,220,0.05)",  "New York"),
]
for start, end, color, label in sessions:
    fig.add_vrect(x0=start, x1=end, fillcolor=color, line_width=0, annotation_text=label,
                  annotation_position="top left",
                  annotation=dict(font=dict(size=10, color="#555")))

fig.update_layout(
    paper_bgcolor="#0d0d0d",
    plot_bgcolor="#0d0d0d",
    font=dict(color="#e0e0e0"),
    xaxis=dict(
        tickmode="array",
        tickvals=list(range(0, 25)),
        ticktext=[f"{h:02d}:00" for h in range(0, 25)],
        gridcolor="#1a1a1a",
        title="Time of Day (UTC)"
    ),
    yaxis=dict(gridcolor="#1a1a1a", title="Density"),
    legend=dict(bgcolor="#111", bordercolor="#222", borderwidth=1),
    margin=dict(l=40, r=40, t=20, b=60),
    height=420
)

st.plotly_chart(fig, use_container_width=True)