import streamlit as st
import plotly.graph_objects as go
from signals import fetch_crypto_data, generate_signals

# App Configuration
st.set_page_config(page_title="Crypto Signal Bot", layout="wide")
st.title("⚡ Crypto Signal Dashboard")
st.caption("A real-time dashboard analyzing crypto trends and generating momentum signals.")

# Sidebar Configuration Control Room
st.sidebar.header("🛠️ Dashboard Configurations")
target_asset = st.sidebar.selectbox("Select Crypto Asset", ["BTC-USD", "ETH-USD", "SOL-USD"])
time_window = st.sidebar.selectbox("Lookback Window", ["60d", "30d", "7d"])
data_interval = st.sidebar.selectbox("Data Resolution Interval", ["1h", "15m"])

# Data Processing Pipeline
with st.spinner("Streaming asset data pools..."):
    raw_data = fetch_crypto_data(target_asset, period=time_window, interval=data_interval)
    processed_data = generate_signals(raw_data)

if processed_data.empty:
    st.error("Data tracking failure. Please adjust your target resolution settings.")
else:
    latest_entry = processed_data.iloc[-1]
    prior_entry = processed_data.iloc[-2]
    
    # Hero Metric Layout Displays
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Current Spot Price", f"${latest_entry['Close']:.2f}", f"{(latest_entry['Close'] - prior_entry['Close']):.2f}")
    with m2:
        st.markdown(f"### Current Status\n## {latest_entry['Signal']}")
    with m3:
        st.metric("RSI Value (14)", f"{latest_entry['RSI']:.1f}")
    with m4:
        st.metric("Strategy Momentum Score", int(latest_entry['Score']))

    st.divider()

    # Visual Mapping Interface using Plotly
    st.subheader("📊 Analytical Technical Chart")
    fig = go.Figure()
    
    # Underlying Spot Price line
    fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['Close'], name='Close Price', line=dict(color='#3a86ff', width=2)))
    # Moving Averages
    fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['EMA_fast'], name='12 EMA (Fast)', line=dict(color='#ff006e', width=1.5, dash='dash')))
    fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['EMA_slow'], name='26 EMA (Slow)', line=dict(color='#38b000', width=1.5, dash='dot')))

    fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=20, b=20), height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Detailed Audit Stream
    st.subheader("📋 Historical Signal Audit Log")
    st.dataframe(processed_data[['Close', 'RSI', 'Score', 'Signal']].sort_index(ascending=False), use_container_width=True)
