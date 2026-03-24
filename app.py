import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import griddata

from src.fetch import get_options_chain, get_risk_free_rate
from src.iv import compute_mid, filter_options, compute_iv
from src.surface import select_otm

st.set_page_config(page_title="Vol Surface Visualiser", layout="wide")
st.title("Options Implied Volatility Visualiser")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value="SPY").upper().strip()
    load = st.button("Load Data", type="primary")

if "data" not in st.session_state:
    st.session_state.data = None

if load:
    with st.spinner(f"Fetching options data for {ticker}..."):
        try:
            spot, options = get_options_chain(ticker)
            rfr = get_risk_free_rate()
            options = compute_mid(options)
            options = filter_options(options, spot)
            options = compute_iv(options, spot, rfr)
            otm = select_otm(options, spot)

            x = otm["moneyness"].values
            y = otm["tte"].values
            z = otm["iv"].values
            xi = np.linspace(x.min(), x.max(), 100)
            yi = np.linspace(y.min(), y.max(), 100)
            Xi, Yi = np.meshgrid(xi, yi)
            method = "cubic" if len(x) >= 10 else "linear"
            Zi = griddata((x, y), z, (Xi, Yi), method=method)

            st.session_state.data = dict(
                ticker=ticker, spot=spot, rfr=rfr,
                options=options, otm=otm,
                xi=xi, yi=yi, Xi=Xi, Yi=Yi, Zi=Zi,
            )
        except Exception as e:
            st.error(f"Failed to load data: {e}")

if st.session_state.data is None:
    st.info("Enter a ticker and click **Load Data** to begin.")
    st.stop()

d = st.session_state.data
ticker = d["ticker"]
spot   = d["spot"]
rfr    = d["rfr"]
options = d["options"]
otm    = d["otm"]
xi, yi = d["xi"], d["yi"]
Xi, Yi, Zi = d["Xi"], d["Yi"], d["Zi"]

x = otm["moneyness"].values
y = otm["tte"].values
z = otm["iv"].values

with st.sidebar:
    st.divider()
    st.metric("Spot", f"${spot:.2f}")
    st.metric("Risk-Free Rate", f"{rfr:.2%}")
    st.metric("Contracts (filtered)", len(options))

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Vol Surface", "Options Chain", "Cross Sections"])

# ── Tab 1: Vol Surface ────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"{ticker} — Implied Volatility Surface")

    fig3d = go.Figure(go.Surface(
        x=Xi, y=Yi, z=Zi,
        colorscale="Viridis",
        opacity=0.92,
        colorbar=dict(title="Implied Vol"),
        hovertemplate=(
            "Moneyness: %{x:.3f}<br>"
            "TTE: %{y:.3f} yrs<br>"
            "IV: %{z:.1%}<extra></extra>"
        ),
    ))
    fig3d.update_layout(
        scene=dict(
            xaxis_title="Moneyness (K/S)",
            yaxis_title="Time to Expiry (yrs)",
            zaxis_title="Implied Volatility",
        ),
        height=620,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig3d, use_container_width=True)

    st.subheader("IV Heatmap")
    fig_hm = go.Figure(go.Heatmap(
        x=xi, y=yi, z=Zi,
        colorscale="Viridis",
        colorbar=dict(title="Implied Vol"),
        hovertemplate=(
            "Moneyness: %{x:.3f}<br>"
            "TTE: %{y:.3f} yrs<br>"
            "IV: %{z:.1%}<extra></extra>"
        ),
    ))
    fig_hm.update_layout(
        xaxis_title="Moneyness (K/S)",
        yaxis_title="Time to Expiry (yrs)",
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

# ── Tab 2: Options Chain ──────────────────────────────────────────────────────
with tab2:
    st.subheader(f"{ticker} — Options Chain")

    col1, col2 = st.columns(2)
    with col1:
        expiries = sorted(options["expiry"].unique())
        selected_expiry = st.selectbox("Expiry", expiries)
    with col2:
        flag_map = {"Both": ["c", "p"], "Calls only": ["c"], "Puts only": ["p"]}
        flag_choice = st.radio("Type", list(flag_map.keys()), horizontal=True)

    chain = options[
        (options["expiry"] == selected_expiry) &
        (options["flag"].isin(flag_map[flag_choice]))
    ][["strike", "flag", "bid", "ask", "mid", "volume", "openInterest", "iv"]].copy()

    chain["flag"] = chain["flag"].map({"c": "Call", "p": "Put"})
    chain["iv"] = (chain["iv"] * 100).round(2).astype(str) + "%"
    chain = chain.rename(columns={
        "strike": "Strike", "flag": "Type",
        "bid": "Bid", "ask": "Ask", "mid": "Mid",
        "volume": "Volume", "openInterest": "Open Interest", "iv": "IV",
    })

    st.dataframe(chain, use_container_width=True, hide_index=True)

# ── Tab 3: Cross Sections ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Surface Cross Sections")

    slice_type = st.radio(
        "Slice type",
        ["Vol Smile  —  fix expiry, vary moneyness",
         "Term Structure  —  fix moneyness, vary expiry"],
        horizontal=True,
    )

    if "Vol Smile" in slice_type:
        expiries = sorted(otm["expiry"].unique())
        selected_expiry = st.selectbox("Select expiry", expiries, key="cs_expiry")
        tte_val = otm[otm["expiry"] == selected_expiry]["tte"].iloc[0]

        # Nearest row in the interpolated grid
        row_idx = int(np.argmin(np.abs(yi - tte_val)))
        smile_iv = Zi[row_idx, :]

        actual = otm[otm["expiry"] == selected_expiry]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xi, y=smile_iv,
            mode="lines", name="Interpolated smile",
            line=dict(color="royalblue", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=actual["moneyness"], y=actual["iv"],
            mode="markers", name="Market quotes",
            marker=dict(color="red", size=7),
        ))
        fig.update_layout(
            title=f"Vol Smile — {selected_expiry}  (TTE: {tte_val:.3f} yrs)",
            xaxis_title="Moneyness (K/S)",
            yaxis_title="Implied Volatility",
            height=460,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        m_min, m_max = float(x.min()), float(x.max())
        moneyness_val = st.slider(
            "Moneyness (K/S)", min_value=m_min, max_value=m_max,
            value=round(min(1.0, m_max), 2), step=0.01, format="%.2f",
        )

        # Nearest column in the interpolated grid
        col_idx = int(np.argmin(np.abs(xi - moneyness_val)))
        term_iv = Zi[:, col_idx]

        # Actual quotes within a narrow moneyness band around the slice
        tol = (m_max - m_min) / 20
        actual = otm[np.abs(otm["moneyness"] - moneyness_val) <= tol]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yi, y=term_iv,
            mode="lines", name="Interpolated term structure",
            line=dict(color="royalblue", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=actual["tte"], y=actual["iv"],
            mode="markers", name="Market quotes (nearby)",
            marker=dict(color="red", size=7),
        ))
        fig.update_layout(
            title=f"Term Structure — Moneyness {moneyness_val:.2f}",
            xaxis_title="Time to Expiry (yrs)",
            yaxis_title="Implied Volatility",
            height=460,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
