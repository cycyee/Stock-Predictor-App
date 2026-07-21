import streamlit as st
import numpy as np
import pandas as pd
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.subplots as sp

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("Stock Prediction & Fourier Analysis")

# ── Sidebar controls ──

st.sidebar.header("Parameters")
selected_stock = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper().strip()
n_years = st.sidebar.slider("Forecast horizon (years)", 1, 4, value=2)
period = n_years * 365
n_harmonics = st.sidebar.slider("Fourier harmonics to keep", 1, 50, value=10)
show_residuals = st.sidebar.checkbox("Show residuals", value=False)


@st.cache_data(ttl=3600)
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text(f"Loading {selected_stock}...")
try:
    data = load_data(selected_stock)
    if data.empty:
        st.error(f"No data found for ticker '{selected_stock}'.")
        st.stop()
    data_load_state.text(f"Loaded {len(data)} trading days for {selected_stock}.")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Flatten MultiIndex columns from yfinance if present
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] if col[1] == "" else col[0] for col in data.columns]

# ── Raw price chart ──

st.subheader("Historical Price")
fig_raw = go.Figure()
fig_raw.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="Open", line=dict(width=1)))
fig_raw.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close", line=dict(width=1)))
fig_raw.update_layout(xaxis_rangeslider_visible=True, height=400, margin=dict(t=30, b=30))
st.plotly_chart(fig_raw, use_container_width=True)

# ── Fourier Decomposition ──

st.subheader("Fourier Decomposition")

close_prices = data["Close"].values.flatten().astype(float)
n = len(close_prices)

# Detrend: subtract linear fit so FFT captures oscillations, not drift
x_axis = np.arange(n)
slope, intercept = np.polyfit(x_axis, close_prices, 1)
trend = slope * x_axis + intercept
detrended = close_prices - trend

# FFT on detrended signal
fft_vals = np.fft.rfft(detrended)
freqs = np.fft.rfftfreq(n, d=1.0)  # cycles per trading day
periods_days = np.where(freqs > 0, 1.0 / freqs, np.inf)
magnitudes = np.abs(fft_vals) / n

# Top harmonics by magnitude (skip DC component at index 0)
harmonic_indices = np.argsort(magnitudes[1:])[::-1] + 1
top_indices = harmonic_indices[:n_harmonics]

# Reconstruct signal from top-N harmonics + trend
fft_filtered = np.zeros_like(fft_vals)
fft_filtered[top_indices] = fft_vals[top_indices]
reconstructed = np.fft.irfft(fft_filtered, n=n) + trend

# ── Frequency spectrum plot ──

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Frequency Spectrum**")
    # Only show periods up to half the dataset length, skip DC
    mask = (freqs > 0) & (periods_days <= n / 2)
    fig_spectrum = go.Figure()
    fig_spectrum.add_trace(go.Bar(
        x=periods_days[mask],
        y=magnitudes[mask],
        marker_color="steelblue",
        opacity=0.7,
    ))
    # Highlight the selected harmonics
    top_mask = np.zeros(len(freqs), dtype=bool)
    top_mask[top_indices] = True
    highlight = mask & top_mask
    fig_spectrum.add_trace(go.Bar(
        x=periods_days[highlight],
        y=magnitudes[highlight],
        marker_color="crimson",
        name=f"Top {n_harmonics}",
    ))
    fig_spectrum.update_layout(
        xaxis_title="Period (trading days)",
        yaxis_title="Magnitude",
        xaxis_type="log",
        showlegend=False,
        height=350,
        margin=dict(t=10, b=40),
    )
    st.plotly_chart(fig_spectrum, use_container_width=True)

with col2:
    st.markdown("**Dominant Cycles**")
    top_periods = periods_days[top_indices]
    top_mags = magnitudes[top_indices]
    sort_idx = np.argsort(top_mags)[::-1]
    cycle_df = pd.DataFrame({
        "Period (days)": np.round(top_periods[sort_idx], 1),
        "Approx. Months": np.round(top_periods[sort_idx] / 21, 1),
        "Magnitude": np.round(top_mags[sort_idx], 2),
    })
    st.dataframe(cycle_df, use_container_width=True, hide_index=True)

# ── Reconstruction overlay ──

st.markdown("**Fourier Reconstruction vs Actual Price**")
fig_recon = go.Figure()
fig_recon.add_trace(go.Scatter(
    x=data["Date"], y=close_prices,
    name="Actual", line=dict(width=1, color="gray"), opacity=0.6,
))
fig_recon.add_trace(go.Scatter(
    x=data["Date"], y=reconstructed,
    name=f"Top {n_harmonics} harmonics", line=dict(width=2, color="crimson"),
))
fig_recon.add_trace(go.Scatter(
    x=data["Date"], y=trend,
    name="Linear trend", line=dict(width=1, color="steelblue", dash="dash"),
))
fig_recon.update_layout(height=400, margin=dict(t=30, b=30))
st.plotly_chart(fig_recon, use_container_width=True)

if show_residuals:
    residuals = close_prices - reconstructed
    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(
        x=data["Date"], y=residuals,
        name="Residual", line=dict(width=1, color="orange"),
    ))
    fig_resid.update_layout(
        title="Residuals (Actual - Reconstruction)",
        height=250, margin=dict(t=40, b=30),
    )
    st.plotly_chart(fig_resid, use_container_width=True)

# ── Fourier extrapolation ──

st.subheader("Fourier Forecast")

# Extrapolate each harmonic forward
forecast_days = period
future_x = np.arange(n, n + forecast_days)
future_trend = slope * future_x + intercept

future_signal = np.zeros(forecast_days)
for idx in top_indices:
    amp = np.abs(fft_vals[idx]) / n
    phase = np.angle(fft_vals[idx])
    freq = freqs[idx]
    future_signal += amp * np.cos(2 * np.pi * freq * future_x + phase)
# Mirror conjugates for real signal reconstruction
future_signal *= 2
future_fourier = future_signal + future_trend

future_dates = pd.bdate_range(data["Date"].iloc[-1], periods=forecast_days + 1)[1:]
# Trim to match if bdate_range returns fewer (holidays)
min_len = min(len(future_dates), len(future_fourier))
future_dates = future_dates[:min_len]
future_fourier = future_fourier[:min_len]

fig_fourier_forecast = go.Figure()
fig_fourier_forecast.add_trace(go.Scatter(
    x=data["Date"], y=close_prices,
    name="Historical", line=dict(width=1, color="gray"), opacity=0.6,
))
fig_fourier_forecast.add_trace(go.Scatter(
    x=data["Date"], y=reconstructed,
    name="Fourier fit", line=dict(width=1.5, color="crimson"), opacity=0.5,
))
fig_fourier_forecast.add_trace(go.Scatter(
    x=future_dates, y=future_fourier,
    name="Fourier forecast", line=dict(width=2, color="crimson"),
))
fig_fourier_forecast.update_layout(height=400, margin=dict(t=30, b=30))
st.plotly_chart(fig_fourier_forecast, use_container_width=True)

# ── Prophet Forecast ──

st.subheader("Prophet Forecast")

df_train = data[["Date", "Close"]].copy()
df_train.columns = ["ds", "y"]
df_train["y"] = df_train["y"].values.flatten()

m = Prophet(daily_seasonality=False)
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

fig_prophet = plot_plotly(m, forecast)
fig_prophet.update_layout(height=400, margin=dict(t=30, b=30))
st.plotly_chart(fig_prophet, use_container_width=True)

st.write("Forecast Components")
fig_components = m.plot_components(forecast)
st.write(fig_components)

# ── Model comparison on historical data ──

st.subheader("Model Comparison (Backtest)")

# Use last 252 trading days (~1 year) as test set
test_size = min(252, n // 4)
train_end = n - test_size

actual_test = close_prices[train_end:]
fourier_test = reconstructed[train_end:]
prophet_test = forecast["yhat"].values[train_end:n]

mae_fourier = np.mean(np.abs(actual_test - fourier_test))
mae_prophet = np.mean(np.abs(actual_test - prophet_test))
rmse_fourier = np.sqrt(np.mean((actual_test - fourier_test) ** 2))
rmse_prophet = np.sqrt(np.mean((actual_test - prophet_test) ** 2))

comp_df = pd.DataFrame({
    "Model": ["Fourier", "Prophet"],
    "MAE": [f"${mae_fourier:.2f}", f"${mae_prophet:.2f}"],
    "RMSE": [f"${rmse_fourier:.2f}", f"${rmse_prophet:.2f}"],
})
st.dataframe(comp_df, use_container_width=True, hide_index=True)
