# Stock Predictor

Time-series forecasting tool that combines Fourier decomposition with Facebook Prophet for stock price prediction and cycle analysis.

## How it works

**Fourier decomposition** — the closing price is detrended (linear fit removed), then an FFT identifies the dominant periodic cycles in the price history. You choose how many harmonics to keep; the app reconstructs the signal from those harmonics and extrapolates them forward as a forecast. A frequency spectrum plot shows which cycles (in trading days) carry the most energy.

**Prophet forecast** — runs separately as a comparison. Prophet handles trend changepoints and yearly/weekly seasonality automatically.

**Backtest comparison** — both models are evaluated on the most recent ~1 year of data using MAE and RMSE so you can see which fits better for a given ticker.

## Features

- Any ticker symbol (free text input, pulled from Yahoo Finance)
- Adjustable forecast horizon (1–4 years)
- Adjustable number of Fourier harmonics (1–50)
- Frequency spectrum with dominant cycle table
- Fourier reconstruction overlay on historical prices
- Fourier extrapolation forecast
- Prophet forecast with component breakdown
- Side-by-side backtest metrics (MAE, RMSE)
- Residuals view (optional)

## Setup

```bash
git clone https://github.com/cycyee/Stock-Predictor-App.git
cd Stock-Predictor-App
pip install -r requirements.txt
streamlit run main.py
```

## Stack

Python, NumPy (FFT), Prophet, Streamlit, Plotly, yfinance
