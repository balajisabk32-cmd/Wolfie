# 📈 Market Predictor

A machine learning model that predicts next-day stock prices using **XGBoost** and technical indicators, with confidence intervals derived from a Random Forest ensemble.

---

## 🔍 Overview

This project fetches historical stock data from Yahoo Finance, engineers quantitative features, trains an XGBoost regression model with hyperparameter tuning, and predicts the next day's closing price — along with a 95% confidence interval.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `yfinance` | Fetch historical stock data |
| `XGBoost` | Primary prediction model |
| `scikit-learn` | Preprocessing, validation, metrics |
| `pandas / numpy` | Data manipulation |
| `matplotlib` | Visualization |

---

## 📁 Project Structure

```
Market_Predictor/
│
├── model.py        # Main model script
└── README.md       # Project documentation
```

---

## ⚙️ Installation

1. **Clone or download** the project folder.

2. **Install dependencies:**

```bash
pip install yfinance scikit-learn xgboost pandas numpy matplotlib
```

3. **Run the model:**

```bash
python model.py
```

---

## 🧠 Features Used

| Feature | Description |
|---------|-------------|
| `close` | Daily closing price |
| `volume` | Trading volume |
| `volatility` | 5-day rolling standard deviation of returns |
| `ma_3` | 3-day moving average |
| `ma_5` | 5-day moving average |
| `ma_10` | 10-day moving average |
| `rsi` | Relative Strength Index (14-day) |
| `macd` | MACD line (EMA12 - EMA26) |
| `macd_signal` | MACD signal line (9-day EMA of MACD) |
| `bb_width` | Bollinger Band width (20-day) |
| `momentum_5` | 5-day price momentum |
| `momentum_10` | 10-day price momentum |

---

## 🔄 How It Works

1. **Data Fetching** — Downloads OHLCV data from Yahoo Finance
2. **Feature Engineering** — Computes RSI, MACD, Bollinger Bands, moving averages, and momentum
3. **Walk-Forward Validation** — Evaluates model on 5 time-series splits for realistic performance
4. **Hyperparameter Tuning** — Uses `RandomizedSearchCV` to find optimal XGBoost parameters
5. **Model Training** — Trains a `StandardScaler + XGBoost` pipeline on 80% of data
6. **Confidence Intervals** — Uses a Random Forest ensemble to estimate prediction uncertainty
7. **Next-Day Prediction** — Predicts tomorrow's closing price using the latest available data

---

## 📊 Output

- **Console metrics:** MAE, RMSE, R² score, walk-forward MAE, sample predictions with confidence intervals
- **Plot:** Actual vs predicted prices with 95% confidence band
- **Next-day prediction:** Printed closing price forecast for the configured symbol

---

## ⚙️ Configuration

At the top of `model.py`, you can change:

```python
SYMBOL     = "GOOG"        # Any valid ticker (e.g. "AAPL", "TSLA", "NVDA")
START_DATE = "2022-01-01"  # Training data start
END_DATE   = "2026-02-17"  # Training data end
```

---

## 📉 Sample Results (GOOG)

```
Walk-Forward Avg MAE: 0.00812
MAE:  $3.24
RMSE: $4.61
R²:   0.921

Next Day Predicted Close for GOOG: $171.43
```

> Results will vary depending on the date range and market conditions.

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. Stock price predictions are inherently uncertain and should **not** be used as financial advice or for real trading decisions.
