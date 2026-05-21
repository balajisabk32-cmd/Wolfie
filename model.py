import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# ── Configuration ──────────────────────────────────────────────
SYMBOL = "RELIANCE.NS"
START_DATE = "2022-01-01"
END_DATE = "2026-02-17"

# ── Fetch Data ─────────────────────────────────────────────────
df = yf.download(SYMBOL, start=START_DATE, end=END_DATE, auto_adjust=True)
df.columns = df.columns.get_level_values(0).str.lower()
df = df.reset_index()

# ── Feature Engineering ────────────────────────────────────────
df["return"] = df["close"].pct_change()

# Volatility & Moving Averages
df["volatility"] = df["return"].rolling(5).std()
df["ma_3"]       = df["close"].rolling(3).mean()
df["ma_5"]       = df["close"].rolling(5).mean()
df["ma_10"]      = df["close"].rolling(10).mean()

# RSI
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = -delta.clip(upper=0).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

df["rsi"] = compute_rsi(df["close"])

# MACD
ema12 = df["close"].ewm(span=12).mean()
ema26 = df["close"].ewm(span=26).mean()
df["macd"]        = ema12 - ema26
df["macd_signal"] = df["macd"].ewm(span=9).mean()

# Bollinger Bands
bb_mid          = df["close"].rolling(20).mean()
bb_std          = df["close"].rolling(20).std()
df["bb_upper"]  = bb_mid + 2 * bb_std
df["bb_lower"]  = bb_mid - 2 * bb_std
df["bb_width"]  = df["bb_upper"] - df["bb_lower"]

# Momentum
df["momentum_5"]  = df["close"] / df["close"].shift(5)  - 1
df["momentum_10"] = df["close"] / df["close"].shift(10) - 1

# Target: next-day return
df["target_return"] = df["return"].shift(-1)
df = df.dropna()

features = [
    "close", "volume", "volatility",
    "ma_3", "ma_5", "ma_10",
    "rsi", "macd", "macd_signal",
    "bb_width", "momentum_5", "momentum_10"
]

X = df[features]
y = df["target_return"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ── Walk-Forward Validation ────────────────────────────────────
print("Running walk-forward validation...")
tscv   = TimeSeriesSplit(n_splits=5)
scores = []

for train_idx, test_idx in tscv.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    _m = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbosity=0)
    _m.fit(X_tr, y_tr)
    scores.append(mean_absolute_error(y_te, _m.predict(X_te)))

print(f"Walk-Forward Avg MAE: {np.mean(scores):.5f}")

# ── Hyperparameter Tuning ──────────────────────────────────────
print("\nTuning hyperparameters...")
param_grid = {
    "n_estimators":     [200, 300, 500],
    "max_depth":        [3, 5, 7],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}

search = RandomizedSearchCV(
    XGBRegressor(random_state=42, verbosity=0),
    param_grid,
    n_iter=20,
    cv=TimeSeriesSplit(n_splits=5),
    scoring="neg_mean_absolute_error",
    random_state=42,
    n_jobs=-1
)
search.fit(X_train, y_train)
print("Best params:", search.best_params_)

# ── Final Model (XGBoost + Scaling) ───────────────────────────
model = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  XGBRegressor(**search.best_params_, random_state=42, verbosity=0))
])
model.fit(X_train, y_train)

# ── Predictions ────────────────────────────────────────────────
pred_returns = model.predict(X_test)

# Confidence interval via Random Forest uncertainty estimate
rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
all_tree_preds = np.stack([t.predict(X_test) for t in rf.estimators_], axis=0)
pred_std    = all_tree_preds.std(axis=0)
lower_bound = pred_returns - 2 * pred_std
upper_bound = pred_returns + 2 * pred_std

pred_prices   = X_test["close"].values * (1 + pred_returns)
lower_prices  = X_test["close"].values * (1 + lower_bound)
upper_prices  = X_test["close"].values * (1 + upper_bound)
y_true_prices = X_test["close"].values * (1 + y_test.values)

# ── Metrics ────────────────────────────────────────────────────
mae  = mean_absolute_error(y_true_prices, pred_prices)
rmse = np.sqrt(mean_squared_error(y_true_prices, pred_prices))
r2   = r2_score(y_true_prices, pred_prices)

print(f"\nMAE:  ${mae:.2f}")
print(f"RMSE: ${rmse:.2f}")
print(f"R²:   {r2:.3f}")

print("\nSample Predictions:")
for i in range(5):
    print(f"  Predicted: ${pred_prices[i]:.2f} | "
          f"95% CI: ${lower_prices[i]:.2f} – ${upper_prices[i]:.2f} | "
          f"Actual: ${y_true_prices[i]:.2f}")

# ── Plot ───────────────────────────────────────────────────────
plt.style.use("ggplot")
plt.figure(figsize=(14, 6))

plt.plot(y_true_prices,  label="Actual Price",    color="#1f77b4", linewidth=2)
plt.plot(pred_prices,    label="Predicted Price", color="#ff7f0e", linewidth=2)
plt.fill_between(
    range(len(pred_prices)),
    lower_prices, upper_prices,
    color="orange", alpha=0.3, label="95% Confidence Interval"
)

plt.title("GOOG Price Prediction (XGBoost + RF Confidence)", fontsize=16)
plt.xlabel("Days",        fontsize=12)
plt.ylabel("Price (USD)", fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

# ── Next Day Prediction ────────────────────────────────────────
latest_idx      = df.index[-1]
latest_features = pd.DataFrame([{
    "close":       df.loc[latest_idx, "close"],
    "volume":      df.loc[latest_idx, "volume"],
    "volatility":  df["return"].iloc[-5:].std(),
    "ma_3":        df["close"].iloc[-3:].mean(),
    "ma_5":        df["close"].iloc[-5:].mean(),
    "ma_10":       df["close"].iloc[-10:].mean(),
    "rsi":         df.loc[latest_idx, "rsi"],
    "macd":        df.loc[latest_idx, "macd"],
    "macd_signal": df.loc[latest_idx, "macd_signal"],
    "bb_width":    df.loc[latest_idx, "bb_width"],
    "momentum_5":  df.loc[latest_idx, "momentum_5"],
    "momentum_10": df.loc[latest_idx, "momentum_10"],
}])

# Clip to training range
for col in latest_features.columns:
    latest_features[col] = latest_features[col].clip(
        lower=X_train[col].min(),
        upper=X_train[col].max()
    )

pred_return_next = model.predict(latest_features)[0]
next_day_price   = df.loc[latest_idx, "close"] * (1 + pred_return_next)

print(f"\nNext Day Predicted Close for {SYMBOL}: ${next_day_price:.2f}")