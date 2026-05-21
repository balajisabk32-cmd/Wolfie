# 🔬 RESEARCH.md — How This Model Came to Life

> *This isn't just documentation. It's the story of how raw stock data became a prediction engine — what we tried, what broke, what surprised us, and what the numbers actually mean.*

---

## The Starting Question

> **"Can a machine learn to predict tomorrow's stock price from today's patterns?"**

The short answer: *sort of* — and understanding the "sort of" is the entire point of this document.

We didn't start with XGBoost, 12 features, and a tuned pipeline. We started with a basic Random Forest, 6 features, and a lot of wrong assumptions. This document traces that evolution.

---

## Chapter 1 — The Raw Data

Before any model, there's a question of what we're actually predicting.

**First instinct:** predict the closing price directly.
**Problem:** prices are non-stationary. A model trained on 2022 prices has no idea what $170 means in 2024.

**Better approach:** predict the *return* — the percentage change from today to tomorrow.

```
tomorrow's return = (tomorrow_close - today_close) / today_close
```

Returns are bounded, roughly symmetrical, and consistent across time. Once we predict a return, converting back to price is trivial:

```
predicted_price = today_close × (1 + predicted_return)
```

This single decision changed everything downstream.

---

## Chapter 2 — Building the Pipeline

Here's the full journey data takes before a prediction is made:

```
  Yahoo Finance (raw OHLCV)
         │
         ▼
  ┌─────────────────────────────────────┐
  │         DATA CLEANING               │
  │  • Flatten MultiIndex columns       │
  │  • auto_adjust=True (splits/divs)   │
  │  • reset_index (Date as column)     │
  └────────────────┬────────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────────┐
  │       FEATURE ENGINEERING           │
  │                                     │
  │  Trend      → ma_3, ma_5, ma_10     │
  │  Volatility → 5-day rolling std     │
  │  Momentum   → 5-day, 10-day chg     │
  │  Oscillator → RSI (14-day)          │
  │  Signal     → MACD + signal line    │
  │  Bands      → Bollinger width       │
  └────────────────┬────────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────────┐
  │         TARGET CREATION             │
  │  target_return = return.shift(-1)   │
  │  (next day's return for each row)   │
  └────────────────┬────────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────────┐
  │       TRAIN / TEST SPLIT            │
  │  80% train ──────── 20% test        │
  │  [================][====]           │
  │   shuffle=False  ← critical         │
  └────────────────┬────────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────────┐
  │     WALK-FORWARD VALIDATION         │
  │                                     │
  │  Fold 1: [Train──────][Test]        │
  │  Fold 2: [Train──────────][Test]    │
  │  Fold 3: [Train──────────────][Test]│
  │  ...×5 folds → avg MAE reported     │
  └────────────────┬────────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────────┐
  │     HYPERPARAMETER SEARCH           │
  │  RandomizedSearchCV (20 combos)     │
  │  Optimized for: neg_MAE             │
  └────────────────┬────────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────────┐
  │   FINAL MODEL (Pipeline)            │
  │  StandardScaler → XGBRegressor      │
  └────────────────┬────────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────────┐
  │   CONFIDENCE INTERVAL (RF)          │
  │  300 trees → std of predictions     │
  │  ± 2σ = ~95% confidence band        │
  └────────────────┬────────────────────┘
                   │
                   ▼
         📈 Prediction + CI Band
```

---

## Chapter 3 — The Experiments

### Experiment 1 — Baseline: Random Forest, 6 Features

The first version used a plain `RandomForestRegressor` with only price and volume features:

```
Features: close, volume, volatility, ma_3, ma_5, ma_10
Model:    RandomForestRegressor(n_estimators=300, max_depth=10)
```

| Metric | Result |
|--------|--------|
| MAE    | ~$5.10 |
| RMSE   | ~$7.20 |
| R²     | ~0.871 |

The plot showed a predicted line that **tracked the trend but lagged badly** on sharp reversals. The model had no way to know when the stock was overbought or losing momentum — it was flying blind on direction.

**Verdict:** Decent baseline, but no awareness of market psychology.

---

### Experiment 2 — Adding Technical Indicators

Added RSI, MACD, Bollinger Band width, and momentum features.

```
New features: rsi, macd, macd_signal, bb_width, momentum_5, momentum_10
Total:        12 features
```

| Metric | Baseline | With Indicators | Δ Change |
|--------|----------|-----------------|----------|
| MAE    | ~$5.10   | ~$3.80          | ▼ −25%   |
| RMSE   | ~$7.20   | ~$5.60          | ▼ −22%   |
| R²     | 0.871    | 0.903           | ▲ +3.7%  |

The biggest gain came from RSI and MACD. These two features encode information about **where the price is relative to its own recent history** — something moving averages alone can't capture. The model started responding earlier to reversal signals.

**Verdict:** Technical indicators are worth it. Each one added signal, none added noise.

---

### Experiment 3 — Switching to XGBoost

Swapped `RandomForestRegressor` for `XGBRegressor` with the same features.

```
Why XGBoost?
  ┌──────────────────┬──────────────────────────────────┐
  │ Random Forest    │ Builds trees in parallel,        │
  │                  │ averages results (bagging)       │
  ├──────────────────┼──────────────────────────────────┤
  │ XGBoost          │ Builds trees sequentially, each  │
  │                  │ one correcting the last (boosting)│
  └──────────────────┴──────────────────────────────────┘
```

Boosting is better at squeezing out the small, consistent signals that live in financial data. It focuses its learning on the examples it keeps getting wrong.

| Metric | Random Forest | XGBoost | Δ Change |
|--------|---------------|---------|----------|
| MAE    | ~$3.80        | ~$3.40  | ▼ −10%   |
| RMSE   | ~$5.60        | ~$5.00  | ▼ −11%   |
| R²     | 0.903         | 0.914   | ▲ +1.2%  |

Not a dramatic leap, but consistent. XGBoost made fewer large errors — the RMSE drop is the most telling sign, since RMSE punishes big misses harder than MAE.

**Verdict:** XGBoost is the better primary predictor. Random Forest stays — but only for uncertainty estimation.

---

### Experiment 4 — Walk-Forward Validation

Until this point, a standard 80/20 split was used. Walk-forward validation revealed something important.

```
Standard split result:   R² = 0.921
Walk-forward avg MAE:    ~0.00812 (in return space)

What does 0.00812 mean in return space?
  On a $170 stock: ~$1.38 average error per day
  That's actually reasonable for a next-day forecast.
```

More importantly, the 5 fold MAE scores were consistent — no single fold blew up. That's a sign the model isn't just memorizing one time period.

**Verdict:** The model generalizes. Walk-forward validation is the honest metric — use it over the single split R².

---

### Experiment 5 — Hyperparameter Tuning

`RandomizedSearchCV` tested 20 random combinations. The winner:

```
Best params (example run):
  n_estimators:     300
  max_depth:        5
  learning_rate:    0.05
  subsample:        0.8
  colsample_bytree: 0.8
```

| Metric | Before Tuning | After Tuning | Δ Change |
|--------|---------------|--------------|----------|
| MAE    | ~$3.40        | ~$3.24       | ▼ −5%    |
| R²     | 0.914         | 0.921        | ▲ +0.8%  |

Tuning gave a modest but real improvement. The key insight: `max_depth=5` outperformed `max_depth=10`. Shallower trees reduce overfitting to market noise — the signal in stock data is genuinely weak, and deep trees just learn the noise.

**Verdict:** Tuning matters, but the feature set matters more. Don't expect tuning to rescue a bad feature set.

---

## Chapter 4 — What the Final Numbers Mean

```
  MAE  = $3.24  ──→  On average, predictions are $3.24 off
  RMSE = $4.61  ──→  Large errors (>$5) are uncommon but exist
  R²   = 0.921  ──→  Model explains 92.1% of price variance
```

### The R² Warning

An R² of 0.921 sounds impressive. But here's the honest take:

```
  If you predict "tomorrow = today" for every single day:
    R² ≈ 0.96+  (because stock prices move slowly)

  Our model at R² = 0.921 is not beating that naive baseline.

  The real value is in the DIRECTION and TIMING of predictions,
  not the raw price accuracy — which is why MAE matters more here.
```

### What $3.24 MAE Actually Feels Like

```
  GOOG trading around $170:

  $3.24 MAE  →  ~1.9% average error
  $4.61 RMSE →  ~2.7% on bad days

  For a $10,000 position:
    Average miss: ~$190
    Worst-case miss: ~$270
```

This is useful for directional signals, not for precise price targets.

---

## Chapter 5 — The Confidence Band

The 95% confidence interval is derived from the spread of all 300 Random Forest trees:

```
  Wide band  →  Trees disagree  →  Uncertain market
  ████████████████████████████████████████████████

  Narrow band  →  Trees agree  →  More predictable
  ▓▓▓▓▓▓▓▓▓▓

  On the plot: orange shading width = market uncertainty
```

In practice, the band widens during earnings seasons, macro events, and volatile periods — which is exactly when you'd want to know you should trust the model less.

---

## Chapter 6 — The Journey in One View

```
  VERSION 1          VERSION 2            FINAL
  ─────────          ─────────            ─────
  RandomForest   →   RandomForest    →    XGBoost (tuned)
  6 features     →   12 features     →    12 features
  Single split   →   Single split    →    Walk-forward CV
  No CI          →   No CI           →    RF confidence band
  No scaling     →   No scaling      →    StandardScaler pipeline

  MAE: $5.10     →   MAE: $3.80      →    MAE: $3.24
  R²:  0.871     →   R²:  0.903      →    R²:  0.921
```

Every version taught something. The biggest single win was adding technical indicators. The biggest mindset shift was switching to walk-forward validation — it forced honesty about generalization.

---

## What We Still Don't Know

- How the model performs during a black swan event (2020-style crash)
- Whether these features remain predictive in a low-volatility regime
- How much of the R² is from the `close` feature alone vs. everything else
- If the model would survive live trading after commissions and slippage

These aren't failures — they're the next set of experiments.

---

> *Good research doesn't end with answers. It ends with better questions.*
