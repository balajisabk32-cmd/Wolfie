# Wolfie
A machine learning model that predicts next-day stock prices using XGBoost and technical indicators, with confidence intervals derived from a Random Forest ensemble.

🔍 Overview
This project fetches historical stock data from Yahoo Finance, engineers quantitative features, trains an XGBoost regression model with hyperparameter tuning, and predicts the next day's closing price — along with a 95% confidence interval.

🛠️ Tech Stack
ToolPurposeyfinanceFetch historical stock dataXGBoostPrimary prediction modelscikit-learnPreprocessing, validation, metricspandas / numpyData manipulationmatplotlibVisualization

📁 Project Structure
Wolfie/
│
├── model.py        # Main model script
└── README.md       # Project documentation

⚙️ Installation

Clone or download the project folder.
Install dependencies:

bashpip install yfinance scikit-learn xgboost pandas numpy matplotlib

Run the model:

bashpython model.py

🧠 Features Used
FeatureDescriptioncloseDaily closing pricevolumeTrading volumevolatility5-day rolling standard deviation of returnsma_33-day moving averagema_55-day moving averagema_1010-day moving averagersiRelative Strength Index (14-day)macdMACD line (EMA12 - EMA26)macd_signalMACD signal line (9-day EMA of MACD)bb_widthBollinger Band width (20-day)momentum_55-day price momentummomentum_1010-day price momentum

🔄 How It Works

Data Fetching — Downloads OHLCV data from Yahoo Finance
Feature Engineering — Computes RSI, MACD, Bollinger Bands, moving averages, and momentum
Walk-Forward Validation — Evaluates model on 5 time-series splits for realistic performance
Hyperparameter Tuning — Uses RandomizedSearchCV to find optimal XGBoost parameters
Model Training — Trains a StandardScaler + XGBoost pipeline on 80% of data
Confidence Intervals — Uses a Random Forest ensemble to estimate prediction uncertainty
Next-Day Prediction — Predicts tomorrow's closing price using the latest available data


📊 Output

Console metrics: MAE, RMSE, R² score, walk-forward MAE, sample predictions with confidence intervals
Plot: Actual vs predicted prices with 95% confidence band
Next-day prediction: Printed closing price forecast for the configured symbol


⚙️ Configuration
At the top of model.py, you can change:
pythonSYMBOL     = "GOOG"        # Any valid ticker (e.g. "AAPL", "TSLA", "NVDA")
START_DATE = "2022-01-01"  # Training data start
END_DATE   = "2026-02-17"  # Training data end

📉 Sample Results (GOOG)
Walk-Forward Avg MAE: 0.00812
MAE:  $3.24
RMSE: $4.61
R²:   0.921

Next Day Predicted Close for GOOG: $171.43

Results will vary depending on the date range and market conditions.


⚠️ Disclaimer
This project is for educational purposes only. Stock price predictions are inherently uncertain and should not be used as financial advice or for real trading decisions.
