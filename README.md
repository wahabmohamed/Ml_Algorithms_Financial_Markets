# Machine Learning Algorithms on Financial Market Data — EUR/USD and S&P 500

A complete data science workflow applied to two very different markets: a currency pair (**EUR/USD**) and a US equity
index (**S&P 500**). **34 forecasting algorithms** are explained (objective, how they work, original source), tuned
on a validation period and evaluated on the same task: forecasting the next day's closing price.

## Content of the notebook

| Part | Content |
|---|---|
| Data collection | EUR/USD from MetaTrader 5, S&P 500 from Yahoo Finance; both 2005-2026 |
| Data cleaning | 8 quality checks per market |
| Exploratory analysis | price history, statistics, ADF stationarity test, autocorrelation |
| Feature engineering | 29 features in percent terms (lags, moving averages, rolling statistics, candle shape, RSI, calendar) |
| Split | chronological: train ≤ 2019, validation 2020-2022, test 2023-2026 |
| Tuning | every family tuned on the validation period (hyperparameter grids, ARIMA order by AIC, network size, input window, learning rate) |
| Robustness | 5 random seeds for every neural network; Holm correction for testing 33 algorithms at once |
| Evaluation metrics | MAE, RMSE, MAPE, R², relative MAE vs naive, Diebold–Mariano test, Holm-adjusted p-value, spread across seeds (with sources) |
| Results | per market: metrics table, accuracy vs training time, forecasts, yearly stability, explanation of each result |
| Comparison | same algorithm on both markets (rank correlation) |

## The 34 algorithms

| Family | Algorithms |
|---|---|
| Baselines | Naive (persistence), Moving average |
| Statistical | Simple Exponential Smoothing, Holt, ARIMA |
| Linear | Linear Regression, Ridge, Lasso |
| Distance / kernel | k-Nearest Neighbours, Support Vector Regression (RBF) |
| Trees | Decision Tree, Random Forest, Extra Trees |
| Boosting | Gradient Boosting, XGBoost, LightGBM, CatBoost |
| Neural networks | MLP, LSTM, GRU, 1D CNN, Transformer |
| Deep forecasting (Nixtla `neuralforecast`) | N-BEATS, N-HiTS, TiDE, DLinear, DeepAR, PatchTST, TFT, TCN, BiTCN |
| AutoML | H2O AutoML |
| Ensembles | average of the deep learning models; average of all models |

## Main results (test period 2023 → October 2026)

| | EUR/USD | S&P 500 |
|---|---|---|
| Naive forecast MAE | 35.92 pips | 36.72 points |
| Best algorithm | SVR (RBF), relative MAE 0.9987 | TFT, relative MAE 0.9930 (± 0.0021 over 5 seeds) |
| Algorithms with relative MAE below 1 | 8 of 33 | 10 of 33 |
| Significantly better than naive (Holm-corrected) | none | none |
| Significantly worse than naive (Holm-corrected) | TiDE, TCN, N-HiTS, N-BEATS | N-BEATS |
| Below 1 in all 4 test years | Holt, SES | 1D CNN, XGBoost |

The rankings of the algorithms on the two markets have a Spearman rank correlation of 0.61.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook ml_algorithms_financial_data.ipynb
```

* H2O AutoML requires a Java runtime (Java 8 or later).
* Markets are defined in the `MARKETS` dictionary (MT5 symbol, Yahoo ticker, start date, price unit, optional fixed
  `source`). `DATA_SOURCE` sets the default order: `"auto"` (MT5 → Yahoo → CSV), `"mt5"`, `"yahoo"` or `"csv"`.
* Runtime: about 75 minutes on a laptop CPU the first time, mostly for the deep forecasting models (4 configurations
  × 5 seeds × 9 models × 2 markets). Every trained model is cached in `data/cache/`, so later runs take a few minutes.

## Disclaimer

Educational data science project. Nothing in it is investment advice.
