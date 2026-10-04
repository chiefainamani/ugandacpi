# Model Selection Results: Uganda CPI Forecasting

Auto-generated from `outputs/model_selection/` (run `uv run python -m forecasting.model_selection`).
Use this as the source for **Chapter 3 (3.2, 3.5, 3.6, 3.7)** and **Chapter 4 (4.3.2, 4.3.4, 4.4)** of the report.

## 1. What was compared

Four machine-learning candidates and two benchmarks, all scored by the same procedure.

| Model | Type | Idea in one line |
|---|---|---|
| Lasso | ML | Regularised linear regression; shrinks weak features to zero |
| Random Forest | ML | Average of many decision trees trained on random subsets |
| Gradient Boosting | ML | Shallow trees added one at a time to fix earlier errors |
| SVR (RBF) | ML | Support vector regression with a smooth kernel |
| Naive-drift | Benchmark | Last 12 months' average growth projected forward |
| OLS-trend | Benchmark | The proposal's linear trend, refit on all data at each origin |

## 2. Method (for Chapter 3)

- **Data:** headline CPI, July 2017 to June 2026, 108 monthly values (UBOS).
- **Target:** cumulative % growth of CPI over the next 1, 2 and 3 months (one model per horizon, the *direct* strategy). The forecast is converted back to a CPI level. Predicting growth instead of the level matters because tree models cannot predict outside the range they were trained on.
- **Features (12):** monthly % change lags 1, 2, 3, 6, 12; rolling mean of monthly change over 3, 6, 12 months; rolling std (6); annual inflation rate; month encoded as sine and cosine. All use only information known at the forecast date.
- **Validation:** walk-forward, expanding window. The first model is trained on 48 months (Jul 2017 to Jun 2021) and forecasts the next 3; the window then grows one month at a time. This gives **58 forecast origins x 3 horizons = 174 forecasts per model**, including the 2022 inflation shock (annual CPI inflation peaked at 10.7% in Oct 2022).
- **Tuning without leakage:** hyper-parameters are chosen by forward-only cross-validation (`TimeSeriesSplit`, 3 splits) on the training data of each origin only, re-tuned every 12 origins.
- **Metrics:** MAPE (primary, same as the proposal's 2% target), RMSE, MAE, bias; reported overall, by horizon, and by economic period.
- **Significance:** Diebold-Mariano test with Harvey-Leybourne-Newbold correction.
- **Selection rule (fixed before looking at results):** rank the four ML models on (1) overall MAPE, (2) overall RMSE, (3) worst-period MAPE. Lowest average rank wins; ties are broken by overall MAPE.

## 3. Results (for Chapter 4)

### 3.1 Overall accuracy (lower is better)

| Model | MAPE_% | RMSE | MAE | Bias |
|---|---|---|---|---|
| SVR | 0.534 | 1.021 | 0.670 | -0.175 |
| GradientBoosting | 0.553 | 1.033 | 0.700 | -0.194 |
| RandomForest | 0.574 | 1.070 | 0.725 | -0.304 |
| Lasso | 0.586 | 1.141 | 0.738 | -0.135 |
| Naive-drift | 0.599 | 1.085 | 0.759 | -0.026 |
| OLS-trend | 2.048 | 3.315 | 2.618 | -2.603 |

### 3.2 MAPE by forecast horizon (%)

| Model | h=1 | h=2 | h=3 |
|---|---|---|---|
| SVR | 0.288 | 0.555 | 0.758 |
| GradientBoosting | 0.291 | 0.572 | 0.796 |
| RandomForest | 0.324 | 0.568 | 0.829 |
| Lasso | 0.327 | 0.569 | 0.862 |
| Naive-drift | 0.341 | 0.607 | 0.850 |
| OLS-trend | 1.925 | 2.047 | 2.174 |

### 3.3 MAPE by economic period (%)

Periods are defined by the forecast origin: pre-shock (to Dec 2021), shock (Jan 2022 to Jun 2023), normalisation (Jul 2023 on).

| Model | Normalisation (Jul 2023 on) | Pre-shock (to Dec 2021) | Shock (Jan 2022 - Jun 2023) |
|---|---|---|---|
| SVR | 0.249 | 0.365 | 1.121 |
| GradientBoosting | 0.337 | 0.333 | 1.035 |
| RandomForest | 0.340 | 0.333 | 1.096 |
| Lasso | 0.300 | 0.347 | 1.203 |
| Naive-drift | 0.322 | 0.281 | 1.231 |
| OLS-trend | 1.388 | 0.266 | 3.952 |

### 3.4 Ranking of the four ML models

| Model | MAPE % | RMSE | Worst-period MAPE % | Rank MAPE | Rank RMSE | Rank worst-period | Avg rank | Overall |
|---|---|---|---|---|---|---|---|---|
| SVR | 0.534 | 1.021 | 1.121 | 1.000 | 1.000 | 3.000 | 1.667 | 1.000 |
| GradientBoosting | 0.553 | 1.033 | 1.035 | 2.000 | 2.000 | 1.000 | 1.667 | 2.000 |
| RandomForest | 0.574 | 1.070 | 1.096 | 3.000 | 3.000 | 2.000 | 2.667 | 3.000 |
| Lasso | 0.586 | 1.141 | 1.203 | 4.000 | 4.000 | 4.000 | 4.000 | 4.000 |

SVR and Gradient Boosting tie on average rank (1.667). The tie-break (lower overall MAPE) selects **SVR**.

### 3.5 Is the winner significantly better? (Diebold-Mariano p-values)

| SVR vs | p (h=1) | p (h=2) | p (h=3) |
|---|---|---|---|
| GradientBoosting | 0.8808 | 0.5370 | 0.9771 |
| Lasso | 0.1418 | 0.4148 | 0.1786 |
| Naive-drift | 0.0129 | 0.3062 | 0.7857 |
| OLS-trend | 0.0000 | 0.0035 | 0.0227 |
| RandomForest | 0.1029 | 0.6821 | 0.2652 |

At the 5% level SVR is significantly better than OLS-trend at all three horizons, and better than Naive-drift at 1 month only. It is **not** significantly different from Lasso, Random Forest or Gradient Boosting at any horizon. The honest reading: all four ML models are close, SVR is nominally first, and the clear statistical win is over the OLS trend.

### 3.6 Feature importance (share of total; SVR has no native importance)

| Feature | Lasso | RandomForest | GradientBoosting |
|---|---|---|---|
| lag1 | 0.000 | 0.515 | 0.503 |
| lag2 | 0.000 | 0.054 | 0.031 |
| lag3 | 0.000 | 0.099 | 0.142 |
| lag6 | 0.000 | 0.009 | 0.002 |
| lag12 | 0.000 | 0.110 | 0.123 |
| roll_mean3 | 0.000 | 0.091 | 0.076 |
| roll_mean6 | 0.000 | 0.018 | 0.023 |
| roll_mean12 | 0.000 | 0.024 | 0.027 |
| roll_std6 | 0.000 | 0.029 | 0.037 |
| annual_pct | 0.000 | 0.023 | 0.005 |
| month_sin | 0.000 | 0.018 | 0.026 |
| month_cos | 0.000 | 0.007 | 0.004 |

Lasso shrank every feature to zero at the 1-month horizon in the final fit, i.e. it reduces to predicting the average growth. The tree models rely most on last month's change (lag1) and, to a lesser degree, lag 12 and lag 3.

### 3.7 Prediction intervals

Intervals are the 5th and 95th percentiles of the selected model's own walk-forward growth errors (percentage points of growth over the horizon): h=1 [-0.59, 0.74], h=2 [-1.26, 1.46], h=3 [-1.34, 2.64]. A hold-out check (quantiles learned on the first 60% of origins, tested on the last 40%) gave coverage of 100%, 100%, 100%. Coverage above the 90% target means the intervals are conservative (wide), because they include errors from the 2022 shock.

## 4. Final tuned parameters (SVR, fitted on all data)

{"1": {"C": 1.0, "epsilon": 0.1}, "2": {"C": 1.0, "epsilon": 0.1}, "3": {"C": 1.0, "epsilon": 0.1}}

## 5. Limitations to state in the report

1. **Small sample.** Only 108 months; the first model trains on about 35 usable rows. Rankings among the top models are within statistical noise.
2. **Naive-drift is a strong benchmark.** ML beats it by a small margin (0.534% vs 0.599% MAPE) and not significantly at 2 and 3 months.
3. **Different protocol from the proposal's OLS figure.** The proposal's OLS MAPE of 1.52% used 82 steps starting at 24 months; here all models start at 48 months (58 origins) so the ML models have enough history. Under this stricter, shock-heavy window OLS scores 2.05%. Never mix the two numbers in one table.
4. **SARIMA was not re-scored on these origins.** The 0.49% SARIMA figure in the README was computed on the original 82-step protocol, so it is not comparable to 0.534%. Run `uv run python -m forecasting.model_selection --with-sarima` on a machine with statsmodels to score SARIMA on identical origins, then report that number.
5. **Grid edges.** SVR chose C=1 (the smallest value tried) and epsilon=0.1 (the largest), so a wider grid may help.
6. **Horizon limit.** The ML forecast supports 1 to 3 months only; longer horizons would need new direct models.
7. **No exogenous drivers.** Features come from the CPI series itself; exchange rate, fuel prices or rainfall are natural future additions.
