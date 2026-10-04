"""
ml_models.py
------------
Machine-learning forecasting module for the Uganda CPI Intelligence System.

This module replaces the single OLS linear-trend forecaster with four
candidate machine-learning models. The candidates are compared by
``forecasting/model_selection.py`` and the winner is used by the API.

Candidate models (all from scikit-learn):
    1. Lasso        - regularised linear regression (L1 penalty).
    2. RandomForest - bagged decision trees.
    3. GradientBoosting - sequentially boosted shallow trees.
    4. SVR          - support vector regression with an RBF kernel.

Design decisions (explained for the project report):

    A. Predict GROWTH, not the CPI level.
       CPI is a trending series. Tree-based models cannot predict values
       outside the range they were trained on, so a model trained on CPI
       levels would "flat-line" below the latest value. We therefore predict
       the cumulative % growth over h months and convert it back to a CPI
       level:  CPI(t+h) = CPI(t) * exp(growth / 100).

    B. Direct multi-step strategy.
       One separate model is trained for each horizon h = 1, 2, 3. This
       avoids feeding a model its own predictions (error accumulation).

    C. Features use only information available at forecast time (no leakage):
       lagged monthly % changes, rolling means/volatility, the current annual
       inflation rate and the calendar month (sine/cosine encoded).

    D. Hyper-parameters are tuned with TimeSeriesSplit (inner, forward-only
       cross-validation) using ONLY the training data available at each
       forecast origin, so the evaluation is never contaminated by the
       future.

Author : Ainamani Dickson & Lumu Richard
Project: Uganda CPI Intelligence System (BSE Final Year Project, UTAMU)
Version: 2.0.0
"""

import logging
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from pipeline.config import FORECAST_HORIZON

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

RANDOM_STATE = 42

# Number of leading observations lost to lags/rolling windows. The first
# usable feature row is at position FIRST_FEATURE_ROW (0-based).
FIRST_FEATURE_ROW = 12

# Minimum number of training rows needed to fit and tune a model.
MIN_TRAIN_ROWS = 12

FEATURE_NAMES = [
    "lag1", "lag2", "lag3", "lag6", "lag12",
    "roll_mean3", "roll_mean6", "roll_mean12", "roll_std6",
    "annual_pct", "month_sin", "month_cos",
]


# Feature engineering

def build_features(headline: pd.Series) -> pd.DataFrame:
    """
    Build the feature matrix for every month in the headline series.

    Row ``t`` contains only information known at the end of month ``t``.
    The first FIRST_FEATURE_ROW rows are NaN (not enough history).

    Parameters
    ----------
    headline : pd.Series
        Headline CPI indexed by month-start dates.

    Returns
    -------
    pd.DataFrame
        Columns are FEATURE_NAMES; same index as ``headline``.
    """
    headline = headline.astype(float)
    m = headline.pct_change() * 100.0           # monthly % change

    feats = pd.DataFrame(index=headline.index)
    for k in (1, 2, 3, 6, 12):
        feats[f"lag{k}"] = m.shift(k - 1)        # lag1 = this month's change
    feats["roll_mean3"] = m.rolling(3).mean()
    feats["roll_mean6"] = m.rolling(6).mean()
    feats["roll_mean12"] = m.rolling(12).mean()
    feats["roll_std6"] = m.rolling(6).std()
    feats["annual_pct"] = headline.pct_change(12) * 100.0
    month = headline.index.month
    feats["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    feats["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    return feats[FEATURE_NAMES]


def build_targets(headline: pd.Series, h: int) -> pd.Series:
    """
    Cumulative % growth of CPI over the next ``h`` months, aligned to the
    forecast origin: target[t] = 100 * ln(CPI[t+h] / CPI[t]).
    """
    logs = np.log(headline.astype(float))
    return (logs.shift(-h) - logs) * 100.0


def make_training_set(headline: pd.Series, h: int):
    """
    Return (X, y) rows that are fully observed using only data in ``headline``.
    Rows whose target would lie beyond the end of the series are dropped.
    """
    feats = build_features(headline)
    target = build_targets(headline, h)
    data = feats.join(target.rename("y")).dropna()
    return data[FEATURE_NAMES], data["y"]


# Candidate models and their (small) tuning grids

def _make_model(name: str, params: dict):
    """Instantiate a scikit-learn estimator for a candidate model name."""
    if name == "Lasso":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", Lasso(max_iter=20000, **params)),
        ])
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=150, random_state=RANDOM_STATE, n_jobs=1, **params
        )
    if name == "GradientBoosting":
        return GradientBoostingRegressor(
            subsample=0.8, random_state=RANDOM_STATE, **params
        )
    if name == "SVR":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", SVR(kernel="rbf", gamma="scale", **params)),
        ])
    raise ValueError(f"Unknown model name: {name}")


PARAM_GRIDS: dict[str, dict[str, list]] = {
    "Lasso": {"alpha": [0.005, 0.02, 0.05, 0.1, 0.3]},
    "RandomForest": {"max_depth": [3, 6], "min_samples_leaf": [2, 5]},
    "GradientBoosting": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2],
    },
    "SVR": {"C": [1.0, 10.0], "epsilon": [0.02, 0.1]},
}

MODEL_NAMES = list(PARAM_GRIDS.keys())

MODEL_DESCRIPTIONS = {
    "Lasso": "Regularised linear regression; shrinks weak features to zero.",
    "RandomForest": "Average of many decision trees trained on random subsets.",
    "GradientBoosting": "Shallow trees added one at a time to fix earlier errors.",
    "SVR": "Support vector regression with a smooth RBF kernel.",
}


def _grid(name: str) -> list[dict]:
    keys = list(PARAM_GRIDS[name].keys())
    return [dict(zip(keys, vals)) for vals in product(*PARAM_GRIDS[name].values())]


def tune_hyperparameters(name: str, X: pd.DataFrame, y: pd.Series,
                         n_splits: int = 3) -> dict:
    """
    Choose hyper-parameters by forward-only cross-validation (TimeSeriesSplit).

    Each candidate is trained on an earlier block and scored on the block
    that follows it, mimicking real forecasting. The candidate with the
    lowest mean absolute error wins.
    """
    grid = _grid(name)
    if len(grid) == 1 or len(X) < MIN_TRAIN_ROWS:
        return grid[0]

    n_splits = max(2, min(n_splits, len(X) // 5))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    best_params, best_score = grid[0], np.inf
    for params in grid:
        errs = []
        for tr, te in splitter.split(X):
            model = _make_model(name, params)
            model.fit(X.iloc[tr].values, y.iloc[tr].values)
            pred = model.predict(X.iloc[te].values)
            errs.append(np.mean(np.abs(pred - y.iloc[te].values)))
        score = float(np.mean(errs))
        if score < best_score:
            best_score, best_params = score, params
    return best_params


def fit_predict_growth(name: str, headline: pd.Series, h: int,
                       params: dict | None = None):
    """
    Fit ``name`` on all data in ``headline`` for horizon ``h`` and predict the
    cumulative growth (%) from the latest observation.

    Returns
    -------
    (growth_pct, params_used)
    """
    X, y = make_training_set(headline, h)
    if len(X) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Only {len(X)} usable training rows for horizon {h}; "
            f"need at least {MIN_TRAIN_ROWS}."
        )
    if params is None:
        params = tune_hyperparameters(name, X, y)
    model = _make_model(name, params)
    model.fit(X.values, y.values)
    x_now = build_features(headline).iloc[[-1]]
    if x_now.isna().any(axis=None):
        raise ValueError("Latest feature row contains NaN; series too short.")
    return float(model.predict(x_now.values)[0]), params


def feature_importance(name: str, headline: pd.Series, h: int = 1,
                       params: dict | None = None) -> dict[str, float]:
    """
    Return normalised feature importance for a fitted model (sums to 1).

    Trees use impurity importance; Lasso uses |standardised coefficient|;
    SVR (no native importance) returns an empty dict.
    """
    X, y = make_training_set(headline, h)
    if params is None:
        params = tune_hyperparameters(name, X, y)
    model = _make_model(name, params)
    model.fit(X.values, y.values)
    if name in ("RandomForest", "GradientBoosting"):
        raw = np.asarray(model.feature_importances_, dtype=float)
    elif name == "Lasso":
        raw = np.abs(model.named_steps["model"].coef_)
    else:
        return {}
    total = raw.sum()
    if total == 0:
        return {f: 0.0 for f in FEATURE_NAMES}
    return {f: round(float(v / total), 4) for f, v in zip(FEATURE_NAMES, raw)}


# Public forecasting API used by the web service

def generate_ml_forecast(
    headline: pd.Series,
    model_name: str,
    params_by_horizon: dict | None = None,
    horizon: int = FORECAST_HORIZON,
    error_quantiles: dict | None = None,
    confidence: float = 0.90,
) -> dict:
    """
    Produce a ``horizon``-month-ahead CPI forecast with the chosen ML model.

    Parameters
    ----------
    headline : pd.Series
        Full headline CPI series (monthly, no NaNs).
    model_name : str
        One of MODEL_NAMES.
    params_by_horizon : dict, optional
        ``{"1": {...}, "2": {...}, "3": {...}}`` tuned hyper-parameters
        saved by the selection script. If omitted they are re-tuned.
    horizon : int
        Months ahead (1 to 3 supported, because models are trained per step).
    error_quantiles : dict, optional
        Empirical growth-error quantiles from walk-forward validation, as
        ``{"1": {"lower": q05, "upper": q95}, ...}`` for the default 90%
        level. They define the prediction interval. If omitted, no interval
        is invented: ``lower`` and ``upper`` are returned as None.
    confidence : float
        Reported confidence level label (the stored quantiles define it).

    Returns
    -------
    dict with the same keys as the OLS/SARIMA forecasts (point, lower, upper,
    forecast_dates, forecast_labels, annual_pct_*), plus ``model``.
    """
    headline = headline.dropna().astype(float)
    if horizon < 1 or horizon > 3:
        raise ValueError("ML forecasting supports horizons of 1 to 3 months.")
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{model_name}'. Use one of {MODEL_NAMES}.")
    if len(headline) < FIRST_FEATURE_ROW + MIN_TRAIN_ROWS + 3:
        raise ValueError(
            f"Series has {len(headline)} observations; at least "
            f"{FIRST_FEATURE_ROW + MIN_TRAIN_ROWS + 3} are needed."
        )

    last_cpi = float(headline.iloc[-1])
    last_date = headline.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1), periods=horizon, freq="MS"
    )

    point, lower, upper = [], [], []
    for h in range(1, horizon + 1):
        params = (params_by_horizon or {}).get(str(h))
        growth, _ = fit_predict_growth(model_name, headline, h, params)
        cpi_point = last_cpi * np.exp(growth / 100.0)
        point.append(round(float(cpi_point), 4))
        q = (error_quantiles or {}).get(str(h))
        if q:
            # error = actual growth - predicted growth, so add to prediction
            lower.append(round(float(last_cpi * np.exp((growth + q["lower"]) / 100.0)), 4))
            upper.append(round(float(last_cpi * np.exp((growth + q["upper"]) / 100.0)), 4))
        else:
            lower.append(None)
            upper.append(None)

    def _annual(values):
        out = []
        for h, v in enumerate(values, start=1):
            prior_idx = len(headline) - 12 + h - 1
            if v is None or not (0 <= prior_idx < len(headline)):
                out.append(None)
            else:
                out.append(round((v / float(headline.iloc[prior_idx]) - 1) * 100, 4))
        return out

    logger.info(
        f"{model_name} forecast: {horizon} months ahead from "
        f"{last_date.strftime('%b %Y')}. Points: {point}"
    )
    return {
        "model": model_name,
        "horizon": horizon,
        "confidence": confidence,
        "last_actual_date": last_date.strftime("%Y-%m-%d"),
        "last_actual_cpi": round(last_cpi, 4),
        "forecast_dates": [d.strftime("%Y-%m-%d") for d in forecast_dates],
        "forecast_labels": [d.strftime("%b %Y") for d in forecast_dates],
        "point": point,
        "lower": lower,
        "upper": upper,
        "annual_pct_point": _annual(point),
        "annual_pct_lower": _annual(lower),
        "annual_pct_upper": _annual(upper),
    }
