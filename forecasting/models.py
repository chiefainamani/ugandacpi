"""
Forecasting module for the Uganda CPI Intelligence System.

Responsibilities:
    1. Fit an Ordinary Least Squares (OLS) linear trend model to the most
       recent training window of headline CPI observations.
    2. Generate a 3-month ahead forecast with 90% confidence bands.
    3. Evaluate forecast accuracy using walk-forward (expanding window)
       cross-validation, reporting MAPE, RMSE, and MAE.

Why OLS linear trend?
    - Uganda's headline CPI shows a strong, persistent upward trend over the
      108-month observation window (Jul 2017 – Jun 2026).
    - Gur (2024) and Naghi et al. (2024) confirm that simple statistical
      methods remain competitive with ML models at short horizons (1–3 months)
      and with small training datasets.
    - A 3-month horizon is deliberately chosen to maximise OLS accuracy,
      consistent with the M4 Competition findings (Makridakis et al., 2020).
    - The model is fully transparent and interpretable — critical for a
      civic-facing dashboard where users must understand forecast uncertainty.

Author : Ainamani Dickson & Lumu Richard
Project: Uganda CPI Intelligence System (BSE Final Year Project, UTAMU)
Version: 1.0.0
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.config import (
    OLS_TRAINING_WINDOW,
    FORECAST_HORIZON,
    WALK_FORWARD_MIN_TRAIN,
    HEADLINE_LABEL,
)

logger = logging.getLogger(__name__)


# OLS model fitting

def _fit_ols(y: np.ndarray) -> tuple[float, float, float]:
    """
    Fit an OLS linear trend model to a 1-D array of CPI values.

    Model: CPI_t = intercept + slope * t
    where t = 0, 1, 2, ..., len(y) - 1

    Parameters
    ----------
    y : np.ndarray
        Array of CPI values to fit (no NaNs allowed).

    Returns
    -------
    slope : float
        Monthly trend increment in CPI index points.
    intercept : float
        Estimated CPI at t=0 (start of training window).
    stderr : float
        Standard error of the regression residuals — used for
        constructing confidence intervals.
    """
    t = np.arange(len(y), dtype=float)
    slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
    # Residual standard error (std of actual minus fitted)
    fitted = intercept + slope * t
    residuals = y - fitted
    residual_stderr = float(np.std(residuals, ddof=2))  # ddof=2 for slope+intercept
    return float(slope), float(intercept), residual_stderr


def _predict_ols(
    slope: float,
    intercept: float,
    n_train: int,
    horizon: int,
    residual_stderr: float,
    confidence: float = 0.90,
) -> dict:
    """
    Generate point forecasts and confidence intervals for future periods.

    The confidence interval is computed as:
        forecast ± t_critical * residual_stderr * sqrt(1 + 1/n + (t - t_mean)^2 / SS_t)

    where t_critical is the t-distribution critical value at the given
    confidence level with (n_train - 2) degrees of freedom.

    Parameters
    ----------
    slope : float
        OLS slope from _fit_ols.
    intercept : float
        OLS intercept from _fit_ols.
    n_train : int
        Number of observations used in training.
    horizon : int
        Number of months ahead to forecast.
    residual_stderr : float
        Residual standard error from _fit_ols.
    confidence : float
        Confidence level for the prediction interval (default 0.90 = 90%).

    Returns
    -------
    dict with keys:
        "point"  : list[float] — point forecasts for each horizon step
        "lower"  : list[float] — lower confidence bound
        "upper"  : list[float] — upper confidence bound
    """
    t_train = np.arange(n_train, dtype=float)
    t_mean = t_train.mean()
    ss_t = np.sum((t_train - t_mean) ** 2)

    # t-distribution critical value
    df = max(n_train - 2, 1)
    alpha = 1.0 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, df=df)

    point_forecasts = []
    lower_bounds = []
    upper_bounds = []

    for h in range(1, horizon + 1):
        t_future = float(n_train - 1 + h)
        point = intercept + slope * t_future
        # Prediction interval margin
        margin = t_crit * residual_stderr * np.sqrt(
            1 + 1 / n_train + (t_future - t_mean) ** 2 / ss_t
        )
        point_forecasts.append(round(point, 4))
        lower_bounds.append(round(point - margin, 4))
        upper_bounds.append(round(point + margin, 4))

    return {
        "point": point_forecasts,
        "lower": lower_bounds,
        "upper": upper_bounds,
    }


# Public forecasting API

def generate_forecast(
    headline: pd.Series,
    training_window: int = OLS_TRAINING_WINDOW,
    horizon: int = FORECAST_HORIZON,
    confidence: float = 0.90,
) -> dict:
    """
    Fit an OLS model to the most recent training_window months and generate
    a horizon-step ahead forecast with confidence bands.

    Parameters
    ----------
    headline : pd.Series
        Full headline CPI series indexed by date (no NaNs expected).
    training_window : int
        Number of most recent months to use for fitting. Default is
        OLS_TRAINING_WINDOW (24) from config.
    horizon : int
        Number of months ahead to forecast. Default is FORECAST_HORIZON (3).
    confidence : float
        Confidence level for prediction intervals. Default 0.90 (90%).

    Returns
    -------
    dict with keys:
        "training_window"   : int
        "horizon"           : int
        "confidence"        : float
        "last_actual_date"  : str — ISO date of final training observation
        "last_actual_cpi"   : float
        "forecast_dates"    : list[str] — ISO dates of forecast periods
        "forecast_labels"   : list[str] — human-readable labels (e.g. "Jul 2026")
        "point"             : list[float] — point forecasts
        "lower"             : list[float] — lower confidence bound
        "upper"             : list[float] — upper confidence bound
        "slope"             : float — OLS trend slope (CPI pts / month)
        "intercept"         : float — OLS intercept

    Raises
    ------
    ValueError
        If the headline series has fewer observations than training_window.
    """
    headline = headline.dropna()

    if len(headline) < training_window:
        raise ValueError(
            f"Headline series has {len(headline)} observations; "
            f"training_window requires {training_window}. "
            "Reduce OLS_TRAINING_WINDOW in config.py or provide more data."
        )

    # Use only the most recent training_window observations
    train_series = headline.iloc[-training_window:]
    y_train = train_series.values.astype(float)

    slope, intercept, residual_stderr = _fit_ols(y_train)

    predictions = _predict_ols(
        slope, intercept,
        n_train=training_window,
        horizon=horizon,
        residual_stderr=residual_stderr,
        confidence=confidence,
    )

    # Generate forecast date labels
    last_date = headline.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=horizon,
        freq="MS",
    )

    logger.info(
        f"OLS forecast generated: {horizon} months ahead from {last_date.strftime('%b %Y')}. "
        f"Slope: {slope:.4f} CPI pts/month. "
        f"Point forecasts: {predictions['point']}"
    )

    # Compute forecast annual inflation rates
# annual_rate at horizon h = (forecast_CPI / CPI 12 months prior) - 1
# The "12 months prior" values come from the actual headline series
    annual_pct_point = []
    annual_pct_lower = []
    annual_pct_upper = []

    for h in range(1, horizon + 1):
        prior_idx = len(headline) - 12 + h - 1
        if 0 <= prior_idx < len(headline):
            prior_cpi = float(headline.iloc[prior_idx])
            annual_pct_point.append(round((predictions["point"][h - 1] / prior_cpi - 1) * 100, 4))
            annual_pct_lower.append(round((predictions["lower"][h - 1] / prior_cpi - 1) * 100, 4))
            annual_pct_upper.append(round((predictions["upper"][h - 1] / prior_cpi - 1) * 100, 4))
        else:
            annual_pct_point.append(None)
            annual_pct_lower.append(None)
            annual_pct_upper.append(None)
    
    return {
        "training_window": training_window,
        "horizon": horizon,
        "confidence": confidence,
        "last_actual_date": last_date.strftime("%Y-%m-%d"),
        "last_actual_cpi": round(float(headline.iloc[-1]), 4),
        "forecast_dates": [d.strftime("%Y-%m-%d") for d in forecast_dates],
        "forecast_labels": [d.strftime("%b %Y") for d in forecast_dates],
        "point": predictions["point"],
        "lower": predictions["lower"],
        "upper": predictions["upper"],
        "slope": round(slope, 6),
        "intercept": round(intercept, 6),
        "annual_pct_point": annual_pct_point,
        "annual_pct_lower": annual_pct_lower,
        "annual_pct_upper": annual_pct_upper,
    }


# Walk-forward cross-validation

def walk_forward_validation(
    headline: pd.Series,
    min_train: int = WALK_FORWARD_MIN_TRAIN,
    horizon: int = FORECAST_HORIZON,
) -> dict:
    """
    Evaluate OLS forecast accuracy using walk-forward (expanding window)
    cross-validation.

    At each validation step:
        1. Train on all observations up to step t.
        2. Forecast h steps ahead (h = 1 to horizon).
        3. Compare forecast to actual observed values.
        4. Expand the training window by one month and repeat.

    This is also known as time-series cross-validation or rolling-origin
    evaluation. It avoids data leakage by never training on future data.

    Parameters
    ----------
    headline : pd.Series
        Full headline CPI series indexed by date.
    min_train : int
        Minimum number of months to use before the first validation step.
        Default is WALK_FORWARD_MIN_TRAIN (24) from config.
    horizon : int
        Forecast horizon to evaluate. Default is FORECAST_HORIZON (3).

    Returns
    -------
    dict with keys:
        "n_validation_steps" : int — number of expanding window steps run
        "horizon"            : int — forecast horizon evaluated
        "mape"               : float — Mean Absolute Percentage Error (%)
        "rmse"               : float — Root Mean Squared Error
        "mae"                : float — Mean Absolute Error
        "mape_by_horizon"    : list[float] — MAPE for each step h=1,2,...,horizon
        "steps"              : list[dict] — per-step details for diagnostics

    Raises
    ------
    ValueError
        If insufficient data exists for even one validation step.
    """
    headline = headline.dropna()
    n = len(headline)

    # Need at least min_train + horizon observations
    if n < min_train + horizon:
        raise ValueError(
            f"Need at least {min_train + horizon} observations for walk-forward "
            f"validation (min_train={min_train}, horizon={horizon}). "
            f"Only {n} available."
        )

    all_errors: list[float] = []
    errors_by_horizon: list[list[float]] = [[] for _ in range(horizon)]
    steps_log: list[dict] = []

    # Expanding window: train on [0:t], forecast [t:t+horizon]
    for t in range(min_train, n - horizon + 1):
        y_train = headline.iloc[:t].values.astype(float)
        slope, intercept, residual_stderr = _fit_ols(y_train)

        step_actuals = headline.iloc[t : t + horizon].values.astype(float)
        step_forecasts = []

        for h in range(1, horizon + 1):
            t_future = float(t - 1 + h)
            point = intercept + slope * t_future
            step_forecasts.append(round(point, 4))

        # Compute errors for this step
        for h_idx, (actual, forecast) in enumerate(zip(step_actuals, step_forecasts)):
            abs_pct_error = abs((actual - forecast) / actual) * 100
            abs_error = abs(actual - forecast)
            all_errors.append(abs_pct_error)
            errors_by_horizon[h_idx].append(abs_pct_error)

        steps_log.append({
            "train_end": headline.index[t - 1].strftime("%b %Y"),
            "forecast_start": headline.index[t].strftime("%b %Y"),
            "actuals": [round(float(a), 4) for a in step_actuals],
            "forecasts": step_forecasts,
        })

    mape = float(np.mean(all_errors))
    rmse = float(np.sqrt(np.mean(
        [(a - f) ** 2
         for step in steps_log
         for a, f in zip(step["actuals"], step["forecasts"])]
    )))
    mae = float(np.mean(
        [abs(a - f)
         for step in steps_log
         for a, f in zip(step["actuals"], step["forecasts"])]
    ))
    mape_by_horizon = [
        round(float(np.mean(h_errors)), 4)
        for h_errors in errors_by_horizon
    ]

    logger.info(
        f"Walk-forward validation complete. "
        f"Steps: {len(steps_log)} | "
        f"MAPE: {mape:.4f}% | RMSE: {rmse:.4f} | MAE: {mae:.4f}"
    )

    return {
        "n_validation_steps": len(steps_log),
        "horizon": horizon,
        "mape": round(mape, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape_by_horizon": mape_by_horizon,
        "steps": steps_log,
    }


# Entry point for manual testing

if __name__ == "__main__":
    import json
    import logging
    from pipeline.ingestion import run_ingestion

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    data = run_ingestion()
    headline = data["headline"]

    print("\n=== OLS FORECAST (3 months ahead) ===")
    forecast = generate_forecast(headline)
    print(f"  Last actual : {forecast['last_actual_date']}  CPI={forecast['last_actual_cpi']}")
    print(f"  Slope       : {forecast['slope']} CPI pts/month")
    for i, label in enumerate(forecast["forecast_labels"]):
        print(
            f"  {label}  point={forecast['point'][i]:.2f}  "
            f"[{forecast['lower'][i]:.2f}, {forecast['upper'][i]:.2f}]"
        )

    print("\n=== WALK-FORWARD VALIDATION ===")
    validation = walk_forward_validation(headline)
    print(f"  Validation steps : {validation['n_validation_steps']}")
    print(f"  MAPE             : {validation['mape']:.4f}%")
    print(f"  RMSE             : {validation['rmse']:.4f}")
    print(f"  MAE              : {validation['mae']:.4f}")
    print(f"  MAPE by horizon  : {validation['mape_by_horizon']}")