"""
sarima.py
---------
SARIMA forecasting module for the Uganda CPI Intelligence System.

Implements a Seasonal Autoregressive Integrated Moving Average (SARIMA)
model as a benchmark against the OLS linear trend model. SARIMA is
appropriate for Uganda's headline CPI because:

    1. The series exhibits a clear upward trend requiring differencing (d ≥ 1).
    2. Uganda's food-price-driven CPI shows annual seasonality with a
       12-month cycle (s = 12), motivating seasonal terms (P, D, Q).
    3. SARIMA/ARIMA has been shown to outperform OLS on series with
       autocorrelated residuals and seasonal patterns (Box et al., 2015;
       Hassan et al., 2023).

Model order selection:
    Rather than fixing an arbitrary (p,d,q)(P,D,Q,12) order, this module
    uses a grid search over a constrained candidate space and selects the
    order minimising the Akaike Information Criterion (AIC). The search
    is bounded to keep computation tractable on the 108-observation series.

Walk-forward validation mirrors the OLS validation in models.py exactly,
allowing a direct apples-to-apples MAPE comparison between the two models.

Author : Ainamani Dickson & Lumu Richard
Project: Uganda CPI Intelligence System (BSE Final Year Project, UTAMU)
Version: 1.0.0
"""

import logging
import warnings
import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pipeline.config import (
    FORECAST_HORIZON,
    WALK_FORWARD_MIN_TRAIN,
    HEADLINE_LABEL,
)

logger = logging.getLogger(__name__)

# Suppress convergence warnings during grid search — expected for some
# candidate orders that do not fit the data well.
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")


# SARIMA candidate order space

# Non-seasonal orders: p (AR), d (differencing), q (MA)
_P_RANGE = [0, 1, 2]
_D_RANGE = [1]          # d=1 is appropriate for a trending CPI series
_Q_RANGE = [0, 1, 2]

# Seasonal orders: P (seasonal AR), D (seasonal diff), Q (seasonal MA), s
_SP_RANGE = [0, 1]
_SD_RANGE = [0, 1]
_SQ_RANGE = [0, 1]
_S = 12                 # monthly data with annual seasonality


# Order selection via AIC grid search

def select_sarima_order(
    y: np.ndarray,
    verbose: bool = False,
) -> tuple[tuple, tuple]:
    """
    Select the best SARIMA (p,d,q)(P,D,Q,12) order by minimising AIC
    over a constrained candidate grid.

    The grid is bounded to (p,q ≤ 2) and (P,Q ≤ 1) to keep the search
    tractable on the 108-observation series. All candidates with
    d=1, s=12 are evaluated. Candidates that fail to converge are skipped.

    Parameters
    ----------
    y : np.ndarray
        1-D array of CPI values (training data, no NaNs).
    verbose : bool
        If True, logs the AIC for each candidate order.

    Returns
    -------
    best_order : tuple (p, d, q)
        Non-seasonal SARIMA order.
    best_seasonal_order : tuple (P, D, Q, s)
        Seasonal SARIMA order.
    """
    best_aic   = np.inf
    best_order = (1, 1, 1)
    best_seasonal = (1, 1, 0, _S)

    candidates = list(itertools.product(
        _P_RANGE, _D_RANGE, _Q_RANGE,
        _SP_RANGE, _SD_RANGE, _SQ_RANGE,
    ))

    logger.info(f"SARIMA grid search over {len(candidates)} candidate orders...")

    for p, d, q, sp, sd, sq in candidates:
        # Skip degenerate all-zero models
        if p + q + sp + sq == 0:
            continue
        try:
            model = SARIMAX(
                y,
                order=(p, d, q),
                seasonal_order=(sp, sd, sq, _S),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=100)
            aic = result.aic

            if verbose:
                logger.debug(
                    f"  SARIMA({p},{d},{q})({sp},{sd},{sq},12) AIC={aic:.2f}"
                )

            if aic < best_aic:
                best_aic      = aic
                best_order    = (p, d, q)
                best_seasonal = (sp, sd, sq, _S)

        except Exception:
            # Failed to converge — skip this candidate
            continue

    logger.info(
        f"Best order: SARIMA{best_order}{best_seasonal} AIC={best_aic:.2f}"
    )
    return best_order, best_seasonal


# Model fitting

def _fit_sarima(
    y: np.ndarray,
    order: tuple,
    seasonal_order: tuple,
) -> object:
    """
    Fit a SARIMA model with the given orders.

    Parameters
    ----------
    y : np.ndarray
        Training data array (no NaNs).
    order : tuple
        Non-seasonal order (p, d, q).
    seasonal_order : tuple
        Seasonal order (P, D, Q, s).

    Returns
    -------
    SARIMAXResults
        Fitted statsmodels SARIMAX result object.
    """
    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=200)


# Public forecasting API

def generate_sarima_forecast(
    headline: pd.Series,
    horizon: int = FORECAST_HORIZON,
    confidence: float = 0.90,
    training_window: int | None = None,
) -> dict:
    """
    Fit a SARIMA model to the headline CPI series and generate a
    horizon-step ahead forecast with prediction intervals.

    The best SARIMA order is selected by AIC grid search on the
    training data. The full available series is used for fitting
    (or the last training_window months if specified).

    Parameters
    ----------
    headline : pd.Series
        Full headline CPI series indexed by date (no NaNs expected).
    horizon : int
        Number of months ahead to forecast. Default is FORECAST_HORIZON (3).
    confidence : float
        Confidence level for prediction intervals (default 0.90 = 90%).
    training_window : int | None
        If set, use only the last N months for fitting.
        If None, use the full available series.

    Returns
    -------
    dict with keys:
        "model"             : str — SARIMA order string
        "horizon"           : int
        "confidence"        : float
        "aic"               : float — AIC of the fitted model
        "last_actual_date"  : str — ISO date of last training observation
        "last_actual_cpi"   : float
        "forecast_dates"    : list[str]
        "forecast_labels"   : list[str]
        "point"             : list[float] — point forecasts
        "lower"             : list[float] — lower prediction interval
        "upper"             : list[float] — upper prediction interval

    Raises
    ------
    ValueError
        If insufficient data is available for fitting.
    """
    headline = headline.dropna()
    n = len(headline)

    if training_window is not None:
        if n < training_window:
            raise ValueError(
                f"Series has {n} observations; training_window={training_window} required."
            )
        train_series = headline.iloc[-training_window:]
    else:
        train_series = headline

    y_train = train_series.values.astype(float)

    if len(y_train) < 24:
        raise ValueError(
            f"SARIMA requires at least 24 observations; only {len(y_train)} provided."
        )

    # Select best order by AIC
    order, seasonal_order = select_sarima_order(y_train)

    # Fit the model
    result = _fit_sarima(y_train, order, seasonal_order)

    # Generate forecast with prediction interval
    alpha = 1.0 - confidence
    forecast_result = result.get_forecast(steps=horizon)
    forecast_mean   = forecast_result.predicted_mean
    conf_int        = forecast_result.conf_int(alpha=alpha)

    point  = [round(float(v), 4) for v in forecast_mean]
    if hasattr(conf_int, "iloc"):
        lower = [round(float(v), 4) for v in conf_int.iloc[:, 0]]
        upper = [round(float(v), 4) for v in conf_int.iloc[:, 1]]
    else:
        lower = [round(float(v), 4) for v in conf_int[:, 0]]
    upper = [round(float(v), 4) for v in conf_int[:, 1]]
    # Generate forecast date labels
    last_date = headline.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=horizon,
        freq="MS",
    )

    order_str = f"SARIMA{order}{seasonal_order}"
    logger.info(
        f"{order_str} forecast generated: {horizon} months ahead from "
        f"{last_date.strftime('%b %Y')}. "
        f"AIC={result.aic:.2f}. Point forecasts: {point}"
    )

    # Compute forecast annual inflation rates
    # annual_rate at horizon h = (forecast_CPI / CPI 12 months prior) - 1
    # Compute forecast annual inflation rates for point, lower and upper bounds
# annual_rate = (forecast_CPI / CPI_12_months_prior - 1) * 100
    annual_pct_point = []
    annual_pct_lower = []
    annual_pct_upper = []

    for h in range(1, horizon + 1):
        prior_idx = len(headline) - 12 + h - 1
        if 0 <= prior_idx < len(headline):
            prior_cpi = float(headline.iloc[prior_idx])
            annual_pct_point.append(round((point[h - 1] / prior_cpi - 1) * 100, 4))
            annual_pct_lower.append(round((lower[h - 1] / prior_cpi - 1) * 100, 4))
            annual_pct_upper.append(round((upper[h - 1] / prior_cpi - 1) * 100, 4))
        else:
            annual_pct_point.append(None)
            annual_pct_lower.append(None)
            annual_pct_upper.append(None)

    return {
        "model":            order_str,
        "horizon":          horizon,
        "confidence":       confidence,
        "aic":              round(float(result.aic), 4),
        "last_actual_date": last_date.strftime("%Y-%m-%d"),
        "last_actual_cpi":  round(float(headline.iloc[-1]), 4),
        "forecast_dates":   [d.strftime("%Y-%m-%d") for d in forecast_dates],
        "forecast_labels":  [d.strftime("%b %Y") for d in forecast_dates],
        "point":            point,
        "lower":            lower,
        "upper":            upper,
        "annual_pct_point": annual_pct_point,
        "annual_pct_lower": annual_pct_lower,
        "annual_pct_upper": annual_pct_upper,
    }


# Walk-forward cross-validation

def walk_forward_validation_sarima(
    headline: pd.Series,
    min_train: int = WALK_FORWARD_MIN_TRAIN,
    horizon: int = FORECAST_HORIZON,
) -> dict:
    """
    Evaluate SARIMA forecast accuracy using walk-forward (expanding window)
    cross-validation — identical protocol to the OLS validation in models.py,
    enabling direct MAPE comparison between the two approaches.

    To keep computation tractable, the SARIMA order is selected once on
    the initial training window and then held fixed across all validation
    steps. This is standard practice for comparative walk-forward studies
    (Box et al., 2015).

    Parameters
    ----------
    headline : pd.Series
        Full headline CPI series indexed by date.
    min_train : int
        Minimum months before the first validation step. Default 24.
    horizon : int
        Forecast horizon to evaluate. Default FORECAST_HORIZON (3).

    Returns
    -------
    dict with keys:
        "model"              : str — SARIMA order used
        "n_validation_steps" : int
        "horizon"            : int
        "mape"               : float
        "rmse"               : float
        "mae"                : float
        "mape_by_horizon"    : list[float]
        "steps"              : list[dict] — per-step diagnostics
    """
    headline = headline.dropna()
    n = len(headline)

    if n < min_train + horizon:
        raise ValueError(
            f"Need at least {min_train + horizon} observations for walk-forward "
            f"validation (min_train={min_train}, horizon={horizon}). "
            f"Only {n} available."
        )

    # Select order once on the initial training window
    y_init = headline.iloc[:min_train].values.astype(float)
    logger.info("Selecting SARIMA order on initial training window...")
    order, seasonal_order = select_sarima_order(y_init)
    order_str = f"SARIMA{order}{seasonal_order}"
    logger.info(f"Fixed order for validation: {order_str}")

    all_errors: list[float] = []
    errors_by_horizon: list[list[float]] = [[] for _ in range(horizon)]
    steps_log: list[dict] = []

    for t in range(min_train, n - horizon + 1):
        y_train = headline.iloc[:t].values.astype(float)

        try:
            result   = _fit_sarima(y_train, order, seasonal_order)
            fc       = result.get_forecast(steps=horizon)
            fc_means = [round(float(v), 4) for v in fc.predicted_mean]
        except Exception as exc:
            logger.warning(f"SARIMA fit failed at step t={t}: {exc} — skipping.")
            continue

        actuals = headline.iloc[t : t + horizon].values.astype(float)

        for h_idx, (actual, forecast) in enumerate(zip(actuals, fc_means)):
            abs_pct_error = abs((actual - forecast) / actual) * 100
            all_errors.append(abs_pct_error)
            errors_by_horizon[h_idx].append(abs_pct_error)

        steps_log.append({
            "train_end":     headline.index[t - 1].strftime("%b %Y"),
            "forecast_start": headline.index[t].strftime("%b %Y"),
            "actuals":       [round(float(a), 4) for a in actuals],
            "forecasts":     fc_means,
        })

    if not all_errors:
        raise RuntimeError("Walk-forward validation produced no valid steps.")

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
        round(float(np.mean(h_errors)), 4) if h_errors else 0.0
        for h_errors in errors_by_horizon
    ]

    logger.info(
        f"SARIMA walk-forward validation complete. "
        f"Steps: {len(steps_log)} | "
        f"MAPE: {mape:.4f}% | RMSE: {rmse:.4f} | MAE: {mae:.4f}"
    )

    return {
        "model":              order_str,
        "n_validation_steps": len(steps_log),
        "horizon":            horizon,
        "mape":               round(mape, 4),
        "rmse":               round(rmse, 4),
        "mae":                round(mae, 4),
        "mape_by_horizon":    mape_by_horizon,
        "steps":              steps_log,
    }


# Entry point for manual testing and model comparison

if __name__ == "__main__":
    import json
    import logging
    from pipeline.ingestion import run_ingestion
    from forecasting.models import (
        generate_forecast as ols_forecast,
        walk_forward_validation as ols_validation,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    data     = run_ingestion()
    headline = data["headline"]

    print("\n=== SARIMA FORECAST (3 months ahead) ===")
    sarima_fc = generate_sarima_forecast(headline)
    print(f"  Model          : {sarima_fc['model']}")
    print(f"  AIC            : {sarima_fc['aic']}")
    print(f"  Last actual    : {sarima_fc['last_actual_date']}  CPI={sarima_fc['last_actual_cpi']}")
    for i, lbl in enumerate(sarima_fc["forecast_labels"]):
        print(
            f"  {lbl}  point={sarima_fc['point'][i]:.2f}  "
            f"[{sarima_fc['lower'][i]:.2f}, {sarima_fc['upper'][i]:.2f}]"
        )

    print("\n=== SARIMA WALK-FORWARD VALIDATION ===")
    sarima_val = walk_forward_validation_sarima(headline)
    print(f"  Steps          : {sarima_val['n_validation_steps']}")
    print(f"  MAPE           : {sarima_val['mape']:.4f}%")
    print(f"  RMSE           : {sarima_val['rmse']:.4f}")
    print(f"  MAE            : {sarima_val['mae']:.4f}")
    print(f"  MAPE by horizon: {sarima_val['mape_by_horizon']}")

    print("\n=== MODEL COMPARISON (OLS vs SARIMA) ===")
    ols_val = ols_validation(headline)
    print(f"  {'Model':<10} {'MAPE':>8} {'RMSE':>8} {'MAE':>8}")
    print(f"  {'OLS':<10} {ols_val['mape']:>8.4f} {ols_val['rmse']:>8.4f} {ols_val['mae']:>8.4f}")
    print(f"  {'SARIMA':<10} {sarima_val['mape']:>8.4f} {sarima_val['rmse']:>8.4f} {sarima_val['mae']:>8.4f}")
    winner = "OLS" if ols_val["mape"] < sarima_val["mape"] else "SARIMA"
    print(f"\n  Lower MAPE: {winner}")
