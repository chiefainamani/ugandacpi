"""
app.py
------
Flask REST API for the Uganda CPI Intelligence System.

Exposes 16 endpoints under the /api/v1/ prefix, serving JSON-formatted
analytical outputs derived from the pipeline and forecasting modules.

All endpoints are read-only (HTTP GET). The pipeline executes once at
startup and results are cached for the lifetime of the server process.

Endpoints
---------
GET /api/v1/                              — API metadata
GET /api/v1/summary                       — Descriptive statistics
GET /api/v1/headline                      — Full headline CPI series
GET /api/v1/categories                    — All 13 category CPI series
GET /api/v1/categories/<name>             — Single category series
GET /api/v1/weights                       — Category weights
GET /api/v1/drivers                       — Weighted contribution ranking
GET /api/v1/growth                        — Cumulative growth from base period
GET /api/v1/burden                        — UGX household cost burden (configurable period)
GET /api/v1/essential-vs-discretionary    — Welch t-test comparison
GET /api/v1/anomalies                     — Anomaly detection results
GET /api/v1/forecast                      — 3-month forecast (selected ML model)
GET /api/v1/forecast/validation           — Walk-forward validation metrics (selected ML model)
GET /api/v1/forecast/models               — ML model-selection evidence (4 models + baselines)
GET /api/v1/forecast/ols                  — OLS benchmark forecast
GET /api/v1/forecast/ols/validation       — OLS benchmark validation

Author : Ainamani Dickson & Lumu Richard
Project: Uganda CPI Intelligence System (BSE Final Year Project, UTAMU)
Version: 1.0.0
"""

import logging
from functools import lru_cache
from flask import Flask, jsonify, request
from flask_cors import CORS

from pipeline.config import (
    API_PREFIX,
    API_HOST,
    API_PORT,
    API_DEBUG,
    API_VERSION,
    CATEGORY_WEIGHTS,
    DATASET_START,
    DATASET_END,
    DATASET_MONTHS,
    FORECAST_HORIZON,
    OLS_TRAINING_WINDOW,
)
from pipeline.ingestion import run_ingestion
from pipeline.analysis import (
    compute_summary,
    compute_weighted_contributions,
    compute_cumulative_growth,
    compute_cost_burden,
    compare_essential_vs_discretionary,
    detect_anomalies,
)
from forecasting.models import generate_forecast, walk_forward_validation
from forecasting.sarima import generate_sarima_forecast,walk_forward_validation_sarima
from forecasting.ml_models import generate_ml_forecast, MODEL_NAMES, MODEL_DESCRIPTIONS
from forecasting.model_selection import load_selection, run_model_selection

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# CORS — allows the HTML dashboard (file:// or different origin) to call the API
CORS(app)


# Pipeline execution at startup — cached for the lifetime of the process

logger.info("Running ingestion pipeline at startup...")
_data = run_ingestion()

_headline    = _data["headline"]
_categories  = _data["categories"]
_monthly_pct = _data["monthly_pct"]
_annual_pct  = _data["annual_pct"]

logger.info("Pipeline complete. API ready.")
# OLS validation is cheap, so it is run once at startup and cached.
_ols_validation = walk_forward_validation(_headline)

# Model-selection evidence produced by forecasting/model_selection.py.
# If it is missing, or was built on older data, it is regenerated (about 2 min).
_selection = load_selection()
if _selection is None or _selection.get("data_end") != _headline.index[-1].strftime("%Y-%m-%d"):
    logger.info("Model-selection results missing or out of date - running selection...")
    _selection = run_model_selection(_headline)
logger.info(f"Selected forecasting model: {_selection['selected_model']}")


@lru_cache(maxsize=1)
def _sarima_validation_cached() -> dict:
    """SARIMA is slow (~2 min), so it is validated on first use only."""
    logger.info("Running SARIMA walk-forward validation (first request only)...")
    return walk_forward_validation_sarima(_headline)


# Helper utilities

def _series_to_dict(series) -> dict:
    """Convert a pandas Series to a JSON-serialisable {ISO-date: value} dict."""
    result = {}
    for date, value in series.items():
        key = date.strftime("%Y-%m-%d")
        result[key] = None if (value != value) else round(float(value), 4)
    return result


def _df_to_dict(df) -> dict:
    """Convert a DataFrame to {column: {ISO-date: value}} dict."""
    return {col: _series_to_dict(df[col]) for col in df.columns}


def _error(message: str, status: int = 400):
    """Return a standardised JSON error response."""
    return jsonify({"error": message}), status


# Endpoints

@app.get(f"{API_PREFIX}/")
def api_root():
    """GET /api/v1/ — API metadata and endpoint listing."""
    return jsonify({
        "api_version":            API_VERSION,
        "system":                 "Uganda CPI Intelligence System",
        "dataset_start":          DATASET_START,
        "dataset_end":            DATASET_END,
        "dataset_months":         DATASET_MONTHS,
        "categories":             len(CATEGORY_WEIGHTS),
        "forecast_horizon_months": FORECAST_HORIZON,
        "endpoints": [
            f"{API_PREFIX}/",
            f"{API_PREFIX}/summary",
            f"{API_PREFIX}/headline",
            f"{API_PREFIX}/categories",
            f"{API_PREFIX}/categories/<name>",
            f"{API_PREFIX}/weights",
            f"{API_PREFIX}/drivers",
            f"{API_PREFIX}/growth",
            f"{API_PREFIX}/burden",
            f"{API_PREFIX}/essential-vs-discretionary",
            f"{API_PREFIX}/anomalies",
            f"{API_PREFIX}/forecast",
            f"{API_PREFIX}/forecast/validation",
            f"{API_PREFIX}/forecast/models",
            f"{API_PREFIX}/forecast/ols",
            f"{API_PREFIX}/forecast/ols/validation",
            f"{API_PREFIX}/forecast/sarima",
            f"{API_PREFIX}/forecast/sarima/validation",
            f"{API_PREFIX}/forecast/comparison",
        ],
    })


@app.get(f"{API_PREFIX}/summary")
def api_summary():
    """GET /api/v1/summary — descriptive statistics for the headline CPI series."""
    try:
        return jsonify(compute_summary(_headline, _annual_pct))
    except Exception as exc:
        logger.exception("Error in /summary")
        return _error(str(exc), 500)


@app.get(f"{API_PREFIX}/headline")
def api_headline():
    """GET /api/v1/headline — full headline CPI series with monthly and annual % changes."""
    try:
        return jsonify({
            "cpi":         _series_to_dict(_headline),
            "monthly_pct": _series_to_dict(_monthly_pct[_headline.name]),
            "annual_pct":  _series_to_dict(_annual_pct[_headline.name]),
        })
    except Exception as exc:
        logger.exception("Error in /headline")
        return _error(str(exc), 500)


@app.get(f"{API_PREFIX}/categories")
def api_categories():
    """GET /api/v1/categories — CPI values for all 13 expenditure categories."""
    try:
        return jsonify({"categories": _df_to_dict(_categories)})
    except Exception as exc:
        logger.exception("Error in /categories")
        return _error(str(exc), 500)


@app.get(f"{API_PREFIX}/categories/<string:name>")
def api_category_single(name: str):
    """
    GET /api/v1/categories/<name> — CPI, annual_pct, and weight for one category.

    URL-encode spaces: /api/v1/categories/Food%20and%20Non-Alcoholic%20Beverages
    """
    if name not in _categories.columns:
        return _error(
            f"Category '{name}' not found. Available: {list(_categories.columns)}", 404
        )
    try:
        return jsonify({
            "category":  name,
            "weight":    CATEGORY_WEIGHTS.get(name),
            "cpi":       _series_to_dict(_categories[name]),
            "annual_pct": _series_to_dict(_annual_pct[name]),
        })
    except Exception as exc:
        logger.exception(f"Error in /categories/{name}")
        return _error(str(exc), 500)


@app.get(f"{API_PREFIX}/weights")
def api_weights():
    """GET /api/v1/weights — national expenditure weights (out of 1 000) for all 13 categories."""
    return jsonify({
        "weights": {cat: round(wgt, 4) for cat, wgt in CATEGORY_WEIGHTS.items()},
        "total":   round(sum(CATEGORY_WEIGHTS.values()), 4),
    })


@app.get(f"{API_PREFIX}/drivers")
def api_drivers():
    """
    GET /api/v1/drivers — weighted contribution of each category to headline inflation.

    Query parameters
    ----------------
    period : str (optional) — ISO date (YYYY-MM-DD). Defaults to latest month.
    """
    period = request.args.get("period", None)
    try:
        contributions_df = compute_weighted_contributions(_annual_pct, period=period)
        return jsonify({
            "period":  period or _annual_pct.index[-1].strftime("%Y-%m-%d"),
            "drivers": contributions_df.to_dict(orient="records"),
        })
    except KeyError as exc:
        return _error(f"Period not found: {exc}", 404)
    except Exception as exc:
        logger.exception("Error in /drivers")
        return _error(str(exc), 500)


@app.get(f"{API_PREFIX}/growth")
def api_growth():
    """
    GET /api/v1/growth — cumulative % growth for headline and all categories.

    Query parameters
    ----------------
    base : str (optional) — ISO date of base month. Defaults to July 2017.
    """
    base_period = request.args.get("base", None)
    try:
        growth_df = compute_cumulative_growth(_headline, _categories, base_period)
        return jsonify({
            "base_period": base_period or _headline.index[0].strftime("%Y-%m-%d"),
            "growth":      _df_to_dict(growth_df),
        })
    except KeyError as exc:
        return _error(f"Base period not found: {exc}", 404)
    except ValueError as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        logger.exception("Error in /growth")
        return _error(str(exc), 500)


@app.get(f"{API_PREFIX}/burden")
def api_burden():
    """
    GET /api/v1/burden — UGX household cost burden for a configurable period.

    Query parameters
    ----------------
    basket : float (required) — monthly household expenditure in UGX.
    base   : str  (optional) — ISO date of reference period. Defaults to July 2017.
    target : str  (optional) — ISO date of comparison period. Defaults to latest month.
    """
    basket_str = request.args.get("basket", None)
    if basket_str is None:
        return _error("Query parameter 'basket' is required (e.g. ?basket=500000).", 400)

    try:
        basket = float(basket_str)
    except ValueError:
        return _error(f"Invalid basket value '{basket_str}'. Must be a number.", 400)

    base_period   = request.args.get("base", None)
    target_period = request.args.get("target", None)

    try:
        result = compute_cost_burden(
            _headline, _categories, basket, base_period, target_period
        )
        return jsonify(result)
    except (ValueError, KeyError) as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        logger.exception("Error in /burden")
        return _error(str(exc), 500)


@app.get(f"{API_PREFIX}/essential-vs-discretionary")
def api_essential_vs_discretionary():
    """
    GET /api/v1/essential-vs-discretionary — Welch t-test comparison of essential
    vs discretionary category inflation rates, with monthly timeseries of group means.
    """
    try:
        return jsonify(compare_essential_vs_discretionary(_annual_pct))
    except Exception as exc:
        logger.exception("Error in /essential-vs-discretionary")
        return _error(str(exc), 500)


@app.get(f"{API_PREFIX}/anomalies")
def api_anomalies():
    """
    GET /api/v1/anomalies — months where headline inflation deviated > z_threshold SDs.

    Query parameters
    ----------------
    threshold : float (optional) — z-score threshold. Defaults to 2.0.
    """
    threshold_str = request.args.get("threshold", None)
    threshold = 2.0

    if threshold_str is not None:
        try:
            threshold = float(threshold_str)
            if threshold <= 0:
                return _error("threshold must be a positive number.", 400)
        except ValueError:
            return _error(f"Invalid threshold '{threshold_str}'. Must be a number.", 400)

    try:
        anomalies = detect_anomalies(_annual_pct, z_threshold=threshold)
        return jsonify({
            "threshold":      threshold,
            "total_anomalies": len(anomalies),
            "anomalies":      anomalies,
        })
    except Exception as exc:
        logger.exception("Error in /anomalies")
        return _error(str(exc), 500)


@app.get(f"{API_PREFIX}/forecast")
def api_forecast():
    """
    GET /api/v1/forecast — CPI forecast from the machine-learning model that
    won the model-selection experiment, with 90% prediction intervals.

    Query parameters
    ----------------
    horizon : int (optional) — months ahead (1-3). Defaults to 3.
              The ML models are trained separately for each of 1, 2 and 3 months.
    """
    try:
        horizon = int(request.args.get("horizon", FORECAST_HORIZON))
    except ValueError as exc:
        return _error(f"Invalid query parameter: {exc}", 400)

    if horizon < 1 or horizon > 3:
        return _error("horizon must be between 1 and 3 for the ML forecast.", 400)

    try:
        result = generate_ml_forecast(
            _headline,
            model_name=_selection["selected_model"],
            params_by_horizon=_selection["params_by_horizon"],
            horizon=horizon,
            error_quantiles=_selection["error_quantiles_90"],
        )
        return jsonify(result)
    except ValueError as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        logger.exception("Error in /forecast")
        return _error(str(exc), 500)


@app.get(f"{API_PREFIX}/forecast/validation")
def api_forecast_validation():
    """
    GET /api/v1/forecast/validation — walk-forward validation metrics of the
    selected ML model: MAPE, RMSE, MAE, and per-horizon MAPE.
    """
    try:
        sel = _selection["selected_model"]
        row = next(r for r in _selection["overall_metrics"] if r["model"] == sel)
        by_h = sorted(
            (r for r in _selection["metrics_by_horizon"] if r["model"] == sel),
            key=lambda r: r["h"],
        )
        return jsonify({
            "model":              sel,
            "n_validation_steps": _selection["evaluation"]["n_origins"],
            "horizon":            FORECAST_HORIZON,
            "mape":               row["MAPE_%"],
            "rmse":               row["RMSE"],
            "mae":                row["MAE"],
            "mape_by_horizon":    [r["MAPE_%"] for r in by_h],
            "protocol":           _selection["evaluation"]["protocol"],
        })
    except Exception as exc:
        logger.exception("Error in /forecast/validation")
        return _error(str(exc), 500)


@app.get(f"{API_PREFIX}/forecast/models")
def api_forecast_models():
    """
    GET /api/v1/forecast/models — the model-selection evidence: metrics of the
    four ML candidates and the baselines, the ranking, and the winner.
    """
    try:
        keys = [
            "selected_model", "selection_rule", "evaluation", "overall_metrics",
            "metrics_by_horizon", "metrics_by_period", "ranking", "baselines",
            "best_baseline", "beats_best_baseline", "calibration_coverage_holdout",
        ]
        payload = {k: _selection[k] for k in keys if k in _selection}
        payload["candidates"] = [
            {"model": m, "description": MODEL_DESCRIPTIONS[m]} for m in MODEL_NAMES
        ]
        return jsonify(payload)
    except Exception as exc:
        logger.exception("Error in /forecast/models")
        return _error(str(exc), 500)


@app.get(f"{API_PREFIX}/forecast/ols")
def api_forecast_ols():
    """
    GET /api/v1/forecast/ols — OLS linear-trend benchmark forecast with
    confidence bands (retained as a transparent benchmark).

    Query parameters
    ----------------
    horizon    : int   (optional) — months ahead (1–12). Defaults to 3.
    window     : int   (optional) — training window in months. Defaults to 24.
    confidence : float (optional) — confidence level (0.50–0.99). Defaults to 0.90.
    """
    try:
        horizon    = int(request.args.get("horizon", FORECAST_HORIZON))
        window     = int(request.args.get("window", OLS_TRAINING_WINDOW))
        confidence = float(request.args.get("confidence", 0.90))
    except ValueError as exc:
        return _error(f"Invalid query parameter: {exc}", 400)

    if horizon < 1 or horizon > 12:
        return _error("horizon must be between 1 and 12.", 400)
    if window < 12:
        return _error("window must be at least 12.", 400)
    if not (0.5 <= confidence <= 0.99):
        return _error("confidence must be between 0.50 and 0.99.", 400)

    try:
        result = generate_forecast(
            _headline, training_window=window,
            horizon=horizon, confidence=confidence
        )
        return jsonify(result)
    except ValueError as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        logger.exception("Error in /forecast/ols")
        return _error(str(exc), 500)


@app.get(f"{API_PREFIX}/forecast/ols/validation")
def api_forecast_ols_validation():
    """
    GET /api/v1/forecast/ols/validation — walk-forward validation metrics
    for the OLS benchmark: MAPE, RMSE, MAE, and per-horizon breakdown.
    """
    try:
        return jsonify({k: v for k, v in _ols_validation.items() if k != "steps"})
    except Exception as exc:
        logger.exception("Error in /forecast/ols/validation")
        return _error(str(exc), 500)


@app.get(f"{API_PREFIX}/forecast/sarima")
def api_forecast_sarima():
    """
    GET /api/v1/forecast/sarima
    Returns a SARIMA 3-month forecast with 90% confidence bands.
    Order is selected by AIC grid search on the training data.

    Query parameters
    ----------------
    horizon    : int   (optional) — months ahead (1–12). Defaults to 3.
    confidence : float (optional) — confidence level (0.50–0.99). Defaults to 0.90.
    """
    try:
        horizon    = int(request.args.get("horizon", FORECAST_HORIZON))
        confidence = float(request.args.get("confidence", 0.90))
    except ValueError as exc:
        return _error(f"Invalid query parameter: {exc}", 400)

    if horizon < 1 or horizon > 12:
        return _error("horizon must be between 1 and 12.", 400)
    if not (0.5 <= confidence <= 0.99):
        return _error("confidence must be between 0.50 and 0.99.", 400)

    try:
        result = generate_sarima_forecast(
            _headline, horizon=horizon, confidence=confidence
        )
        return jsonify(result)
    except ValueError as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        logger.exception("Error in /forecast/sarima")
        return _error(str(exc), 500)


@app.get(f"{API_PREFIX}/forecast/sarima/validation")
def api_forecast_sarima_validation():
    """
    GET /api/v1/forecast/sarima/validation
    Returns cached walk-forward validation metrics for the SARIMA model.
    """
    try:
        return jsonify({k: v for k, v in _sarima_validation_cached().items() if k != "steps"})
    except Exception as exc:
        logger.exception("Error in /forecast/sarima/validation")
        return _error(str(exc), 500)


@app.get(f"{API_PREFIX}/forecast/comparison")
def api_forecast_comparison():
    """
    GET /api/v1/forecast/comparison
    Returns a side-by-side comparison of OLS and SARIMA walk-forward
    validation metrics — MAPE, RMSE, MAE, and per-horizon MAPE breakdown.
    """
    try:
        _sarima_validation = _sarima_validation_cached()
        return jsonify({
            "ols": {
                "model":           "OLS Linear Trend",
                "mape":            _ols_validation["mape"],
                "rmse":            _ols_validation["rmse"],
                "mae":             _ols_validation["mae"],
                "mape_by_horizon": _ols_validation["mape_by_horizon"],
                "n_steps":         _ols_validation["n_validation_steps"],
            },
            "sarima": {
                "model":           _sarima_validation["model"],
                "mape":            _sarima_validation["mape"],
                "rmse":            _sarima_validation["rmse"],
                "mae":             _sarima_validation["mae"],
                "mape_by_horizon": _sarima_validation["mape_by_horizon"],
                "n_steps":         _sarima_validation["n_validation_steps"],
            },
            "winner":   "sarima" if _sarima_validation["mape"] < _ols_validation["mape"] else "ols",
            "verdict":  (
                f"SARIMA achieves MAPE={_sarima_validation['mape']:.4f}% vs "
                f"OLS MAPE={_ols_validation['mape']:.4f}% over {_sarima_validation['n_validation_steps']} "
                f"walk-forward validation steps. "
                f"{'SARIMA' if _sarima_validation['mape'] < _ols_validation['mape'] else 'OLS'} "
                f"is the better forecasting model for Uganda's headline CPI at a 3-month horizon."
            ),
        })
    except Exception as exc:
        logger.exception("Error in /forecast/comparison")
        return _error(str(exc), 500)


# 404 handler

@app.errorhandler(404)
def not_found(error):
    return _error(
        f"Endpoint not found. See {API_PREFIX}/ for available endpoints.", 404
    )


# Entry point

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", API_PORT))
    logger.info(f"Starting Uganda CPI Intelligence API on port {port}")
    app.run(host=API_HOST, port=port, debug=False)
