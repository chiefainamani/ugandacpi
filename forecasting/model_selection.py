"""
model_selection.py
------------------
Selects the best machine-learning model for the Uganda CPI forecast.

What this script does (the story told in the project report):

    1. Loads the headline CPI series produced by the ingestion pipeline.
    2. Evaluates four ML candidates (Lasso, Random Forest, Gradient Boosting,
       SVR) and two reference baselines (Naive-drift, OLS linear trend) using
       the SAME walk-forward (expanding-window) procedure on the SAME
       forecast origins, for horizons of 1, 2 and 3 months.
    3. Tunes hyper-parameters only on data available at each forecast origin
       (no look-ahead).
    4. Scores every model with MAPE, RMSE and MAE, per horizon and per
       economic period, and tests whether differences are statistically
       significant (Diebold-Mariano test).
    5. Ranks the ML candidates with a transparent rule and saves the winner,
       its tuned parameters and its prediction-interval quantiles to
       ``outputs/model_selection/selection.json`` for the API to use.

Run from the project root:

    uv run python -m forecasting.model_selection
    uv run python -m forecasting.model_selection --with-sarima   (slow; needs statsmodels)

Every table and figure written to ``outputs/model_selection/`` can be pasted
directly into Chapter 3 (method) and Chapter 4 (results) of the report.

Author : Ainamani Dickson & Lumu Richard
Project: Uganda CPI Intelligence System (BSE Final Year Project, UTAMU)
Version: 2.0.0
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from forecasting import ml_models
from forecasting.ml_models import MODEL_NAMES, build_features, make_training_set
from forecasting.models import _fit_ols
from pipeline.config import FORECAST_HORIZON, OUTPUTS_DIR

logger = logging.getLogger(__name__)

SELECTION_DIR: Path = OUTPUTS_DIR / "model_selection"
SELECTION_FILE: Path = SELECTION_DIR / "selection.json"

# First forecast origin: models are trained on this many months (Jul 2017 to
# Jun 2021 when 48). ML models need enough history for lags and tuning.
MIN_TRAIN = 48

# Hyper-parameters are re-tuned every TUNE_EVERY origins to keep run time sane.
TUNE_EVERY = 12

BASELINES = ["Naive-drift", "OLS-trend"]

# Economic periods, defined by the forecast ORIGIN date (end of training data).
PERIODS = {
    "Pre-shock (to Dec 2021)": (None, "2021-12-01"),
    "Shock (Jan 2022 - Jun 2023)": ("2022-01-01", "2023-06-01"),
    "Normalisation (Jul 2023 on)": ("2023-07-01", None),
}


# Baseline forecasters (growth in % over h months from the last observation)

def _naive_drift_growth(train: pd.Series, h: int) -> float:
    """Average monthly log growth of the last 12 months, projected h months."""
    logs = np.log(train.values[-13:])
    drift = (logs[-1] - logs[0]) / 12.0
    return float(drift * h * 100.0)


def _ols_growth(train: pd.Series, h: int) -> float:
    """OLS linear trend on all training data (expanding), as in models.py."""
    y = train.values.astype(float)
    slope, intercept, _ = _fit_ols(y)
    t_future = float(len(y) - 1 + h)
    point = intercept + slope * t_future
    return float(np.log(point / y[-1]) * 100.0)


# Walk-forward evaluation

def run_walk_forward(headline: pd.Series, horizon: int = FORECAST_HORIZON,
                     min_train: int = MIN_TRAIN) -> pd.DataFrame:
    """
    Run the walk-forward evaluation for all models and return a long table
    with one row per (origin, horizon, model).

    Columns: origin, origin_date, h, target_date, model, actual, forecast,
             actual_growth, pred_growth.
    """
    headline = headline.dropna().astype(float)
    n = len(headline)
    if n < min_train + horizon:
        raise ValueError(f"Need at least {min_train + horizon} observations.")

    rows = []
    tuned: dict[tuple[str, int], dict] = {}
    t_start = time.time()

    for t in range(min_train, n - horizon + 1):
        train = headline.iloc[:t]
        last = float(train.iloc[-1])
        retune = (t - min_train) % TUNE_EVERY == 0

        pred_growth: dict[tuple[str, int], float] = {}
        for h in range(1, horizon + 1):
            pred_growth[("Naive-drift", h)] = _naive_drift_growth(train, h)
            pred_growth[("OLS-trend", h)] = _ols_growth(train, h)
            X, y = make_training_set(train, h)
            x_now = build_features(train).iloc[[-1]].values
            for name in MODEL_NAMES:
                if retune or (name, h) not in tuned:
                    tuned[(name, h)] = ml_models.tune_hyperparameters(name, X, y)
                model = ml_models._make_model(name, tuned[(name, h)])
                model.fit(X.values, y.values)
                pred_growth[(name, h)] = float(model.predict(x_now)[0])

        for (name, h), g in pred_growth.items():
            actual = float(headline.iloc[t + h - 1])
            rows.append({
                "origin": t,
                "origin_date": headline.index[t - 1],
                "h": h,
                "target_date": headline.index[t + h - 1],
                "model": name,
                "actual": actual,
                "forecast": last * np.exp(g / 100.0),
                "actual_growth": float(np.log(actual / last) * 100.0),
                "pred_growth": g,
            })
        if (t - min_train) % 10 == 0:
            logger.info(f"  origin {t - min_train + 1}/{n - horizon - min_train + 1} "
                        f"({time.time() - t_start:.0f}s elapsed)")

    df = pd.DataFrame(rows)
    df["error"] = df["actual"] - df["forecast"]
    df["abs_pct_error"] = (df["error"].abs() / df["actual"]) * 100.0
    df["growth_error"] = df["actual_growth"] - df["pred_growth"]
    return df


def _baselines_in(df: pd.DataFrame) -> list[str]:
    """Names of non-ML (benchmark) models present in the results table."""
    return [m for m in df["model"].unique() if m not in MODEL_NAMES]


def _append_step_forecasts(df: pd.DataFrame, headline: pd.Series, steps: list,
                           model_name: str, min_train: int) -> pd.DataFrame:
    """
    Add an externally produced walk-forward result (a list of steps, each with
    'actuals' and 'forecasts' lists of CPI levels) to the results table, so a
    benchmark such as SARIMA is scored on exactly the same origins.
    """
    rows = []
    for i, step in enumerate(steps):
        t = min_train + i
        last = float(headline.iloc[t - 1])
        for h, (actual, fc) in enumerate(zip(step["actuals"], step["forecasts"]), start=1):
            rows.append({
                "origin": t,
                "origin_date": headline.index[t - 1],
                "h": h,
                "target_date": headline.index[t + h - 1],
                "model": model_name,
                "actual": float(actual),
                "forecast": float(fc),
                "actual_growth": float(np.log(actual / last) * 100.0),
                "pred_growth": float(np.log(fc / last) * 100.0),
            })
    extra = pd.DataFrame(rows)
    extra["error"] = extra["actual"] - extra["forecast"]
    extra["abs_pct_error"] = extra["error"].abs() / extra["actual"] * 100.0
    extra["growth_error"] = extra["actual_growth"] - extra["pred_growth"]
    return pd.concat([df, extra], ignore_index=True)


def add_sarima_baseline(df: pd.DataFrame, headline: pd.Series,
                        min_train: int = MIN_TRAIN) -> pd.DataFrame:
    """
    Optionally score the project's existing SARIMA model on the same origins.
    Requires statsmodels (slow: it refits SARIMA at every origin).
    """
    from forecasting.sarima import walk_forward_validation_sarima
    res = walk_forward_validation_sarima(headline, min_train=min_train,
                                         horizon=FORECAST_HORIZON)
    return _append_step_forecasts(df, headline, res["steps"], "SARIMA", min_train)


# Metrics

def _metrics(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "MAPE_%": g["abs_pct_error"].mean(),
        "RMSE": float(np.sqrt((g["error"] ** 2).mean())),
        "MAE": g["error"].abs().mean(),
        "Bias": -g["error"].mean(),     # forecast minus actual
        "n": len(g),
    })


def summarise(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Overall, per-horizon and per-period metric tables."""
    overall = df.groupby("model").apply(_metrics, include_groups=False)
    by_h = df.groupby(["model", "h"]).apply(_metrics, include_groups=False)

    parts = []
    for label, (lo, hi) in PERIODS.items():
        mask = pd.Series(True, index=df.index)
        if lo:
            mask &= df["origin_date"] >= pd.Timestamp(lo)
        if hi:
            mask &= df["origin_date"] <= pd.Timestamp(hi)
        sub = df[mask]
        if sub.empty:
            continue
        m = sub.groupby("model").apply(_metrics, include_groups=False)
        m["period"] = label
        parts.append(m.reset_index())
    by_period = pd.concat(parts, ignore_index=True).set_index(["period", "model"])
    return {"overall": overall, "by_horizon": by_h, "by_period": by_period}


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int) -> tuple[float, float]:
    """
    Diebold-Mariano test (squared-error loss) with the Harvey-Leybourne-Newbold
    small-sample correction. Negative statistic: model 1 is more accurate.

    Returns (statistic, two-sided p-value).
    """
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    n = len(d)
    dbar = d.mean()
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for k in range(1, h):
        cov = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2 * cov
    var = max(var / n, 1e-12)
    dm = dbar / np.sqrt(var)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= corr
    p = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)


def dm_table(df: pd.DataFrame, reference: str) -> pd.DataFrame:
    """DM test of every other model against ``reference``, per horizon."""
    rows = []
    for h in range(1, int(df["h"].max()) + 1):
        ref = df[(df["model"] == reference) & (df["h"] == h)].sort_values("origin")
        for name in df["model"].unique():
            if name == reference:
                continue
            oth = df[(df["model"] == name) & (df["h"] == h)].sort_values("origin")
            stat, p = diebold_mariano(ref["error"].values, oth["error"].values, h)
            rows.append({"reference": reference, "model": name, "h": h,
                         "DM_stat": round(stat, 3), "p_value": round(p, 4),
                         "reference_better": stat < 0,
                         "significant_5%": p < 0.05})
    return pd.DataFrame(rows)


# Selection rule

def rank_models(summary: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Rank the ML candidates with a transparent, objective rule.

    Three criteria, each ranked 1 (best) to 4 among ML models:
        1. Overall walk-forward MAPE            (average accuracy)
        2. Overall RMSE                         (penalises large misses)
        3. Worst-period MAPE                    (robustness through the shock)
    The winner has the lowest average rank; ties are broken by MAPE.
    """
    overall = summary["overall"].loc[MODEL_NAMES]
    by_period = summary["by_period"].reset_index()
    by_period = by_period[by_period["model"].isin(MODEL_NAMES)]
    worst = by_period.groupby("model")["MAPE_%"].max().reindex(MODEL_NAMES)

    table = pd.DataFrame({
        "MAPE_%": overall["MAPE_%"],
        "RMSE": overall["RMSE"],
        "Worst_period_MAPE_%": worst,
    })
    for col in table.columns:
        table[f"rank_{col}"] = table[col].rank(method="min")
    rank_cols = [c for c in table.columns if c.startswith("rank_")]
    table["avg_rank"] = table[rank_cols].mean(axis=1)
    table = table.sort_values(["avg_rank", "MAPE_%"])
    table["overall_rank"] = range(1, len(table) + 1)
    return table


def _error_quantiles(df: pd.DataFrame, model: str, level: float = 0.90) -> dict:
    """Empirical growth-error quantiles per horizon for prediction intervals."""
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    out = {}
    for h in sorted(df["h"].unique()):
        e = df[(df["model"] == model) & (df["h"] == h)]["growth_error"].values
        out[str(int(h))] = {"lower": float(np.quantile(e, lo_q)),
                            "upper": float(np.quantile(e, hi_q))}
    return out


def _calibration_check(df: pd.DataFrame, model: str, level: float = 0.90) -> dict:
    """
    Honest interval check: learn quantiles on the first 60% of origins and
    measure coverage on the final 40%.
    """
    res = {}
    for h in sorted(df["h"].unique()):
        sub = df[(df["model"] == model) & (df["h"] == h)].sort_values("origin")
        cut = int(len(sub) * 0.6)
        lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
        lo = np.quantile(sub["growth_error"].values[:cut], lo_q)
        hi = np.quantile(sub["growth_error"].values[:cut], hi_q)
        test = sub["growth_error"].values[cut:]
        res[str(int(h))] = round(float(np.mean((test >= lo) & (test <= hi))), 4)
    return res


# Figures (optional: requires matplotlib)

def _make_figures(df: pd.DataFrame, summary: dict, ranking: pd.DataFrame,
                  best: str, out_dir: Path, baselines: list[str]) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed - skipping figures.")
        return []

    files = []
    order = ranking.index.tolist() + baselines
    colors = {m: c for m, c in zip(order, ["#1b6ca8", "#e07b39", "#3a9d5d", "#8e5bb5",
                                           "#7f7f7f", "#b0b0b0", "#c9a227"])}

    # Figure 1: MAPE by horizon
    by_h = summary["by_horizon"]["MAPE_%"].unstack("h").loc[order]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(order)
    for i, m in enumerate(order):
        ax.bar(np.arange(3) + i * width, by_h.loc[m].values, width,
               label=m, color=colors[m])
    ax.set_xticks(np.arange(3) + width * (len(order) - 1) / 2)
    ax.set_xticklabels(["1 month", "2 months", "3 months"])
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Walk-forward MAPE by forecast horizon")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "fig_mape_by_horizon.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    files.append(p.name)

    # Figure 2: actual vs forecast at the 3-month horizon
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sub = df[(df["h"] == 3)]
    act = sub[sub["model"] == best].sort_values("target_date")
    ax.plot(act["target_date"], act["actual"], color="black", lw=2, label="Actual CPI")
    for m in [best] + baselines:
        s = sub[sub["model"] == m].sort_values("target_date")
        ax.plot(s["target_date"], s["forecast"], lw=1.4, color=colors[m],
                label=f"{m} (3-month ahead)")
    ax.set_ylabel("Headline CPI")
    ax.set_title("Three-month-ahead forecasts vs actual CPI (walk-forward)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "fig_actual_vs_forecast_h3.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    files.append(p.name)

    # Figure 3: MAPE by economic period
    per = summary["by_period"]["MAPE_%"].unstack("model")[order]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / len(order)
    for i, m in enumerate(order):
        ax.bar(np.arange(len(per)) + i * width, per[m].values, width,
               label=m, color=colors[m])
    ax.set_xticks(np.arange(len(per)) + width * (len(order) - 1) / 2)
    ax.set_xticklabels([x.replace(" (", "\n(") for x in per.index], fontsize=8)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Accuracy by economic period (robustness check)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "fig_mape_by_period.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    files.append(p.name)
    return files


# Orchestration

def run_model_selection(headline: pd.Series, out_dir: Path = SELECTION_DIR,
                        include_sarima: bool = False) -> dict:
    """
    Execute the whole selection experiment, save evidence files, and return the
    selection record (also written to selection.json).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    headline = headline.dropna().astype(float)

    logger.info("Running walk-forward evaluation of ML candidates and baselines...")
    df = run_walk_forward(headline)
    if include_sarima:
        logger.info("Adding SARIMA benchmark (slow)...")
        df = add_sarima_baseline(df, headline)
    baselines = _baselines_in(df)
    summary = summarise(df)
    ranking = rank_models(summary)
    best = ranking.index[0]

    # DM tests: winner vs every other model
    dm = dm_table(df, reference=best)

    # Winner's tuned parameters on ALL data, per horizon, and feature importance
    params_by_h = {}
    for h in range(1, FORECAST_HORIZON + 1):
        X, y = make_training_set(headline, h)
        params_by_h[str(h)] = ml_models.tune_hyperparameters(best, X, y)
    importance = {}
    for name in MODEL_NAMES:
        importance[name] = ml_models.feature_importance(
            name, headline, h=1,
            params=params_by_h["1"] if name == best else None)

    quantiles = _error_quantiles(df, best)
    calibration = _calibration_check(df, best)

    # Save evidence tables
    df.to_csv(out_dir / "walkforward_forecasts.csv", index=False)
    summary["overall"].round(4).to_csv(out_dir / "metrics_overall.csv")
    summary["by_horizon"].round(4).to_csv(out_dir / "metrics_by_horizon.csv")
    summary["by_period"].round(4).to_csv(out_dir / "metrics_by_period.csv")
    ranking.round(4).to_csv(out_dir / "ranking_ml_models.csv")
    dm.to_csv(out_dir / "dm_tests.csv", index=False)
    pd.DataFrame(importance).round(4).to_csv(out_dir / "feature_importance.csv")
    figures = _make_figures(df, summary, ranking, best, out_dir, baselines)

    overall = summary["overall"]
    best_baseline = min(baselines, key=lambda m: overall.loc[m, "MAPE_%"])
    record = {
        "selected_model": best,
        "selection_rule": ("Lowest average rank across overall MAPE, overall RMSE "
                           "and worst-period MAPE among the four ML candidates."),
        "evaluation": {
            "protocol": "Walk-forward (expanding window), direct multi-step",
            "first_training_months": MIN_TRAIN,
            "n_origins": int(df["origin"].nunique()),
            "horizons": list(range(1, FORECAST_HORIZON + 1)),
            "tuning": f"TimeSeriesSplit, re-tuned every {TUNE_EVERY} origins",
        },
        "overall_metrics": overall.round(4).reset_index().to_dict("records"),
        "metrics_by_horizon": summary["by_horizon"].round(4).reset_index().to_dict("records"),
        "metrics_by_period": summary["by_period"].round(4).reset_index().to_dict("records"),
        "baselines": baselines,
        "ranking": ranking.round(4).reset_index().to_dict("records"),
        "best_baseline": best_baseline,
        "beats_best_baseline": bool(overall.loc[best, "MAPE_%"]
                                    < overall.loc[best_baseline, "MAPE_%"]),
        "params_by_horizon": params_by_h,
        "error_quantiles_90": quantiles,
        "calibration_coverage_holdout": calibration,
        "feature_importance_selected": importance.get(best, {}),
        "figures": figures,
        "data_end": headline.index[-1].strftime("%Y-%m-%d"),
    }
    with open(SELECTION_FILE, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=str)
    logger.info(f"Selected model: {best}. Evidence saved to {out_dir}")
    return record


def load_selection() -> dict | None:
    """Load the saved selection record, or None if the script was not run."""
    if SELECTION_FILE.exists():
        with open(SELECTION_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%H:%M:%S")
    from pipeline.ingestion import run_ingestion

    import sys
    data = run_ingestion()
    rec = run_model_selection(data["headline"],
                              include_sarima="--with-sarima" in sys.argv)

    print("\n=== OVERALL WALK-FORWARD METRICS ===")
    print(pd.DataFrame(rec["overall_metrics"]).set_index("model").round(3))
    print("\n=== ML MODEL RANKING ===")
    print(pd.DataFrame(rec["ranking"]).set_index("model").round(3))
    print(f"\nSELECTED MODEL: {rec['selected_model']}")
    print(f"Beats best baseline ({rec['best_baseline']}): {rec['beats_best_baseline']}")
