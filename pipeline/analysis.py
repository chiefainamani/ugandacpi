"""
analysis.py
-----------
Statistical analysis module for the Uganda CPI Intelligence System.

Responsibilities:
    1. Weighted inflation contribution decomposition (Laspeyres)
    2. Cumulative percentage growth from a configurable base period
    3. UGX household cost burden calculation
    4. Essential vs. discretionary inflation comparison (Welch t-test)
    5. Anomaly detection via z-score thresholding

Author : Ainamani Dickson & Lumu Richard
Project: Uganda CPI Intelligence System (BSE Final Year Project, UTAMU)
Version: 1.0.0
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.config import (
    CATEGORY_WEIGHTS,
    CATEGORIES,
    ESSENTIAL_CATEGORIES,
    DISCRETIONARY_CATEGORIES,
    ANOMALY_Z_THRESHOLD,
    HEADLINE_LABEL,
)

logger = logging.getLogger(__name__)


# Weighted contribution analysis

def compute_weighted_contributions(
    annual_pct: pd.DataFrame,
    period: str | None = None,
) -> pd.DataFrame:
    """
    Compute each category's weighted contribution to headline annual inflation.

    Uses the Laspeyres additive decomposition:
        contribution_i = (annual_rate_i * weight_i) / 1000

    Parameters
    ----------
    annual_pct : pd.DataFrame
        Annual percentage change DataFrame indexed by date.
    period : str | None
        ISO date string to compute contributions for. Defaults to latest.

    Returns
    -------
    pd.DataFrame
        Columns: category, annual_rate, weight, contribution.
        Sorted descending by contribution. Includes a headline summary row.
    """
    if period is None:
        row = annual_pct.iloc[-1]
        period_label = annual_pct.index[-1].strftime("%b %Y")
    else:
        row = annual_pct.loc[period]
        period_label = period

    logger.info(f"Computing weighted contributions for: {period_label}")

    records = []
    for category in CATEGORIES:
        if category not in annual_pct.columns:
            logger.warning(f"Category '{category}' not found in annual_pct — skipping.")
            continue
        annual_rate  = row[category]
        weight       = CATEGORY_WEIGHTS[category]
        contribution = (annual_rate * weight) / 1000.0
        records.append({
            "category":     category,
            "annual_rate":  round(annual_rate, 4),
            "weight":       round(weight, 4),
            "contribution": round(contribution, 4),
        })

    result_df = pd.DataFrame(records)
    result_df = result_df.sort_values("contribution", ascending=False).reset_index(drop=True)

    headline_rate = row[HEADLINE_LABEL] if HEADLINE_LABEL in row.index else np.nan
    headline_row  = pd.DataFrame([{
        "category":     HEADLINE_LABEL,
        "annual_rate":  round(headline_rate, 4),
        "weight":       1000.0,
        "contribution": round(headline_rate, 4),
    }])
    result_df = pd.concat([result_df, headline_row], ignore_index=True)

    logger.info(
        f"Contributions computed. Top driver: {result_df.iloc[0]['category']} "
        f"({result_df.iloc[0]['contribution']:.3f} pp)"
    )
    return result_df


# Cumulative growth analysis

def compute_cumulative_growth(
    headline: pd.Series,
    categories_df: pd.DataFrame,
    base_period: str | None = None,
) -> pd.DataFrame:
    """
    Compute cumulative percentage growth for headline and all categories
    relative to a specified base period.

    Formula: ((CPI_t / CPI_base) - 1) * 100

    Parameters
    ----------
    headline : pd.Series
        Headline CPI series indexed by date.
    categories_df : pd.DataFrame
        Category CPI DataFrame indexed by date.
    base_period : str | None
        ISO date string of the base month. Defaults to July 2017.

    Returns
    -------
    pd.DataFrame
        Cumulative growth (%) indexed by date. First row is 0.0.
    """
    all_df = pd.concat([headline, categories_df], axis=1)

    base_idx = all_df.index[0] if base_period is None else pd.to_datetime(base_period)

    if base_idx not in all_df.index:
        raise KeyError(
            f"Base period '{base_period}' not found in CPI index. "
            f"Available range: {all_df.index[0].date()} to {all_df.index[-1].date()}"
        )

    base_values = all_df.loc[base_idx]

    if (base_values == 0).any():
        zero_cols = base_values[base_values == 0].index.tolist()
        raise ValueError(f"Zero CPI base value for: {zero_cols} — cannot compute growth.")

    growth_df = ((all_df / base_values) - 1) * 100
    growth_df = growth_df.round(4)

    logger.info(
        f"Cumulative growth computed from {base_idx.strftime('%b %Y')}. "
        f"Latest headline growth: {growth_df[HEADLINE_LABEL].iloc[-1]:.2f}%"
    )
    return growth_df


# UGX household cost burden

def compute_cost_burden(
    headline: pd.Series,
    categories_df: pd.DataFrame,
    monthly_basket_ugx: float,
    base_period: str | None = None,
    target_period: str | None = None,
) -> dict:
    """
    Estimate the additional UGX expenditure a household faces due to inflation,
    given a specified monthly basket size and a configurable comparison period.

    Headline burden formula:
        extra_ugx = monthly_basket * (CPI_target / CPI_base - 1)

    Category-level burden allocates the basket proportionally by weight:
        category_basket = monthly_basket * (weight_i / 1000)
        category_extra  = category_basket * (CPI_category_target / CPI_category_base - 1)

    Parameters
    ----------
    headline : pd.Series
        Headline CPI series indexed by date.
    categories_df : pd.DataFrame
        Category CPI DataFrame indexed by date.
    monthly_basket_ugx : float
        The household's total monthly expenditure in UGX (e.g. 500_000).
    base_period : str | None
        ISO date string for the reference (start) period.
        Defaults to the first available month (July 2017).
    target_period : str | None
        ISO date string for the comparison (end) period.
        Defaults to the latest available month (June 2026).

    Returns
    -------
    dict with keys:
        "base_period"        : str
        "target_period"      : str
        "monthly_basket_ugx" : float
        "headline_burden"    : dict — extra_ugx and growth_pct
        "category_burdens"   : list[dict] — per-category breakdown sorted by burden
    """
    if monthly_basket_ugx <= 0:
        raise ValueError(
            f"monthly_basket_ugx must be positive, got {monthly_basket_ugx}."
        )

    all_df = pd.concat([headline, categories_df], axis=1)

    # Resolve base period
    base_idx = all_df.index[0] if base_period is None else pd.to_datetime(base_period)
    if base_idx not in all_df.index:
        raise KeyError(
            f"Base period '{base_period}' not found. "
            f"Available range: {all_df.index[0].date()} to {all_df.index[-1].date()}"
        )

    # Resolve target period
    target_idx = all_df.index[-1] if target_period is None else pd.to_datetime(target_period)
    if target_idx not in all_df.index:
        raise KeyError(
            f"Target period '{target_period}' not found. "
            f"Available range: {all_df.index[0].date()} to {all_df.index[-1].date()}"
        )

    if target_idx <= base_idx:
        raise ValueError(
            f"target_period ({target_idx.date()}) must be after "
            f"base_period ({base_idx.date()})."
        )

    base_values   = all_df.loc[base_idx]
    target_values = all_df.loc[target_idx]

    # Headline burden
    headline_growth    = (target_values[HEADLINE_LABEL] / base_values[HEADLINE_LABEL]) - 1
    headline_extra_ugx = monthly_basket_ugx * headline_growth

    # Category-level burdens
    category_burdens = []
    for category in CATEGORIES:
        if category not in categories_df.columns:
            continue
        weight     = CATEGORY_WEIGHTS[category]
        cat_basket = monthly_basket_ugx * (weight / 1000.0)
        cat_growth = (target_values[category] / base_values[category]) - 1
        cat_extra  = cat_basket * cat_growth
        category_burdens.append({
            "category":      category,
            "weight":        round(weight, 4),
            "allocated_ugx": round(cat_basket, 2),
            "extra_ugx":     round(cat_extra, 2),
            "growth_pct":    round(cat_growth * 100, 4),
        })

    category_burdens.sort(key=lambda x: x["extra_ugx"], reverse=True)

    result = {
        "base_period":        base_idx.strftime("%b %Y"),
        "target_period":      target_idx.strftime("%b %Y"),
        "monthly_basket_ugx": monthly_basket_ugx,
        "headline_burden": {
            "extra_ugx":  round(headline_extra_ugx, 2),
            "growth_pct": round(headline_growth * 100, 4),
        },
        "category_burdens": category_burdens,
    }

    logger.info(
        f"Cost burden: UGX {monthly_basket_ugx:,.0f} basket | "
        f"{base_idx.strftime('%b %Y')} → {target_idx.strftime('%b %Y')} | "
        f"Extra: UGX {headline_extra_ugx:,.2f}"
    )
    return result


# Essential vs. discretionary comparison

def compare_essential_vs_discretionary(
    annual_pct: pd.DataFrame,
) -> dict:
    """
    Compare mean annual inflation rates between essential and discretionary
    expenditure categories using a Welch two-sample t-test.

    The Welch t-test (unequal variances) is used because the two groups have
    different sizes (6 essential, 7 discretionary) and there is no prior
    assumption of equal variance across the groups.

    Parameters
    ----------
    annual_pct : pd.DataFrame
        Annual percentage change DataFrame. NaN rows are excluded automatically.

    Returns
    -------
    dict with keys:
        essential_mean, discretionary_mean, essential_std, discretionary_std,
        t_statistic, p_value, significant, interpretation, timeseries
    """
    valid_df = annual_pct.dropna()

    essential_cols     = [c for c in ESSENTIAL_CATEGORIES if c in valid_df.columns]
    discretionary_cols = [c for c in DISCRETIONARY_CATEGORIES if c in valid_df.columns]

    essential_df     = valid_df[essential_cols]
    discretionary_df = valid_df[discretionary_cols]

    essential_monthly     = essential_df.mean(axis=1).round(4)
    discretionary_monthly = discretionary_df.mean(axis=1).round(4)

    essential_values     = essential_df.values.flatten()
    discretionary_values = discretionary_df.values.flatten()
    essential_values     = essential_values[~np.isnan(essential_values)]
    discretionary_values = discretionary_values[~np.isnan(discretionary_values)]

    t_stat, p_value = stats.ttest_ind(
        essential_values, discretionary_values, equal_var=False,
    )

    essential_mean     = float(np.mean(essential_values))
    discretionary_mean = float(np.mean(discretionary_values))
    significant        = bool(p_value < 0.05)

    if significant:
        higher = "essential" if essential_mean > discretionary_mean else "discretionary"
        interpretation = (
            f"The difference is statistically significant (p={p_value:.4f} < 0.05). "
            f"{higher.capitalize()} categories experienced meaningfully higher "
            f"average inflation over the analysis period."
        )
    else:
        interpretation = (
            f"The difference is not statistically significant (p={p_value:.4f} ≥ 0.05). "
            f"There is insufficient evidence that essential and discretionary categories "
            f"experienced different average inflation rates."
        )

    logger.info(
        f"Welch t-test: essential={essential_mean:.3f}%, "
        f"discretionary={discretionary_mean:.3f}%, "
        f"p={p_value:.4f}, significant={significant}"
    )

    return {
        "essential_mean":      round(essential_mean, 4),
        "discretionary_mean":  round(discretionary_mean, 4),
        "essential_std":       round(float(np.std(essential_values)), 4),
        "discretionary_std":   round(float(np.std(discretionary_values)), 4),
        "t_statistic":         round(float(t_stat), 4),
        "p_value":             round(float(p_value), 6),
        "significant":         significant,
        "interpretation":      interpretation,
        "timeseries": {
            "dates":              [d.strftime("%Y-%m-%d") for d in essential_monthly.index],
            "essential_mean":     essential_monthly.tolist(),
            "discretionary_mean": discretionary_monthly.tolist(),
        },
    }


# Anomaly detection

def detect_anomalies(
    annual_pct: pd.DataFrame,
    z_threshold: float = ANOMALY_Z_THRESHOLD,
) -> list[dict]:
    """
    Identify months where headline annual inflation deviated unusually from
    the long-run mean, using z-score thresholding.

    A month is flagged when: |z_score| = |(rate - mean) / std| > z_threshold

    Parameters
    ----------
    annual_pct : pd.DataFrame
        Annual percentage change DataFrame. Only the headline column is used.
    z_threshold : float
        Standard deviation threshold. Default is 2.0 from config.

    Returns
    -------
    list[dict]
        Each dict: date, month_label, annual_rate, z_score, direction.
        Sorted by absolute z_score descending.
    """
    if HEADLINE_LABEL not in annual_pct.columns:
        logger.warning(f"'{HEADLINE_LABEL}' not found — cannot detect anomalies.")
        return []

    headline_annual = annual_pct[HEADLINE_LABEL].dropna()

    if len(headline_annual) < 3:
        logger.warning("Insufficient data for anomaly detection.")
        return []

    mean_rate = headline_annual.mean()
    std_rate  = headline_annual.std()

    if std_rate == 0:
        logger.warning("Zero standard deviation — no anomalies possible.")
        return []

    z_scores     = (headline_annual - mean_rate) / std_rate
    anomaly_mask = z_scores.abs() > z_threshold

    anomalies = []
    for date, z_score in z_scores[anomaly_mask].items():
        anomalies.append({
            "date":        date.strftime("%Y-%m-%d"),
            "month_label": date.strftime("%b %Y"),
            "annual_rate": round(float(headline_annual[date]), 4),
            "z_score":     round(float(z_score), 4),
            "direction":   "high" if z_score > 0 else "low",
        })

    anomalies.sort(key=lambda x: abs(x["z_score"]), reverse=True)

    logger.info(
        f"Anomaly detection: ±{z_threshold} SD threshold → "
        f"{len(anomalies)} anomalous month(s) found."
    )
    return anomalies


# Descriptive summary

def compute_summary(
    headline: pd.Series,
    annual_pct: pd.DataFrame,
) -> dict:
    """
    Compute high-level descriptive statistics for the headline CPI series.

    Parameters
    ----------
    headline : pd.Series
        Headline CPI series indexed by date.
    annual_pct : pd.DataFrame
        Annual percentage change DataFrame.

    Returns
    -------
    dict with keys:
        first_period, latest_period, total_months, first_cpi, latest_cpi,
        cumulative_growth_pct, mean_annual_rate, max_annual_rate, min_annual_rate
    """
    headline_annual = annual_pct[HEADLINE_LABEL].dropna()

    max_idx = headline_annual.idxmax()
    min_idx = headline_annual.idxmin()

    first_cpi         = float(headline.iloc[0])
    latest_cpi        = float(headline.iloc[-1])
    cumulative_growth = ((latest_cpi / first_cpi) - 1) * 100

    summary = {
        "first_period":          headline.index[0].strftime("%b %Y"),
        "latest_period":         headline.index[-1].strftime("%b %Y"),
        "total_months":          len(headline),
        "first_cpi":             round(first_cpi, 2),
        "latest_cpi":            round(latest_cpi, 2),
        "cumulative_growth_pct": round(cumulative_growth, 2),
        "mean_annual_rate":      round(float(headline_annual.mean()), 4),
        "max_annual_rate": {
            "rate": round(float(headline_annual[max_idx]), 4),
            "date": max_idx.strftime("%b %Y"),
        },
        "min_annual_rate": {
            "rate": round(float(headline_annual[min_idx]), 4),
            "date": min_idx.strftime("%b %Y"),
        },
    }

    logger.info(
        f"Summary: {summary['first_period']} to {summary['latest_period']} | "
        f"CPI: {first_cpi:.2f} → {latest_cpi:.2f} | "
        f"Growth: {cumulative_growth:.2f}%"
    )
    return summary


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

    print("\n=== SUMMARY ===")
    print(json.dumps(compute_summary(data["headline"], data["annual_pct"]), indent=2))

    print("\n=== COST BURDEN (configurable period: Jan 2020 → Jun 2026) ===")
    burden = compute_cost_burden(
        data["headline"], data["categories"], 500_000,
        base_period="2020-01-01", target_period="2026-06-01"
    )
    print(f"  {burden['base_period']} → {burden['target_period']}")
    print(f"  Extra UGX : {burden['headline_burden']['extra_ugx']:,.2f}")
    print(f"  Growth    : {burden['headline_burden']['growth_pct']:.2f}%")
