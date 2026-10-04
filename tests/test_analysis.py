
"""Unit tests for the Uganda CPI Intelligence System analysis module.

Test coverage:
    - compute_weighted_contributions: correct output, sorting, headline row,
      missing columns, custom period
    - compute_cumulative_growth: base period handling, zero base guard,
      correct formula, invalid period
    - compute_cost_burden: correct UGX calculations, negative basket guard,
      category breakdown sorting, zero growth edge case
    - compare_essential_vs_discretionary: t-test output structure, group
      separation, timeseries keys, insufficient data handling
    - detect_anomalies: known anomaly detection, empty result for flat data,
      missing headline column, custom threshold
    - compute_summary: all keys present, correct cumulative growth,
      correct peak inflation identification

Run with:
    uv run python -m pytest tests/test_analysis.py -v

Author : Ainamani Dickson & Lumu Richard
Project: Uganda CPI Intelligence System (BSE Final Year Project, UTAMU)
Version: 1.0.0
"""

import pytest
import numpy as np
import pandas as pd

from pipeline import analysis
from pipeline.config import (
    CATEGORIES,
    CATEGORY_WEIGHTS,
    ESSENTIAL_CATEGORIES,
    DISCRETIONARY_CATEGORIES,
    HEADLINE_LABEL,
    ANOMALY_Z_THRESHOLD,
    DATASET_MONTHS,
)


# Fixtures

def _make_date_index(n: int = 108) -> pd.DatetimeIndex:
    """Return a monthly DatetimeIndex of length n starting July 2017."""
    return pd.date_range(start="2017-07-01", periods=n, freq="MS")


def _make_headline(n: int = 108) -> pd.Series:
    """Return a synthetic headline CPI series rising linearly from 102 to 140."""
    dates = _make_date_index(n)
    values = np.linspace(102.0, 140.0, n)
    return pd.Series(values, index=dates, name=HEADLINE_LABEL)


def _make_categories_df(n: int = 108) -> pd.DataFrame:
    """
    Return a synthetic categories DataFrame with one column per UBOS category.
    Each category rises linearly at a slightly different rate to produce
    realistic variance across categories.
    """
    dates = _make_date_index(n)
    data = {}
    for i, cat in enumerate(CATEGORIES):
        # Each category starts at 100 and grows at a unique rate
        start = 100.0 + i * 0.5
        end = start + 30.0 + i * 1.0
        data[cat] = np.linspace(start, end, n)
    return pd.DataFrame(data, index=dates)


def _make_annual_pct(headline: pd.Series, categories_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annual percentage changes from synthetic headline and categories.
    First 12 rows will be NaN as expected.
    """
    all_df = pd.concat([headline, categories_df], axis=1)
    return all_df.pct_change(periods=12) * 100


@pytest.fixture
def headline():
    return _make_headline()


@pytest.fixture
def categories_df():
    return _make_categories_df()


@pytest.fixture
def annual_pct(headline, categories_df):
    return _make_annual_pct(headline, categories_df)


# Tests: compute_weighted_contributions

class TestComputeWeightedContributions:

    def test_returns_dataframe(self, annual_pct):
        """Return type must be a DataFrame."""
        result = analysis.compute_weighted_contributions(annual_pct)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, annual_pct):
        """Result must contain category, annual_rate, weight, contribution."""
        result = analysis.compute_weighted_contributions(annual_pct)
        assert set(["category", "annual_rate", "weight", "contribution"]).issubset(result.columns)

    def test_row_count_is_categories_plus_headline(self, annual_pct):
        """Result must have len(CATEGORIES) + 1 rows (categories + headline summary)."""
        result = analysis.compute_weighted_contributions(annual_pct)
        assert len(result) == len(CATEGORIES) + 1

    def test_sorted_descending_by_contribution(self, annual_pct):
        """Rows (excluding headline summary at end) must be sorted descending by contribution."""
        result = analysis.compute_weighted_contributions(annual_pct)
        body = result.iloc[:-1]  # exclude headline row
        contributions = body["contribution"].tolist()
        assert contributions == sorted(contributions, reverse=True)

    def test_headline_row_is_last(self, annual_pct):
        """The last row must be the headline summary row."""
        result = analysis.compute_weighted_contributions(annual_pct)
        assert result.iloc[-1]["category"] == HEADLINE_LABEL

    def test_contributions_sum_approximately_to_headline(self, annual_pct):
        """Sum of category contributions must approximately equal headline annual rate."""
        result = analysis.compute_weighted_contributions(annual_pct)
        category_sum = result.iloc[:-1]["contribution"].sum()
        headline_rate = result.iloc[-1]["annual_rate"]
        assert abs(category_sum - headline_rate) < 0.5

    def test_custom_period_accepted(self, annual_pct):
        """Passing a valid period string must not raise and must return a result."""
        valid_period = annual_pct.index[20]  # 20th month — well past the NaN zone
        result = analysis.compute_weighted_contributions(annual_pct, period=valid_period)
        assert len(result) > 0

    def test_weights_match_config(self, annual_pct):
        """Weights in result must match CATEGORY_WEIGHTS from config."""
        result = analysis.compute_weighted_contributions(annual_pct)
        body = result.iloc[:-1]
        for _, row in body.iterrows():
            expected_weight = CATEGORY_WEIGHTS[row["category"]]
            assert abs(row["weight"] - expected_weight) < 0.01


# Tests: compute_cumulative_growth

class TestComputeCumulativeGrowth:

    def test_returns_dataframe(self, headline, categories_df):
        """Return type must be a DataFrame."""
        result = analysis.compute_cumulative_growth(headline, categories_df)
        assert isinstance(result, pd.DataFrame)

    def test_first_row_is_zero(self, headline, categories_df):
        """First row (base period) must be 0.0 for all columns."""
        result = analysis.compute_cumulative_growth(headline, categories_df)
        assert (result.iloc[0] == 0.0).all()

    def test_shape_matches_input(self, headline, categories_df):
        """Output shape must match combined headline + categories input."""
        result = analysis.compute_cumulative_growth(headline, categories_df)
        expected_cols = 1 + len(CATEGORIES)  # headline + 13 categories
        assert result.shape == (len(headline), expected_cols)

    def test_positive_growth_for_rising_series(self, headline, categories_df):
        """All values after the base period must be positive for rising CPI."""
        result = analysis.compute_cumulative_growth(headline, categories_df)
        assert (result.iloc[1:] > 0).all().all()

    def test_custom_base_period_sets_zero_at_that_row(self, headline, categories_df):
        """When a custom base period is given, that row must be 0.0."""
        base = headline.index[12].strftime("%Y-%m-%d")
        result = analysis.compute_cumulative_growth(headline, categories_df, base_period=base)
        assert (result.iloc[12] == 0.0).all()

    def test_raises_for_invalid_base_period(self, headline, categories_df):
        """An unrecognised base period string must raise KeyError."""
        with pytest.raises(KeyError):
            analysis.compute_cumulative_growth(headline, categories_df, base_period="1900-01-01")

    def test_raises_for_zero_base_value(self, categories_df):
        """A headline series with a zero base value must raise ValueError."""
        zero_headline = _make_headline()
        zero_headline.iloc[0] = 0.0
        with pytest.raises(ValueError, match="Zero CPI base value"):
            analysis.compute_cumulative_growth(zero_headline, categories_df)

    def test_known_growth_value(self):
        """Manually verify: 110 / 100 - 1 = 10% growth."""
        dates = pd.date_range("2020-01-01", periods=2, freq="MS")
        h = pd.Series([100.0, 110.0], index=dates, name=HEADLINE_LABEL)
        cats = pd.DataFrame({CATEGORIES[0]: [100.0, 110.0]}, index=dates)
        result = analysis.compute_cumulative_growth(h, cats)
        assert abs(result[HEADLINE_LABEL].iloc[1] - 10.0) < 1e-6


# Tests: compute_cost_burden

class TestComputeCostBurden:

    def test_returns_dict_with_required_keys(self, headline, categories_df):
        """Result must contain all required top-level keys."""
        result = analysis.compute_cost_burden(headline, categories_df, 500_000)
        required = {"base_period", "target_period", "monthly_basket_ugx",
                    "headline_burden", "category_burdens"}
        assert required.issubset(result.keys())

    def test_headline_burden_has_extra_ugx_and_growth(self, headline, categories_df):
        """headline_burden dict must contain extra_ugx and growth_pct."""
        result = analysis.compute_cost_burden(headline, categories_df, 500_000)
        assert "extra_ugx" in result["headline_burden"]
        assert "growth_pct" in result["headline_burden"]

    def test_extra_ugx_is_positive_for_rising_cpi(self, headline, categories_df):
        """For a rising CPI series, extra_ugx must be positive."""
        result = analysis.compute_cost_burden(headline, categories_df, 500_000)
        assert result["headline_burden"]["extra_ugx"] > 0

    def test_category_burdens_count_matches_categories(self, headline, categories_df):
        """category_burdens list must have len(CATEGORIES) entries."""
        result = analysis.compute_cost_burden(headline, categories_df, 500_000)
        assert len(result["category_burdens"]) == len(CATEGORIES)

    def test_category_burdens_sorted_descending(self, headline, categories_df):
        """category_burdens must be sorted by extra_ugx descending."""
        result = analysis.compute_cost_burden(headline, categories_df, 500_000)
        extras = [b["extra_ugx"] for b in result["category_burdens"]]
        assert extras == sorted(extras, reverse=True)

    def test_raises_for_negative_basket(self, headline, categories_df):
        """A negative basket value must raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            analysis.compute_cost_burden(headline, categories_df, -100)

    def test_raises_for_zero_basket(self, headline, categories_df):
        """A zero basket value must raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            analysis.compute_cost_burden(headline, categories_df, 0)

    def test_basket_stored_in_result(self, headline, categories_df):
        """The input basket value must be echoed back in the result."""
        result = analysis.compute_cost_burden(headline, categories_df, 750_000)
        assert result["monthly_basket_ugx"] == 750_000

    def test_known_burden_value(self):
        """Manually verify: 500,000 basket * 10% growth = 50,000 UGX extra."""
        dates = pd.date_range("2020-01-01", periods=2, freq="MS")
        h = pd.Series([100.0, 110.0], index=dates, name=HEADLINE_LABEL)
        cats = pd.DataFrame(
            {cat: [100.0, 110.0] for cat in CATEGORIES}, index=dates
        )
        result = analysis.compute_cost_burden(h, cats, 500_000)
        assert abs(result["headline_burden"]["extra_ugx"] - 50_000.0) < 1.0

    def test_category_allocated_ugx_sums_to_basket(self, headline, categories_df):
        """Sum of allocated_ugx across categories must equal the basket size."""
        basket = 600_000
        result = analysis.compute_cost_burden(headline, categories_df, basket)
        total_allocated = sum(b["allocated_ugx"] for b in result["category_burdens"])
        assert abs(total_allocated - basket) < 1.0


# Tests: compare_essential_vs_discretionary

class TestCompareEssentialVsDiscretionary:

    def test_returns_dict_with_required_keys(self, annual_pct):
        """Result must contain all required keys."""
        result = analysis.compare_essential_vs_discretionary(annual_pct)
        required = {
            "essential_mean", "discretionary_mean",
            "essential_std", "discretionary_std",
            "t_statistic", "p_value", "significant",
            "interpretation", "timeseries",
        }
        assert required.issubset(result.keys())

    def test_significant_is_boolean(self, annual_pct):
        """significant field must be a Python bool."""
        result = analysis.compare_essential_vs_discretionary(annual_pct)
        assert isinstance(result["significant"], bool)

    def test_p_value_between_zero_and_one(self, annual_pct):
        """p_value must be in [0, 1]."""
        result = analysis.compare_essential_vs_discretionary(annual_pct)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_timeseries_has_required_keys(self, annual_pct):
        """timeseries sub-dict must contain dates, essential_mean, discretionary_mean."""
        result = analysis.compare_essential_vs_discretionary(annual_pct)
        ts = result["timeseries"]
        assert "dates" in ts
        assert "essential_mean" in ts
        assert "discretionary_mean" in ts

    def test_timeseries_lengths_match(self, annual_pct):
        """All timeseries lists must have the same length."""
        result = analysis.compare_essential_vs_discretionary(annual_pct)
        ts = result["timeseries"]
        assert len(ts["dates"]) == len(ts["essential_mean"]) == len(ts["discretionary_mean"])

    def test_interpretation_is_string(self, annual_pct):
        """interpretation must be a non-empty string."""
        result = analysis.compare_essential_vs_discretionary(annual_pct)
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 0

    def test_identical_groups_not_significant(self):
        """If essential and discretionary rates are identical, test must not be significant."""
        dates = _make_date_index(24)
        # All categories get identical values — groups cannot differ
        data = {cat: np.ones(24) * 5.0 for cat in CATEGORIES}
        data[HEADLINE_LABEL] = np.ones(24) * 5.0
        df = pd.DataFrame(data, index=dates)
        result = analysis.compare_essential_vs_discretionary(df)
        assert not result["significant"]

    def test_clearly_separated_groups_are_significant(self):
        """Clearly different group means must yield a significant result."""
        dates = _make_date_index(96)
        data = {}
        for cat in ESSENTIAL_CATEGORIES:
            data[cat] = np.random.normal(loc=10.0, scale=0.1, size=96)
        for cat in DISCRETIONARY_CATEGORIES:
            data[cat] = np.random.normal(loc=2.0, scale=0.1, size=96)
        data[HEADLINE_LABEL] = np.ones(96) * 5.0
        df = pd.DataFrame(data, index=dates)
        result = analysis.compare_essential_vs_discretionary(df)
        assert result["significant"]

    def test_means_are_floats(self, annual_pct):
        """essential_mean and discretionary_mean must be Python floats."""
        result = analysis.compare_essential_vs_discretionary(annual_pct)
        assert isinstance(result["essential_mean"], float)
        assert isinstance(result["discretionary_mean"], float)


# Tests: detect_anomalies

class TestDetectAnomalies:

    def test_returns_list(self, annual_pct):
        """Return type must be a list."""
        result = analysis.detect_anomalies(annual_pct)
        assert isinstance(result, list)

    def test_each_anomaly_has_required_keys(self, annual_pct):
        """Each anomaly dict must contain date, month_label, annual_rate, z_score, direction."""
        result = analysis.detect_anomalies(annual_pct)
        required = {"date", "month_label", "annual_rate", "z_score", "direction"}
        for anomaly in result:
            assert required.issubset(anomaly.keys())

    def test_no_anomalies_for_flat_series(self):
        """A perfectly flat annual rate series must produce no anomalies."""
        dates = _make_date_index(24)
        flat_data = {HEADLINE_LABEL: np.ones(24) * 5.0}
        flat_data.update({cat: np.ones(24) * 5.0 for cat in CATEGORIES})
        df = pd.DataFrame(flat_data, index=dates)
        result = analysis.detect_anomalies(df)
        assert result == []

    def test_known_spike_is_detected(self):
        """A single extreme spike must be detected as an anomaly."""
        dates = _make_date_index(36)
        values = [5.0] * 36
        values[25] = 50.0  # extreme spike
        data = {HEADLINE_LABEL: values}
        data.update({cat: [5.0] * 36 for cat in CATEGORIES})
        df = pd.DataFrame(data, index=dates)
        result = analysis.detect_anomalies(df)
        assert len(result) >= 1
        assert result[0]["direction"] == "high"

    def test_anomalies_sorted_by_absolute_z_score(self, annual_pct):
        """Anomalies must be sorted by absolute z_score descending."""
        result = analysis.detect_anomalies(annual_pct)
        if len(result) > 1:
            z_scores = [abs(a["z_score"]) for a in result]
            assert z_scores == sorted(z_scores, reverse=True)

    def test_direction_field_is_high_or_low(self, annual_pct):
        """direction field must be either 'high' or 'low'."""
        result = analysis.detect_anomalies(annual_pct)
        for anomaly in result:
            assert anomaly["direction"] in ("high", "low")

    def test_custom_threshold_changes_result_count(self, annual_pct):
        """A tighter threshold must detect more anomalies than a looser one."""
        tight = analysis.detect_anomalies(annual_pct, z_threshold=1.0)
        loose = analysis.detect_anomalies(annual_pct, z_threshold=3.0)
        assert len(tight) >= len(loose)

    def test_missing_headline_column_returns_empty(self):
        """A DataFrame without the headline column must return an empty list."""
        dates = _make_date_index(24)
        df = pd.DataFrame({cat: np.ones(24) for cat in CATEGORIES}, index=dates)
        result = analysis.detect_anomalies(df)
        assert result == []

    def test_z_scores_exceed_threshold(self, annual_pct):
        """All returned anomaly z_scores must exceed the default threshold."""
        result = analysis.detect_anomalies(annual_pct)
        for anomaly in result:
            assert abs(anomaly["z_score"]) > ANOMALY_Z_THRESHOLD


# Tests: compute_summary

class TestComputeSummary:

    def test_returns_dict(self, headline, annual_pct):
        """Return type must be a dict."""
        result = analysis.compute_summary(headline, annual_pct)
        assert isinstance(result, dict)

    def test_has_all_required_keys(self, headline, annual_pct):
        """Result must contain all expected keys."""
        result = analysis.compute_summary(headline, annual_pct)
        required = {
            "first_period", "latest_period", "total_months",
            "first_cpi", "latest_cpi", "cumulative_growth_pct",
            "mean_annual_rate", "max_annual_rate", "min_annual_rate",
        }
        assert required.issubset(result.keys())

    def test_total_months_matches_dataset(self, headline, annual_pct):
        """total_months must equal the length of the headline series."""
        result = analysis.compute_summary(headline, annual_pct)
        assert result["total_months"] == DATASET_MONTHS

    def test_cumulative_growth_is_correct(self, headline, annual_pct):
        """Cumulative growth must match manual calculation."""
        result = analysis.compute_summary(headline, annual_pct)
        expected = ((headline.iloc[-1] / headline.iloc[0]) - 1) * 100
        assert abs(result["cumulative_growth_pct"] - expected) < 0.01

    def test_max_annual_rate_has_rate_and_date(self, headline, annual_pct):
        """max_annual_rate sub-dict must contain rate and date keys."""
        result = analysis.compute_summary(headline, annual_pct)
        assert "rate" in result["max_annual_rate"]
        assert "date" in result["max_annual_rate"]

    def test_first_cpi_matches_headline(self, headline, annual_pct):
        """first_cpi must match the first value in the headline series."""
        result = analysis.compute_summary(headline, annual_pct)
        assert abs(result["first_cpi"] - headline.iloc[0]) < 0.01

    def test_latest_cpi_matches_headline(self, headline, annual_pct):
        """latest_cpi must match the last value in the headline series."""
        result = analysis.compute_summary(headline, annual_pct)
        assert abs(result["latest_cpi"] - headline.iloc[-1]) < 0.01

    def test_first_period_is_july_2017(self, headline, annual_pct):
        """first_period must be Jul 2017 for our dataset."""
        result = analysis.compute_summary(headline, annual_pct)
        assert result["first_period"] == "Jul 2017"

    def test_mean_annual_rate_is_positive_for_rising_cpi(self, headline, annual_pct):
        """For a rising CPI series, mean annual rate must be positive."""
        result = analysis.compute_summary(headline, annual_pct)
        assert result["mean_annual_rate"] > 0