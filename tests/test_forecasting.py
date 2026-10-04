"""
test_forecasting.py
-------------------
Unit tests for the Uganda CPI Intelligence System forecasting module.

Test coverage:
    - _fit_ols: slope/intercept correctness, residual stderr, known values
    - _predict_ols: output structure, confidence interval width, horizon length
    - generate_forecast: return keys, date labels, insufficient data guard,
      point forecast direction, confidence level effect
    - walk_forward_validation: output structure, MAPE range, step count,
      horizon-level MAPE list, insufficient data guard

Run with:
    uv run python -m pytest tests/test_forecasting.py -v

Author : Ainamani Dickson & Lumu Richard
Project: Uganda CPI Intelligence System (BSE Final Year Project, UTAMU)
Version: 1.0.0
"""

import pytest
import numpy as np
import pandas as pd

from forecasting import models
from pipeline.config import (
    OLS_TRAINING_WINDOW,
    FORECAST_HORIZON,
    WALK_FORWARD_MIN_TRAIN,
    HEADLINE_LABEL,
    DATASET_MONTHS,
)


# Fixtures

def _make_headline(n: int = 108, start: float = 102.0, slope: float = 0.37) -> pd.Series:
    """
    Return a synthetic headline CPI series with a known linear trend.

    Using a known slope makes it straightforward to verify OLS output
    against expected values.
    """
    dates = pd.date_range(start="2017-07-01", periods=n, freq="MS")
    values = start + slope * np.arange(n)
    return pd.Series(values, index=dates, name=HEADLINE_LABEL)


@pytest.fixture
def headline():
    """Default 108-month synthetic headline series."""
    return _make_headline()


@pytest.fixture
def short_headline():
    """Headline series shorter than OLS_TRAINING_WINDOW — triggers error paths."""
    return _make_headline(n=OLS_TRAINING_WINDOW - 1)


@pytest.fixture
def y_train(headline):
    """Training array: last OLS_TRAINING_WINDOW values from the headline."""
    return headline.iloc[-OLS_TRAINING_WINDOW:].values.astype(float)


# Tests: _fit_ols

class TestFitOls:

    def test_returns_three_values(self, y_train):
        """_fit_ols must return exactly three values: slope, intercept, stderr."""
        result = models._fit_ols(y_train)
        assert len(result) == 3

    def test_slope_is_float(self, y_train):
        """slope must be a Python float."""
        slope, _, _ = models._fit_ols(y_train)
        assert isinstance(slope, float)

    def test_intercept_is_float(self, y_train):
        """intercept must be a Python float."""
        _, intercept, _ = models._fit_ols(y_train)
        assert isinstance(intercept, float)

    def test_residual_stderr_is_non_negative(self, y_train):
        """Residual standard error must be >= 0."""
        _, _, stderr = models._fit_ols(y_train)
        assert stderr >= 0.0

    def test_known_slope_recovered(self):
        """For a perfectly linear series with slope 0.37, OLS must recover ~0.37."""
        y = np.array([102.0 + 0.37 * t for t in range(24)])
        slope, _, _ = models._fit_ols(y)
        assert abs(slope - 0.37) < 1e-6

    def test_known_intercept_recovered(self):
        """For a perfectly linear series starting at 102.0, intercept must be ~102.0."""
        y = np.array([102.0 + 0.37 * t for t in range(24)])
        _, intercept, _ = models._fit_ols(y)
        assert abs(intercept - 102.0) < 1e-6

    def test_perfect_fit_has_zero_stderr(self):
        """A perfectly linear series produces zero residual standard error."""
        y = np.array([100.0 + 0.5 * t for t in range(30)])
        _, _, stderr = models._fit_ols(y)
        assert stderr < 1e-6

    def test_noisy_series_has_positive_stderr(self):
        """A series with noise must produce a positive residual standard error."""
        rng = np.random.default_rng(42)
        y = np.array([100.0 + 0.5 * t for t in range(30)]) + rng.normal(0, 1, 30)
        _, _, stderr = models._fit_ols(y)
        assert stderr > 0.0


# Tests: _predict_ols

class TestPredictOls:

    def test_returns_dict_with_required_keys(self):
        """Result must contain point, lower, upper keys."""
        result = models._predict_ols(0.37, 102.0, 24, 3, 0.5)
        assert set(result.keys()) == {"point", "lower", "upper"}

    def test_point_forecast_length_equals_horizon(self):
        """point list length must equal the requested horizon."""
        result = models._predict_ols(0.37, 102.0, 24, 3, 0.5)
        assert len(result["point"]) == 3

    def test_lower_length_equals_horizon(self):
        """lower list length must equal the requested horizon."""
        result = models._predict_ols(0.37, 102.0, 24, 3, 0.5)
        assert len(result["lower"]) == 3

    def test_upper_length_equals_horizon(self):
        """upper list length must equal the requested horizon."""
        result = models._predict_ols(0.37, 102.0, 24, 3, 0.5)
        assert len(result["upper"]) == 3

    def test_lower_less_than_point(self):
        """lower bound must always be below point forecast."""
        result = models._predict_ols(0.37, 102.0, 24, 3, 1.0)
        for lo, pt in zip(result["lower"], result["point"]):
            assert lo < pt

    def test_upper_greater_than_point(self):
        """upper bound must always be above point forecast."""
        result = models._predict_ols(0.37, 102.0, 24, 3, 1.0)
        for hi, pt in zip(result["upper"], result["point"]):
            assert hi > pt

    def test_wider_interval_with_higher_confidence(self):
        """90% interval must be wider than 50% interval."""
        r50 = models._predict_ols(0.37, 102.0, 24, 3, 0.5, confidence=0.50)
        r90 = models._predict_ols(0.37, 102.0, 24, 3, 0.5, confidence=0.90)
        width50 = r50["upper"][0] - r50["lower"][0]
        width90 = r90["upper"][0] - r90["lower"][0]
        assert width90 > width50

    def test_point_forecast_increases_with_positive_slope(self):
        """For a positive slope, each successive point forecast must be larger."""
        result = models._predict_ols(0.37, 102.0, 24, 3, 0.1)
        assert result["point"][0] < result["point"][1] < result["point"][2]

    def test_known_point_value(self):
        """
        For slope=1.0, intercept=0.0, n_train=10, the forecast at h=1
        is intercept + slope * (n_train - 1 + 1) = 10.0.
        """
        result = models._predict_ols(1.0, 0.0, 10, 1, 0.0)
        assert abs(result["point"][0] - 10.0) < 1e-4


# Tests: generate_forecast

class TestGenerateForecast:

    def test_returns_dict_with_required_keys(self, headline):
        """Result must contain all expected keys."""
        result = models.generate_forecast(headline)
        required = {
            "training_window", "horizon", "confidence",
            "last_actual_date", "last_actual_cpi",
            "forecast_dates", "forecast_labels",
            "point", "lower", "upper",
            "slope", "intercept",
        }
        assert required.issubset(result.keys())

    def test_forecast_dates_length_equals_horizon(self, headline):
        """forecast_dates list length must equal FORECAST_HORIZON."""
        result = models.generate_forecast(headline)
        assert len(result["forecast_dates"]) == FORECAST_HORIZON

    def test_forecast_labels_length_equals_horizon(self, headline):
        """forecast_labels list length must equal FORECAST_HORIZON."""
        result = models.generate_forecast(headline)
        assert len(result["forecast_labels"]) == FORECAST_HORIZON

    def test_first_forecast_date_is_next_month(self, headline):
        """First forecast date must be one month after the last actual date."""
        result = models.generate_forecast(headline)
        last_actual = pd.to_datetime(result["last_actual_date"])
        first_forecast = pd.to_datetime(result["forecast_dates"][0])
        expected = last_actual + pd.DateOffset(months=1)
        assert first_forecast == expected

    def test_forecast_dates_are_sequential_monthly(self, headline):
        """Forecast dates must be consecutive month-start dates."""
        result = models.generate_forecast(headline)
        dates = pd.to_datetime(result["forecast_dates"])
        for i in range(1, len(dates)):
            delta = dates[i] - dates[i - 1]
            assert 28 <= delta.days <= 31

    def test_point_forecasts_are_plausible(self, headline):
        """Point forecasts must be within ±20 of the last actual CPI."""
        result = models.generate_forecast(headline)
        last_cpi = result["last_actual_cpi"]
        for pt in result["point"]:
            assert abs(pt - last_cpi) < 20.0

    def test_raises_for_insufficient_data(self, short_headline):
        """Series shorter than training_window must raise ValueError."""
        with pytest.raises(ValueError, match="training_window"):
            models.generate_forecast(short_headline)

    def test_custom_horizon_respected(self, headline):
        """Passing horizon=1 must return exactly 1 forecast date."""
        result = models.generate_forecast(headline, horizon=1)
        assert len(result["forecast_dates"]) == 1

    def test_custom_training_window_respected(self, headline):
        """training_window value must be echoed back in result."""
        result = models.generate_forecast(headline, training_window=36)
        assert result["training_window"] == 36

    def test_slope_is_positive_for_rising_series(self, headline):
        """For a monotonically rising CPI series, slope must be positive."""
        result = models.generate_forecast(headline)
        assert result["slope"] > 0


# Tests: walk_forward_validation

class TestWalkForwardValidation:

    def test_returns_dict_with_required_keys(self, headline):
        """Result must contain all expected keys."""
        result = models.walk_forward_validation(headline)
        required = {
            "n_validation_steps", "horizon",
            "mape", "rmse", "mae",
            "mape_by_horizon", "steps",
        }
        assert required.issubset(result.keys())

    # test_mape_is_positive
    def test_mape_is_positive(self, headline):
        """MAPE must be non-negative (zero only for a perfectly linear series)."""
        result = models.walk_forward_validation(headline)
        assert result["mape"] >= 0.0

    # test_rmse_is_positive
    def test_rmse_is_positive(self, headline):
        """RMSE must be non-negative (zero only for a perfectly linear series)."""
        result = models.walk_forward_validation(headline)
        assert result["rmse"] >= 0.0

    # test_mae_is_positive
    def test_mae_is_positive(self, headline):
        """MAE must be non-negative (zero only for a perfectly linear series)."""
        result = models.walk_forward_validation(headline)
        assert result["mae"] >= 0.0

    # test_raises_for_insufficient_data
    def test_raises_for_insufficient_data(self):
        """Series too short for even one validation step must raise ValueError."""
        tiny = _make_headline(n=WALK_FORWARD_MIN_TRAIN + FORECAST_HORIZON - 1)
        with pytest.raises(ValueError, match="Need at least"):
            models.walk_forward_validation(tiny)

    def test_mape_by_horizon_length_equals_horizon(self, headline):
        """mape_by_horizon list must have one entry per horizon step."""
        result = models.walk_forward_validation(headline)
        assert len(result["mape_by_horizon"]) == FORECAST_HORIZON

    def test_steps_list_length_is_correct(self, headline):
        """
        Number of validation steps must equal
        len(headline) - min_train - horizon + 1.
        """
        n = len(headline.dropna())
        expected_steps = n - WALK_FORWARD_MIN_TRAIN - FORECAST_HORIZON + 1
        result = models.walk_forward_validation(headline)
        assert result["n_validation_steps"] == expected_steps

    def test_each_step_has_required_keys(self, headline):
        """Each step dict must contain train_end, forecast_start, actuals, forecasts."""
        result = models.walk_forward_validation(headline)
        required = {"train_end", "forecast_start", "actuals", "forecasts"}
        for step in result["steps"]:
            assert required.issubset(step.keys())

    def test_actuals_length_equals_horizon_per_step(self, headline):
        """Each step's actuals list must have FORECAST_HORIZON entries."""
        result = models.walk_forward_validation(headline)
        for step in result["steps"]:
            assert len(step["actuals"]) == FORECAST_HORIZON

    def test_forecasts_length_equals_horizon_per_step(self, headline):
        """Each step's forecasts list must have FORECAST_HORIZON entries."""
        result = models.walk_forward_validation(headline)
        for step in result["steps"]:
            assert len(step["forecasts"]) == FORECAST_HORIZON

    def test_mape_below_proposal_target_on_real_data(self):
        """
        On the actual UBOS dataset, MAPE must be below 2% — the target
        stated in the project proposal.
        """
        from pipeline.ingestion import run_ingestion
        data = run_ingestion()
        result = models.walk_forward_validation(data["headline"])
        assert result["mape"] < 2.0, (
            f"MAPE {result['mape']:.4f}% exceeds the 2% proposal target. "
            "Consider reviewing the training window or forecast horizon."
        )

    def test_horizon_echoed_in_result(self, headline):
        """horizon value must be echoed back in the result dict."""
        result = models.walk_forward_validation(headline, horizon=FORECAST_HORIZON)
        assert result["horizon"] == FORECAST_HORIZON