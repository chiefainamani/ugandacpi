"""
Integration tests for the Uganda CPI Intelligence System Flask REST API.

Each test makes a real HTTP request to the Flask test client — no mocking
of the pipeline or analysis layer — so these tests verify end-to-end
behaviour from HTTP request through to JSON response.

Test coverage:
    - GET /api/v1/                           (5 tests)
    - GET /api/v1/summary                    (6 tests)
    - GET /api/v1/headline                   (5 tests)
    - GET /api/v1/categories                 (4 tests)
    - GET /api/v1/categories/<name>          (5 tests)
    - GET /api/v1/weights                    (4 tests)
    - GET /api/v1/drivers                    (6 tests)
    - GET /api/v1/growth                     (5 tests)
    - GET /api/v1/burden                     (7 tests)
    - GET /api/v1/essential-vs-discretionary (5 tests)
    - GET /api/v1/anomalies                  (5 tests)
    - GET /api/v1/forecast                   (selected ML model)
    - GET /api/v1/forecast/models            (model-selection evidence)
    - GET /api/v1/forecast/ols               (OLS benchmark)
    - GET /api/v1/forecast/validation        (5 tests)
    - Error handling / 404                   (5 tests)

Run with:
    uv run python -m pytest tests/test_api.py -v

Author : Ainamani Dickson & Lumu Richard
Project: Uganda CPI Intelligence System (BSE Final Year Project, UTAMU)
Version: 1.0.0
"""

import pytest
import json
from api.app import app
from pipeline.config import (
    API_PREFIX,
    CATEGORIES,
    CATEGORY_WEIGHTS,
    FORECAST_HORIZON,
    OLS_TRAINING_WINDOW,
    DATASET_MONTHS,
)


# Fixtures

@pytest.fixture(scope="module")
def client():
    """
    Flask test client fixture scoped to the module.

    Using module scope means the pipeline runs once for all 74 tests,
    keeping the test suite fast while still exercising real data.
    """
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _get(client, path: str, params: dict = None):
    """Helper — perform a GET request and return (response, json_data)."""
    url = f"{API_PREFIX}{path}"
    if params:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{query}"
    response = client.get(url)
    try:
        data = json.loads(response.data)
    except Exception:
        data = {}
    return response, data


# Tests: GET /api/v1/

class TestApiRoot:

    def test_status_200(self, client):
        """Root endpoint must return HTTP 200."""
        resp, _ = _get(client, "/")
        assert resp.status_code == 200

    def test_has_api_version(self, client):
        """Response must contain api_version field."""
        _, data = _get(client, "/")
        assert "api_version" in data

    def test_has_correct_dataset_months(self, client):
        """dataset_months must equal DATASET_MONTHS from config."""
        _, data = _get(client, "/")
        assert data["dataset_months"] == DATASET_MONTHS

    def test_has_13_categories(self, client):
        """categories count must be 13."""
        _, data = _get(client, "/")
        assert data["categories"] == 13

    def test_endpoints_list_includes_ml_forecast_endpoints(self, client):
        """endpoints list must advertise the ML forecast and model-selection routes."""
        _, data = _get(client, "/")
        assert len(data["endpoints"]) >= 13
        for route in ("/forecast", "/forecast/validation", "/forecast/models"):
            assert f"{API_PREFIX}{route}" in data["endpoints"]


# Tests: GET /api/v1/summary

class TestSummary:

    def test_status_200(self, client):
        resp, _ = _get(client, "/summary")
        assert resp.status_code == 200

    def test_has_required_keys(self, client):
        """Summary must contain all expected descriptive statistic fields."""
        _, data = _get(client, "/summary")
        required = {
            "first_period", "latest_period", "total_months",
            "first_cpi", "latest_cpi", "cumulative_growth_pct",
            "mean_annual_rate", "max_annual_rate", "min_annual_rate",
        }
        assert required.issubset(data.keys())

    def test_total_months_is_108(self, client):
        """total_months must equal DATASET_MONTHS (108)."""
        _, data = _get(client, "/summary")
        assert data["total_months"] == DATASET_MONTHS

    def test_cumulative_growth_is_positive(self, client):
        """Cumulative growth must be positive for our rising CPI series."""
        _, data = _get(client, "/summary")
        assert data["cumulative_growth_pct"] > 0

    def test_first_period_is_jul_2017(self, client):
        """First period must be Jul 2017."""
        _, data = _get(client, "/summary")
        assert data["first_period"] == "Jul 2017"

    def test_latest_cpi_is_plausible(self, client):
        """Latest CPI must be in a realistic range (100–200)."""
        _, data = _get(client, "/summary")
        assert 100 < data["latest_cpi"] < 200


# Tests: GET /api/v1/headline

class TestHeadline:

    def test_status_200(self, client):
        resp, _ = _get(client, "/headline")
        assert resp.status_code == 200

    def test_has_cpi_monthly_annual_keys(self, client):
        """Response must contain cpi, monthly_pct, annual_pct keys."""
        _, data = _get(client, "/headline")
        assert "cpi" in data
        assert "monthly_pct" in data
        assert "annual_pct" in data

    def test_cpi_has_108_observations(self, client):
        """CPI dict must have DATASET_MONTHS entries."""
        _, data = _get(client, "/headline")
        assert len(data["cpi"]) == DATASET_MONTHS

    def test_annual_pct_first_entries_are_null(self, client):
        """First 12 annual_pct values must be null (no prior year data)."""
        _, data = _get(client, "/headline")
        values = list(data["annual_pct"].values())
        assert all(v is None for v in values[:12])

    def test_cpi_values_are_numeric(self, client):
        """All CPI values must be numeric (float or int)."""
        _, data = _get(client, "/headline")
        for v in data["cpi"].values():
            assert isinstance(v, (int, float))


# Tests: GET /api/v1/categories

class TestCategories:

    def test_status_200(self, client):
        resp, _ = _get(client, "/categories")
        assert resp.status_code == 200

    def test_has_categories_key(self, client):
        """Response must have a 'categories' key."""
        _, data = _get(client, "/categories")
        assert "categories" in data

    def test_has_13_categories(self, client):
        """categories dict must have 13 entries."""
        _, data = _get(client, "/categories")
        assert len(data["categories"]) == 13

    def test_each_category_has_108_observations(self, client):
        """Each category series must have DATASET_MONTHS observations."""
        _, data = _get(client, "/categories")
        for cat, series in data["categories"].items():
            assert len(series) == DATASET_MONTHS, f"{cat} has {len(series)} obs"


# Tests: GET /api/v1/categories/<name>

class TestCategorySingle:

    def test_status_200_for_valid_category(self, client):
        """A valid category name must return HTTP 200."""
        name = CATEGORIES[0]
        resp, _ = _get(client, f"/categories/{name}")
        assert resp.status_code == 200

    def test_has_required_keys(self, client):
        """Response must contain category, weight, cpi, annual_pct."""
        name = CATEGORIES[0]
        _, data = _get(client, f"/categories/{name}")
        assert {"category", "weight", "cpi", "annual_pct"}.issubset(data.keys())

    def test_weight_matches_config(self, client):
        """Returned weight must match CATEGORY_WEIGHTS in config."""
        name = CATEGORIES[0]
        _, data = _get(client, f"/categories/{name}")
        assert abs(data["weight"] - CATEGORY_WEIGHTS[name]) < 0.01

    def test_status_404_for_unknown_category(self, client):
        """An unrecognised category name must return HTTP 404."""
        resp, data = _get(client, "/categories/NonExistentCategory")
        assert resp.status_code == 404
        assert "error" in data

    def test_cpi_series_length(self, client):
        """Single category CPI series must have DATASET_MONTHS entries."""
        name = CATEGORIES[3]
        _, data = _get(client, f"/categories/{name}")
        assert len(data["cpi"]) == DATASET_MONTHS


# Tests: GET /api/v1/weights

class TestWeights:

    def test_status_200(self, client):
        resp, _ = _get(client, "/weights")
        assert resp.status_code == 200

    def test_has_weights_and_total(self, client):
        """Response must contain weights dict and total."""
        _, data = _get(client, "/weights")
        assert "weights" in data
        assert "total" in data

    def test_weights_count_is_13(self, client):
        """weights dict must have 13 entries."""
        _, data = _get(client, "/weights")
        assert len(data["weights"]) == 13

    def test_total_is_approximately_1000(self, client):
        """total must sum to approximately 1000."""
        _, data = _get(client, "/weights")
        assert abs(data["total"] - 1000.0) < 1.0


# Tests: GET /api/v1/drivers

class TestDrivers:

    def test_status_200(self, client):
        resp, _ = _get(client, "/drivers")
        assert resp.status_code == 200

    def test_has_period_and_drivers(self, client):
        """Response must contain period and drivers keys."""
        _, data = _get(client, "/drivers")
        assert "period" in data
        assert "drivers" in data

    def test_drivers_count_is_14(self, client):
        """drivers list must have 13 categories + 1 headline = 14 rows."""
        _, data = _get(client, "/drivers")
        assert len(data["drivers"]) == 14

    def test_each_driver_has_required_fields(self, client):
        """Each driver record must have category, annual_rate, weight, contribution."""
        _, data = _get(client, "/drivers")
        required = {"category", "annual_rate", "weight", "contribution"}
        for driver in data["drivers"]:
            assert required.issubset(driver.keys())

    def test_contributions_sum_near_headline(self, client):
        """Sum of category contributions must approximately equal headline rate."""
        _, data = _get(client, "/drivers")
        drivers = data["drivers"]
        headline = next(d for d in drivers if d["category"] == "All Items")
        category_sum = sum(
            d["contribution"] for d in drivers if d["category"] != "All Items"
        )
        assert abs(category_sum - headline["annual_rate"]) < 0.5

    def test_invalid_period_returns_error(self, client):
        """An invalid period parameter must return an error response."""
        resp, data = _get(client, "/drivers", {"period": "1800-01-01"})
        assert resp.status_code in (400, 404, 500)
        assert "error" in data


# Tests: GET /api/v1/growth

class TestGrowth:

    def test_status_200(self, client):
        resp, _ = _get(client, "/growth")
        assert resp.status_code == 200

    def test_has_base_period_and_growth(self, client):
        """Response must contain base_period and growth keys."""
        _, data = _get(client, "/growth")
        assert "base_period" in data
        assert "growth" in data

    def test_growth_has_14_series(self, client):
        """growth dict must have 14 series (headline + 13 categories)."""
        _, data = _get(client, "/growth")
        assert len(data["growth"]) == 14

    def test_custom_base_period_accepted(self, client):
        """Passing a valid base period must return 200."""
        resp, _ = _get(client, "/growth", {"base": "2020-01-01"})
        assert resp.status_code == 200

    def test_invalid_base_period_returns_error(self, client):
        """An invalid base period must return an error."""
        resp, data = _get(client, "/growth", {"base": "1800-01-01"})
        assert resp.status_code in (400, 404, 500)
        assert "error" in data


# Tests: GET /api/v1/burden

class TestBurden:

    def test_status_200_with_valid_basket(self, client):
        resp, _ = _get(client, "/burden", {"basket": 500000})
        assert resp.status_code == 200

    def test_missing_basket_returns_400(self, client):
        """Omitting basket parameter must return HTTP 400."""
        resp, data = _get(client, "/burden")
        assert resp.status_code == 400
        assert "error" in data

    def test_invalid_basket_returns_400(self, client):
        """A non-numeric basket must return HTTP 400."""
        resp, data = _get(client, "/burden", {"basket": "abc"})
        assert resp.status_code == 400

    def test_has_required_keys(self, client):
        """Response must contain all required top-level keys."""
        _, data = _get(client, "/burden", {"basket": 500000})
        required = {
            "base_period", "target_period", "monthly_basket_ugx",
            "headline_burden", "category_burdens",
        }
        assert required.issubset(data.keys())

    def test_extra_ugx_is_positive(self, client):
        """For our rising CPI, extra UGX must be positive."""
        _, data = _get(client, "/burden", {"basket": 500000})
        assert data["headline_burden"]["extra_ugx"] > 0

    def test_category_burdens_count_is_13(self, client):
        """category_burdens must have 13 entries."""
        _, data = _get(client, "/burden", {"basket": 500000})
        assert len(data["category_burdens"]) == 13

    def test_basket_echoed_in_response(self, client):
        """monthly_basket_ugx must equal the requested basket."""
        _, data = _get(client, "/burden", {"basket": 750000})
        assert data["monthly_basket_ugx"] == 750000.0


# Tests: GET /api/v1/essential-vs-discretionary

class TestEssentialVsDiscretionary:

    def test_status_200(self, client):
        resp, _ = _get(client, "/essential-vs-discretionary")
        assert resp.status_code == 200

    def test_has_required_keys(self, client):
        """Response must contain all t-test output keys."""
        _, data = _get(client, "/essential-vs-discretionary")
        required = {
            "essential_mean", "discretionary_mean",
            "t_statistic", "p_value", "significant",
            "interpretation", "timeseries",
        }
        assert required.issubset(data.keys())

    def test_p_value_between_0_and_1(self, client):
        """p_value must be in [0, 1]."""
        _, data = _get(client, "/essential-vs-discretionary")
        assert 0.0 <= data["p_value"] <= 1.0

    def test_significant_is_boolean(self, client):
        """significant field must be a boolean."""
        _, data = _get(client, "/essential-vs-discretionary")
        assert isinstance(data["significant"], bool)

    def test_timeseries_has_dates(self, client):
        """timeseries sub-object must contain a dates list."""
        _, data = _get(client, "/essential-vs-discretionary")
        assert "dates" in data["timeseries"]


# Tests: GET /api/v1/anomalies

class TestAnomalies:

    def test_status_200(self, client):
        resp, _ = _get(client, "/anomalies")
        assert resp.status_code == 200

    def test_has_required_keys(self, client):
        """Response must contain threshold, total_anomalies, anomalies."""
        _, data = _get(client, "/anomalies")
        assert {"threshold", "total_anomalies", "anomalies"}.issubset(data.keys())

    def test_default_threshold_is_2(self, client):
        """Default threshold must be 2.0."""
        _, data = _get(client, "/anomalies")
        assert data["threshold"] == 2.0

    def test_custom_threshold_accepted(self, client):
        """Passing a custom threshold must return 200."""
        resp, _ = _get(client, "/anomalies", {"threshold": 1.5})
        assert resp.status_code == 200

    def test_invalid_threshold_returns_400(self, client):
        """A non-numeric threshold must return HTTP 400."""
        resp, data = _get(client, "/anomalies", {"threshold": "abc"})
        assert resp.status_code == 400


# Tests: GET /api/v1/forecast  (selected machine-learning model)

class TestForecast:

    def test_status_200(self, client):
        resp, _ = _get(client, "/forecast")
        assert resp.status_code == 200

    def test_has_required_keys(self, client):
        """Response must contain all forecast output keys."""
        _, data = _get(client, "/forecast")
        required = {
            "model", "horizon", "confidence",
            "last_actual_date", "last_actual_cpi",
            "forecast_dates", "forecast_labels",
            "point", "lower", "upper",
            "annual_pct_point", "annual_pct_lower", "annual_pct_upper",
        }
        assert required.issubset(data.keys())

    def test_model_is_one_of_the_four_candidates(self, client):
        _, data = _get(client, "/forecast")
        assert data["model"] in {"Lasso", "RandomForest", "GradientBoosting", "SVR"}

    def test_default_horizon_is_3(self, client):
        """Default horizon must equal FORECAST_HORIZON (3)."""
        _, data = _get(client, "/forecast")
        assert data["horizon"] == FORECAST_HORIZON

    def test_forecast_dates_length_equals_horizon(self, client):
        _, data = _get(client, "/forecast")
        assert len(data["forecast_dates"]) == data["horizon"]

    def test_custom_horizon_accepted(self, client):
        """Passing horizon=1 must return 1 forecast date."""
        _, data = _get(client, "/forecast", {"horizon": 1})
        assert len(data["forecast_dates"]) == 1

    def test_horizon_out_of_range_returns_400(self, client):
        """The ML models support horizons 1-3 only; 4 and 13 must give 400."""
        for bad in (0, 4, 13):
            resp, data = _get(client, "/forecast", {"horizon": bad})
            assert resp.status_code == 400
            assert "error" in data

    def test_non_numeric_horizon_returns_400(self, client):
        resp, _ = _get(client, "/forecast", {"horizon": "abc"})
        assert resp.status_code == 400

    def test_point_forecasts_are_near_last_actual(self, client):
        """Point forecasts must be within +/-20 of the last actual CPI."""
        _, data = _get(client, "/forecast")
        last = data["last_actual_cpi"]
        for pt in data["point"]:
            assert abs(pt - last) < 20.0

    def test_interval_brackets_point(self, client):
        _, data = _get(client, "/forecast")
        for lo, pt, hi in zip(data["lower"], data["point"], data["upper"]):
            assert lo < pt < hi


# Tests: GET /api/v1/forecast/validation  (selected machine-learning model)

class TestForecastValidation:

    def test_status_200(self, client):
        resp, _ = _get(client, "/forecast/validation")
        assert resp.status_code == 200

    def test_has_required_keys(self, client):
        """Response must contain model, mape, rmse, mae, mape_by_horizon."""
        _, data = _get(client, "/forecast/validation")
        required = {"model", "n_validation_steps", "horizon", "mape", "rmse",
                    "mae", "mape_by_horizon"}
        assert required.issubset(data.keys())

    def test_mape_below_proposal_target(self, client):
        """MAPE on real data must be below the 2% proposal target."""
        _, data = _get(client, "/forecast/validation")
        assert data["mape"] < 2.0

    def test_steps_key_not_in_response(self, client):
        _, data = _get(client, "/forecast/validation")
        assert "steps" not in data

    def test_mape_by_horizon_has_3_entries(self, client):
        _, data = _get(client, "/forecast/validation")
        assert len(data["mape_by_horizon"]) == FORECAST_HORIZON

    def test_validated_model_matches_forecast_model(self, client):
        _, fc = _get(client, "/forecast")
        _, val = _get(client, "/forecast/validation")
        assert fc["model"] == val["model"]


# Tests: GET /api/v1/forecast/models  (model-selection evidence)

class TestForecastModels:

    def test_status_200(self, client):
        resp, _ = _get(client, "/forecast/models")
        assert resp.status_code == 200

    def test_has_required_keys(self, client):
        _, data = _get(client, "/forecast/models")
        required = {"selected_model", "selection_rule", "evaluation",
                    "overall_metrics", "metrics_by_horizon", "ranking",
                    "best_baseline", "beats_best_baseline", "candidates"}
        assert required.issubset(data.keys())

    def test_four_ml_candidates_ranked(self, client):
        _, data = _get(client, "/forecast/models")
        assert len(data["ranking"]) == 4
        assert len(data["candidates"]) == 4

    def test_selected_model_is_ranked_first(self, client):
        _, data = _get(client, "/forecast/models")
        assert data["ranking"][0]["model"] == data["selected_model"]

    def test_baselines_are_included_in_metrics(self, client):
        _, data = _get(client, "/forecast/models")
        names = {row["model"] for row in data["overall_metrics"]}
        assert {"Naive-drift", "OLS-trend"}.issubset(names)

    def test_selected_model_beats_ols_benchmark(self, client):
        _, data = _get(client, "/forecast/models")
        mape = {r["model"]: r["MAPE_%"] for r in data["overall_metrics"]}
        assert mape[data["selected_model"]] < mape["OLS-trend"]


# Tests: GET /api/v1/forecast/ols  (OLS benchmark)

class TestForecastOLS:

    def test_status_200(self, client):
        resp, _ = _get(client, "/forecast/ols")
        assert resp.status_code == 200

    def test_has_required_keys(self, client):
        """Response must contain all forecast output keys."""
        _, data = _get(client, "/forecast/ols")
        required = {
            "training_window", "horizon", "confidence",
            "last_actual_date", "last_actual_cpi",
            "forecast_dates", "forecast_labels",
            "point", "lower", "upper",
            "slope", "intercept",
        }
        assert required.issubset(data.keys())

    def test_default_horizon_is_3(self, client):
        """Default horizon must equal FORECAST_HORIZON (3)."""
        _, data = _get(client, "/forecast/ols")
        assert data["horizon"] == FORECAST_HORIZON

    def test_forecast_dates_length_equals_horizon(self, client):
        """forecast_dates list length must equal horizon."""
        _, data = _get(client, "/forecast/ols")
        assert len(data["forecast_dates"]) == data["horizon"]

    def test_custom_horizon_accepted(self, client):
        """Passing horizon=1 must return 1 forecast date."""
        _, data = _get(client, "/forecast/ols", {"horizon": 1})
        assert len(data["forecast_dates"]) == 1

    def test_horizon_out_of_range_returns_400(self, client):
        """horizon > 12 must return HTTP 400."""
        resp, data = _get(client, "/forecast/ols", {"horizon": 13})
        assert resp.status_code == 400

    def test_point_forecasts_are_near_last_actual(self, client):
        """Point forecasts must be within ±20 of the last actual CPI."""
        _, data = _get(client, "/forecast/ols")
        last = data["last_actual_cpi"]
        for pt in data["point"]:
            assert abs(pt - last) < 20.0


# Tests: GET /api/v1/forecast/ols/validation

class TestForecastOLSValidation:

    def test_status_200(self, client):
        resp, _ = _get(client, "/forecast/ols/validation")
        assert resp.status_code == 200

    def test_has_required_keys(self, client):
        """Response must contain mape, rmse, mae, mape_by_horizon."""
        _, data = _get(client, "/forecast/ols/validation")
        required = {"n_validation_steps", "horizon", "mape", "rmse", "mae", "mape_by_horizon"}
        assert required.issubset(data.keys())

    def test_mape_below_proposal_target(self, client):
        """MAPE on real data must be below the 2% proposal target."""
        _, data = _get(client, "/forecast/ols/validation")
        assert data["mape"] < 2.0

    def test_steps_key_not_in_response(self, client):
        """Verbose steps list must be excluded from the API response."""
        _, data = _get(client, "/forecast/ols/validation")
        assert "steps" not in data

    def test_mape_by_horizon_has_3_entries(self, client):
        """mape_by_horizon must have FORECAST_HORIZON entries."""
        _, data = _get(client, "/forecast/ols/validation")
        assert len(data["mape_by_horizon"]) == FORECAST_HORIZON


# Tests: Error handling

class TestErrorHandling:

    def test_unknown_endpoint_returns_404(self, client):
        """An unrecognised endpoint must return HTTP 404."""
        resp = client.get(f"{API_PREFIX}/nonexistent")
        assert resp.status_code == 404

    def test_404_response_has_error_key(self, client):
        """404 response body must contain an error key."""
        resp = client.get(f"{API_PREFIX}/nonexistent")
        data = json.loads(resp.data)
        assert "error" in data

    def test_content_type_is_json(self, client):
        """All successful responses must have application/json content type."""
        resp, _ = _get(client, "/")
        assert "application/json" in resp.content_type

    def test_negative_basket_returns_400(self, client):
        """A negative basket value must return HTTP 400."""
        resp, data = _get(client, "/burden", {"basket": -1})
        assert resp.status_code == 400
        assert "error" in data

    def test_zero_threshold_returns_400(self, client):
        """A zero anomaly threshold must return HTTP 400."""
        resp, data = _get(client, "/anomalies", {"threshold": 0})
        assert resp.status_code == 400
        assert "error" in data