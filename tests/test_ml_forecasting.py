"""
test_ml_forecasting.py
----------------------
Unit tests for the machine-learning forecasting module and the model-selection
experiment.

Test coverage:
    - build_features / build_targets / make_training_set: shape, NaN handling,
      correct arithmetic, and NO look-ahead leakage
    - tune_hyperparameters / fit_predict_growth: valid output for all 4 models
    - generate_ml_forecast: output structure, interval ordering, error guards
    - run_walk_forward: structure and a strict no-leakage check
    - diebold_mariano, rank_models, _append_step_forecasts: statistics helpers
    - load_selection: saved selection record is consistent

Run with:
    uv run python -m pytest tests/test_ml_forecasting.py -v

Author : Ainamani Dickson & Lumu Richard
Project: Uganda CPI Intelligence System (BSE Final Year Project, UTAMU)
Version: 2.0.0
"""

import numpy as np
import pandas as pd
import pytest

from forecasting import ml_models, model_selection
from forecasting.ml_models import (
    FEATURE_NAMES,
    FIRST_FEATURE_ROW,
    MODEL_NAMES,
    build_features,
    build_targets,
    fit_predict_growth,
    generate_ml_forecast,
    make_training_set,
    tune_hyperparameters,
)
from pipeline.config import FORECAST_HORIZON, HEADLINE_LABEL


# Helpers

def _make_headline(n: int = 80, seed: int = 0) -> pd.Series:
    """Synthetic CPI: random walk with positive drift (about 0.3% a month)."""
    rng = np.random.default_rng(seed)
    growth = 0.003 + 0.004 * rng.standard_normal(n)
    values = 100.0 * np.exp(np.cumsum(growth))
    dates = pd.date_range("2017-07-01", periods=n, freq="MS")
    return pd.Series(values, index=dates, name=HEADLINE_LABEL)


@pytest.fixture(scope="module")
def headline():
    return _make_headline()


# Feature engineering

class TestFeatures:

    def test_columns_match_feature_names(self, headline):
        assert list(build_features(headline).columns) == FEATURE_NAMES

    def test_same_index_as_input(self, headline):
        assert build_features(headline).index.equals(headline.index)

    def test_leading_rows_are_nan_then_complete(self, headline):
        feats = build_features(headline)
        assert feats.iloc[:FIRST_FEATURE_ROW].isna().any(axis=1).all()
        assert not feats.iloc[FIRST_FEATURE_ROW:].isna().any(axis=None)

    def test_lag1_is_current_month_change(self, headline):
        feats = build_features(headline)
        expected = headline.pct_change() * 100
        assert feats["lag1"].iloc[20] == pytest.approx(expected.iloc[20])

    def test_month_encoding_is_on_unit_circle(self, headline):
        feats = build_features(headline)
        radius = feats["month_sin"] ** 2 + feats["month_cos"] ** 2
        assert np.allclose(radius, 1.0)

    def test_features_do_not_use_future_values(self, headline):
        """Changing the last value must not change any earlier feature row."""
        altered = headline.copy()
        altered.iloc[-1] = altered.iloc[-1] * 1.5
        a = build_features(headline).iloc[:-1]
        b = build_features(altered).iloc[:-1]
        pd.testing.assert_frame_equal(a, b)


class TestTargets:

    def test_target_equals_log_growth(self, headline):
        y = build_targets(headline, h=2)
        expected = 100 * np.log(headline.iloc[22] / headline.iloc[20])
        assert y.iloc[20] == pytest.approx(expected)

    def test_last_h_targets_are_nan(self, headline):
        y = build_targets(headline, h=3)
        assert y.iloc[-3:].isna().all()
        assert not y.iloc[:-3].isna().any()

    def test_training_set_has_no_nan_and_matching_lengths(self, headline):
        X, y = make_training_set(headline, h=3)
        assert len(X) == len(y)
        assert not X.isna().any(axis=None)
        assert not y.isna().any()
        assert len(X) == len(headline) - FIRST_FEATURE_ROW - 3


# Model fitting and tuning

class TestModels:

    def test_there_are_four_candidates(self):
        assert len(MODEL_NAMES) == 4
        assert set(MODEL_NAMES) == {"Lasso", "RandomForest", "GradientBoosting", "SVR"}

    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_tuning_returns_param_from_grid(self, name, headline):
        X, y = make_training_set(headline, h=1)
        params = tune_hyperparameters(name, X, y)
        assert set(params.keys()) == set(ml_models.PARAM_GRIDS[name].keys())
        for key, value in params.items():
            assert value in ml_models.PARAM_GRIDS[name][key]

    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_growth_prediction_is_finite_and_plausible(self, name, headline):
        growth, params = fit_predict_growth(name, headline, h=1)
        assert np.isfinite(growth)
        assert -5.0 < growth < 5.0       # monthly CPI growth in %

    def test_fixed_params_are_respected(self, headline):
        _, params = fit_predict_growth("SVR", headline, h=1,
                                       params={"C": 1.0, "epsilon": 0.1})
        assert params == {"C": 1.0, "epsilon": 0.1}

    def test_results_are_reproducible(self, headline):
        a, _ = fit_predict_growth("RandomForest", headline, h=2)
        b, _ = fit_predict_growth("RandomForest", headline, h=2)
        assert a == b

    def test_too_little_data_raises(self):
        short = _make_headline(n=20)
        with pytest.raises(ValueError):
            fit_predict_growth("SVR", short, h=1)

    def test_unknown_model_raises(self, headline):
        with pytest.raises(ValueError):
            fit_predict_growth("NotAModel", headline, h=1,
                               params={})


# Public forecast function

QUANTILES = {
    "1": {"lower": -0.5, "upper": 0.6},
    "2": {"lower": -1.0, "upper": 1.2},
    "3": {"lower": -1.5, "upper": 1.8},
}


class TestGenerateMLForecast:

    def test_required_keys(self, headline):
        out = generate_ml_forecast(headline, "SVR", error_quantiles=QUANTILES)
        required = {
            "model", "horizon", "confidence", "last_actual_date",
            "last_actual_cpi", "forecast_dates", "forecast_labels",
            "point", "lower", "upper",
            "annual_pct_point", "annual_pct_lower", "annual_pct_upper",
        }
        assert required.issubset(out.keys())

    def test_lengths_equal_horizon(self, headline):
        out = generate_ml_forecast(headline, "SVR", horizon=2,
                                   error_quantiles=QUANTILES)
        for key in ("forecast_dates", "forecast_labels", "point", "lower", "upper"):
            assert len(out[key]) == 2

    def test_dates_follow_last_observation(self, headline):
        out = generate_ml_forecast(headline, "SVR", error_quantiles=QUANTILES)
        expected = (headline.index[-1] + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
        assert out["forecast_dates"][0] == expected

    def test_interval_brackets_point_forecast(self, headline):
        out = generate_ml_forecast(headline, "GradientBoosting",
                                   error_quantiles=QUANTILES)
        for lo, pt, hi in zip(out["lower"], out["point"], out["upper"]):
            assert lo < pt < hi

    def test_interval_is_none_without_quantiles(self, headline):
        out = generate_ml_forecast(headline, "SVR")
        assert out["lower"] == [None] * FORECAST_HORIZON
        assert out["upper"] == [None] * FORECAST_HORIZON

    def test_point_forecasts_stay_near_last_value(self, headline):
        out = generate_ml_forecast(headline, "Lasso", error_quantiles=QUANTILES)
        last = out["last_actual_cpi"]
        for pt in out["point"]:
            assert abs(pt / last - 1) < 0.10

    def test_horizon_above_three_raises(self, headline):
        with pytest.raises(ValueError):
            generate_ml_forecast(headline, "SVR", horizon=4)

    def test_unknown_model_raises(self, headline):
        with pytest.raises(ValueError):
            generate_ml_forecast(headline, "Prophet")

    def test_short_series_raises(self):
        with pytest.raises(ValueError):
            generate_ml_forecast(_make_headline(n=25), "SVR")


# Walk-forward evaluation

class TestWalkForward:

    @pytest.fixture(scope="class")
    def results(self):
        h = _make_headline(n=56, seed=1)
        return h, model_selection.run_walk_forward(h, horizon=3, min_train=48)

    def test_row_count(self, results):
        h, df = results
        n_origins = len(h) - 3 - 48 + 1
        n_models = len(MODEL_NAMES) + 2          # + Naive-drift + OLS-trend
        assert len(df) == n_origins * 3 * n_models

    def test_all_models_present(self, results):
        _, df = results
        assert set(df["model"]) == set(MODEL_NAMES) | {"Naive-drift", "OLS-trend"}

    def test_errors_are_consistent(self, results):
        _, df = results
        assert (df["abs_pct_error"] >= 0).all()
        assert np.allclose(df["error"], df["actual"] - df["forecast"])

    def test_forecast_never_uses_future_data(self, results):
        """
        Altering the final observation must leave every forecast unchanged,
        because no training window ever includes it.
        """
        h, df = results
        altered = h.copy()
        altered.iloc[-1] = altered.iloc[-1] * 1.3
        df2 = model_selection.run_walk_forward(altered, horizon=3, min_train=48)
        key = ["origin", "h", "model"]
        a = df.sort_values(key)["forecast"].values
        b = df2.sort_values(key)["forecast"].values
        assert np.allclose(a, b)

    def test_too_short_series_raises(self):
        with pytest.raises(ValueError):
            model_selection.run_walk_forward(_make_headline(n=40), min_train=48)


# Statistics helpers

class TestStatistics:

    def test_dm_identical_errors_not_significant(self):
        e = np.random.default_rng(0).standard_normal(60)
        _, p = model_selection.diebold_mariano(e, e, h=1)
        assert p > 0.9

    def test_dm_detects_clearly_better_model(self):
        rng = np.random.default_rng(1)
        good = 0.2 * rng.standard_normal(80)
        bad = 1.0 * rng.standard_normal(80)
        stat, p = model_selection.diebold_mariano(good, bad, h=1)
        assert stat < 0
        assert p < 0.05

    def test_rank_models_prefers_lower_error(self):
        overall = pd.DataFrame(
            {"MAPE_%": [0.5, 0.6, 0.7, 0.8], "RMSE": [1.0, 1.1, 1.2, 1.3]},
            index=MODEL_NAMES,
        )
        periods = pd.DataFrame({
            "period": ["P1"] * 4,
            "model": MODEL_NAMES,
            "MAPE_%": [0.5, 0.6, 0.7, 0.8],
        }).set_index(["period", "model"])
        ranking = model_selection.rank_models(
            {"overall": overall, "by_period": periods}
        )
        assert ranking.index[0] == MODEL_NAMES[0]
        assert ranking["overall_rank"].tolist() == [1, 2, 3, 4]

    def test_append_step_forecasts_scores_external_model(self):
        h = _make_headline(n=52)
        base = pd.DataFrame(columns=["origin", "origin_date", "h", "target_date",
                                     "model", "actual", "forecast",
                                     "actual_growth", "pred_growth",
                                     "error", "abs_pct_error", "growth_error"])
        steps = [{
            "actuals": [float(v) for v in h.iloc[48:51]],
            "forecasts": [float(v) * 1.01 for v in h.iloc[48:51]],
        }]
        out = model_selection._append_step_forecasts(base, h, steps, "SARIMA", 48)
        assert len(out) == 3
        assert set(out["model"]) == {"SARIMA"}
        assert np.allclose(out["abs_pct_error"], 1.0)   # forecasts were 1% too high


# Saved selection record

class TestSelectionRecord:

    def test_record_is_consistent_if_present(self):
        record = model_selection.load_selection()
        if record is None:
            pytest.skip("Run `python -m forecasting.model_selection` first.")
        assert record["selected_model"] in MODEL_NAMES
        assert set(record["params_by_horizon"].keys()) == {"1", "2", "3"}
        assert set(record["error_quantiles_90"].keys()) == {"1", "2", "3"}
        ranked = [r["model"] for r in record["ranking"]]
        assert ranked[0] == record["selected_model"]
        for q in record["error_quantiles_90"].values():
            assert q["lower"] < q["upper"]
