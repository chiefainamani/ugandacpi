"""Unit tests for the Uganda CPI Intelligence System ingestion module.

Test coverage:
    - File loading (valid path, missing file, wrong sheet name)
    - Structural validation (correct shape, wrong rows, wrong columns)
    - Weight validation (matching weights, mismatched weights, non-numeric)
    - Date extraction (valid dates, unparseable dates)
    - Series extraction (normal values, null handling)
    - Percentage change computation (monthly and annual)
    - CSV output (files created, correct index)
    - run_ingestion() integration smoke test

Run with:
    uv run python -m pytest tests/test_ingestion.py -v

Author : Ainamani Dickson & Lumu Richard
Project: Uganda CPI Intelligence System (BSE Final Year Project, UTAMU)
Version: 1.0.0
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from pipeline import ingestion
from pipeline.config import (
    CATEGORIES,
    CATEGORY_WEIGHTS,
    UBOS_FIRST_DATA_ROW,
    UBOS_LAST_DATA_ROW,
    UBOS_GRAND_TOTAL_ROW,
    UBOS_CATEGORY_COL,
    UBOS_WEIGHT_COL,
    UBOS_FIRST_VALUE_COL,
    UBOS_HEADER_ROW,
    DATASET_MONTHS,
    MIN_MONTHS_FOR_ANNUAL,
    OUTPUTS_DIR,
    CSV_HEADLINE,
    CSV_CATEGORIES,
    CSV_ANNUAL_PCT,
    CSV_MONTHLY_PCT,
    CSV_WEIGHTS,
    HEADLINE_LABEL,
)


# Fixtures

def _make_valid_raw_df() -> pd.DataFrame:
    """
    Build a minimal but structurally valid raw DataFrame that mirrors the
    UBOS workbook layout used in the ingestion module.

    The DataFrame uses object dtype to hold mixed types (dates, strings,
    floats) in the same structure as the real UBOS Excel sheet, which
    pandas reads as object dtype before our ingestion module processes it.
    """
    n_months = DATASET_MONTHS
    n_rows = UBOS_GRAND_TOTAL_ROW + 1
    n_cols = UBOS_FIRST_VALUE_COL + n_months

    # Use object dtype — matches what pandas produces when reading a
    # mixed-content Excel sheet with no header parsing
    df = pd.DataFrame(
        index=range(n_rows),
        columns=range(n_cols),
        dtype=object,
    )

    # Header row: synthetic monthly dates starting July 2017
    dates = pd.date_range(start="2017-07-01", periods=n_months, freq="MS")
    for j, d in enumerate(dates):
        df.iloc[UBOS_HEADER_ROW, UBOS_FIRST_VALUE_COL + j] = d

    # Category rows: name, weight, and linearly increasing CPI values
    categories = list(CATEGORY_WEIGHTS.keys())
    weights = list(CATEGORY_WEIGHTS.values())

    for i, (cat, wgt) in enumerate(zip(categories, weights)):
        row = UBOS_FIRST_DATA_ROW + i
        df.iloc[row, UBOS_CATEGORY_COL] = cat
        df.iloc[row, UBOS_WEIGHT_COL] = wgt
        for j in range(n_months):
            df.iloc[row, UBOS_FIRST_VALUE_COL + j] = 100.0 + j * 0.2

    # Grand total row
    df.iloc[UBOS_GRAND_TOTAL_ROW, UBOS_CATEGORY_COL] = HEADLINE_LABEL
    df.iloc[UBOS_GRAND_TOTAL_ROW, UBOS_WEIGHT_COL] = 1000.0
    for j in range(n_months):
        df.iloc[UBOS_GRAND_TOTAL_ROW, UBOS_FIRST_VALUE_COL + j] = 102.0 + j * 0.35

    return df


@pytest.fixture
def valid_df() -> pd.DataFrame:
    """Fixture: structurally valid raw workbook DataFrame."""
    return _make_valid_raw_df()


@pytest.fixture
def valid_dates(valid_df) -> pd.DatetimeIndex:
    """Fixture: date index extracted from the valid DataFrame."""
    return ingestion._extract_dates(valid_df)


@pytest.fixture
def all_series_df(valid_df, valid_dates) -> pd.DataFrame:
    """Fixture: combined headline + category DataFrame for pct-change tests."""
    headline, categories_df = ingestion._extract_all_series(valid_df, valid_dates)
    return pd.concat([headline, categories_df], axis=1)


# Tests: _load_raw_sheet

class TestLoadRawSheet:

    def test_raises_file_not_found_for_missing_path(self, tmp_path):
        """Loading from a non-existent path must raise FileNotFoundError."""
        missing = tmp_path / "does_not_exist.xlsx"
        with pytest.raises(FileNotFoundError, match="UBOS workbook not found"):
            ingestion._load_raw_sheet(missing)

    def test_raises_value_error_for_wrong_sheet(self, tmp_path):
        """A workbook without the expected sheet name must raise ValueError."""
        # Create a minimal Excel file with a different sheet name
        wb_path = tmp_path / "test.xlsx"
        pd.DataFrame({"a": [1]}).to_excel(wb_path, sheet_name="WrongSheet", index=False)
        with pytest.raises(ValueError, match="Could not read sheet"):
            ingestion._load_raw_sheet(wb_path)

    def test_returns_dataframe_for_valid_file(self, tmp_path):
        """A valid file with the correct sheet name returns a DataFrame."""
        from pipeline.config import UBOS_SHEET_NAME
        wb_path = tmp_path / "test.xlsx"
        pd.DataFrame({"a": [1, 2]}).to_excel(wb_path, sheet_name=UBOS_SHEET_NAME, index=False)
        result = ingestion._load_raw_sheet(wb_path)
        assert isinstance(result, pd.DataFrame)


# Tests: _validate_structure

class TestValidateStructure:

    def test_passes_for_valid_dataframe(self, valid_df):
        """No exception should be raised for a structurally correct DataFrame."""
        ingestion._validate_structure(valid_df)  # must not raise

    def test_raises_for_too_few_rows(self, valid_df):
        """Truncating rows below the minimum must raise ValueError."""
        truncated = valid_df.iloc[:UBOS_GRAND_TOTAL_ROW, :]
        with pytest.raises(ValueError, match="rows"):
            ingestion._validate_structure(truncated)

    def test_raises_for_too_few_columns(self, valid_df):
        """Truncating columns below the minimum must raise ValueError."""
        truncated = valid_df.iloc[:, : UBOS_FIRST_VALUE_COL + 10]
        with pytest.raises(ValueError, match="columns"):
            ingestion._validate_structure(truncated)

    def test_raises_for_missing_category_names(self, valid_df):
        """Nulling out category name cells must trigger the category count check."""
        bad_df = valid_df.copy()
        # Remove names from first 3 category rows
        for i in range(3):
            bad_df.iloc[UBOS_FIRST_DATA_ROW + i, UBOS_CATEGORY_COL] = np.nan
        with pytest.raises(ValueError, match="category rows"):
            ingestion._validate_structure(bad_df)


# Tests: _validate_weights

class TestValidateWeights:

    def test_passes_for_correct_weights(self, valid_df):
        """No exception for weights that match config (within tolerance)."""
        ingestion._validate_weights(valid_df)  # must not raise

    def test_raises_for_mismatched_weight(self, valid_df):
        """A weight that deviates by more than 1.0 must raise ValueError."""
        bad_df = valid_df.copy()
        # Corrupt the first category weight substantially
        bad_df.iloc[UBOS_FIRST_DATA_ROW, UBOS_WEIGHT_COL] = 999.0
        with pytest.raises(ValueError, match="Weight mismatch"):
            ingestion._validate_weights(bad_df)

    def test_raises_for_non_numeric_weight(self, valid_df):
        """A non-numeric weight cell must raise ValueError."""
        bad_df = valid_df.copy()
        bad_df.iloc[UBOS_FIRST_DATA_ROW, UBOS_WEIGHT_COL] = "N/A"
        with pytest.raises(ValueError, match="Non-numeric weight"):
            ingestion._validate_weights(bad_df)

    def test_tolerates_small_rounding_difference(self, valid_df):
        """A weight within ±1.0 of the config value must pass validation."""
        ok_df = valid_df.copy()
        first_cat = list(CATEGORY_WEIGHTS.keys())[0]
        ok_df.iloc[UBOS_FIRST_DATA_ROW, UBOS_WEIGHT_COL] = CATEGORY_WEIGHTS[first_cat] + 0.9
        ingestion._validate_weights(ok_df)  # must not raise


# Tests: _extract_dates

class TestExtractDates:

    def test_returns_correct_length(self, valid_df):
        """Extracted date index length must equal DATASET_MONTHS."""
        dates = ingestion._extract_dates(valid_df)
        assert len(dates) == DATASET_MONTHS

    def test_first_date_is_july_2017(self, valid_df):
        """First date must be July 2017 (UBOS series start)."""
        dates = ingestion._extract_dates(valid_df)
        assert dates[0].year == 2017
        assert dates[0].month == 7

    def test_returns_datetime_index(self, valid_df):
        """Return type must be a DatetimeIndex."""
        dates = ingestion._extract_dates(valid_df)
        assert isinstance(dates, pd.DatetimeIndex)

    def test_raises_for_unparseable_dates(self, valid_df):
        """Non-date values in the header row must raise ValueError."""
        bad_df = valid_df.copy()
        bad_df.iloc[UBOS_HEADER_ROW, UBOS_FIRST_VALUE_COL] = "not-a-date"
        bad_df.iloc[UBOS_HEADER_ROW, UBOS_FIRST_VALUE_COL + 1] = "also-bad"
        with pytest.raises(ValueError, match="parse dates"):
            ingestion._extract_dates(bad_df)


# Tests: _extract_series

class TestExtractSeries:

    def test_returns_series_of_correct_length(self, valid_df, valid_dates):
        """Extracted series length must match the date index."""
        series = ingestion._extract_series(valid_df, UBOS_GRAND_TOTAL_ROW, valid_dates)
        assert len(series) == len(valid_dates)

    def test_returns_float_dtype(self, valid_df, valid_dates):
        """Extracted series must have float dtype."""
        series = ingestion._extract_series(valid_df, UBOS_GRAND_TOTAL_ROW, valid_dates)
        assert series.dtype == float

    def test_non_numeric_cells_become_nan(self, valid_df, valid_dates):
        """Non-numeric cells must be coerced to NaN rather than raising."""
        bad_df = valid_df.copy()
        bad_df.iloc[UBOS_GRAND_TOTAL_ROW, UBOS_FIRST_VALUE_COL] = "N/A"
        series = ingestion._extract_series(bad_df, UBOS_GRAND_TOTAL_ROW, valid_dates)
        assert np.isnan(series.iloc[0])

    def test_index_matches_dates(self, valid_df, valid_dates):
        """Extracted series index must match the provided DatetimeIndex."""
        series = ingestion._extract_series(valid_df, UBOS_GRAND_TOTAL_ROW, valid_dates)
        assert series.index.equals(valid_dates)


# Tests: _extract_all_series

class TestExtractAllSeries:

    def test_headline_is_series(self, valid_df, valid_dates):
        """Headline must be returned as a pd.Series."""
        headline, _ = ingestion._extract_all_series(valid_df, valid_dates)
        assert isinstance(headline, pd.Series)

    def test_categories_df_has_correct_columns(self, valid_df, valid_dates):
        """Categories DataFrame must have exactly len(CATEGORIES) columns."""
        _, categories_df = ingestion._extract_all_series(valid_df, valid_dates)
        assert categories_df.shape[1] == len(CATEGORIES)

    def test_headline_name_is_all_items(self, valid_df, valid_dates):
        """Headline series name must equal HEADLINE_LABEL from config."""
        headline, _ = ingestion._extract_all_series(valid_df, valid_dates)
        assert headline.name == HEADLINE_LABEL

    def test_no_unexpected_nulls_in_synthetic_data(self, valid_df, valid_dates):
        """Synthetic data has no nulls — neither output should contain any."""
        headline, categories_df = ingestion._extract_all_series(valid_df, valid_dates)
        assert headline.isna().sum() == 0
        assert categories_df.isna().sum().sum() == 0


# Tests: _compute_monthly_pct

class TestComputeMonthlyPct:

    def test_first_row_is_nan(self, all_series_df):
        """First row of monthly % change must always be NaN (no prior period)."""
        result = ingestion._compute_monthly_pct(all_series_df)
        assert result.iloc[0].isna().all()

    def test_remaining_rows_are_finite(self, all_series_df):
        """All rows after the first must contain finite (non-NaN) values."""
        result = ingestion._compute_monthly_pct(all_series_df)
        assert result.iloc[1:].notna().all().all()

    def test_correct_output_shape(self, all_series_df):
        """Output shape must match input shape."""
        result = ingestion._compute_monthly_pct(all_series_df)
        assert result.shape == all_series_df.shape

    def test_known_percentage_value(self):
        """Manually verify: 110 / 100 - 1 = 10%."""
        df = pd.DataFrame({"x": [100.0, 110.0]})
        result = ingestion._compute_monthly_pct(df)
        assert abs(result["x"].iloc[1] - 10.0) < 1e-9


# Tests: _compute_annual_pct

class TestComputeAnnualPct:

    def test_first_twelve_rows_are_nan(self, all_series_df):
        """First 12 rows must be NaN — no 12-month prior observation exists."""
        result = ingestion._compute_annual_pct(all_series_df)
        assert result.iloc[:12].isna().all().all()

    def test_rows_from_13_onwards_are_finite(self, all_series_df):
        """Rows from index 12 onward must contain finite values."""
        result = ingestion._compute_annual_pct(all_series_df)
        assert result.iloc[12:].notna().all().all()

    def test_correct_output_shape(self, all_series_df):
        """Output shape must match input shape."""
        result = ingestion._compute_annual_pct(all_series_df)
        assert result.shape == all_series_df.shape

    def test_known_annual_value(self):
        """Manually verify: 13 months, last / first[0] - 1 = known rate."""
        values = [100.0] * 12 + [110.0]
        df = pd.DataFrame({"x": values})
        result = ingestion._compute_annual_pct(df)
        assert abs(result["x"].iloc[12] - 10.0) < 1e-9


# Tests: _save_csv and _save_weights

class TestSaveCsv:

    def test_csv_file_is_created(self, tmp_path, monkeypatch):
        """_save_csv must create a file at the expected path."""
        monkeypatch.setattr(ingestion, "OUTPUTS_DIR", tmp_path)
        df = pd.DataFrame({"a": [1, 2, 3]})
        ingestion._save_csv(df, "test_output.csv")
        assert (tmp_path / "test_output.csv").exists()

    def test_csv_has_date_index_label(self, tmp_path, monkeypatch):
        """The index column in the saved CSV must be labelled 'date'."""
        monkeypatch.setattr(ingestion, "OUTPUTS_DIR", tmp_path)
        series = pd.Series([1.0, 2.0], name="val",
                           index=pd.date_range("2017-07-01", periods=2, freq="MS"))
        ingestion._save_csv(series, "test_series.csv")
        loaded = pd.read_csv(tmp_path / "test_series.csv")
        assert "date" in loaded.columns

    def test_save_weights_creates_file(self, tmp_path, monkeypatch):
        """_save_weights must create cpi_weights.csv in OUTPUTS_DIR."""
        monkeypatch.setattr(ingestion, "OUTPUTS_DIR", tmp_path)
        ingestion._save_weights()
        assert (tmp_path / CSV_WEIGHTS).exists()

    def test_save_weights_has_correct_row_count(self, tmp_path, monkeypatch):
        """Weights CSV must have exactly len(CATEGORIES) rows."""
        monkeypatch.setattr(ingestion, "OUTPUTS_DIR", tmp_path)
        ingestion._save_weights()
        loaded = pd.read_csv(tmp_path / CSV_WEIGHTS)
        assert len(loaded) == len(CATEGORIES)


# Tests: run_ingestion (integration smoke test)

class TestRunIngestion:

    def test_returns_all_expected_keys(self):
        """run_ingestion must return a dict with all four expected keys."""
        result = ingestion.run_ingestion()
        assert set(result.keys()) == {"headline", "categories", "monthly_pct", "annual_pct"}

    def test_headline_length_equals_dataset_months(self):
        """Headline series must have exactly DATASET_MONTHS observations."""
        result = ingestion.run_ingestion()
        assert len(result["headline"]) == DATASET_MONTHS

    def test_categories_has_correct_column_count(self):
        """Categories DataFrame must have len(CATEGORIES) columns."""
        result = ingestion.run_ingestion()
        assert result["categories"].shape[1] == len(CATEGORIES)

    def test_all_output_csvs_exist(self):
        """All five output CSV files must exist after run_ingestion."""
        ingestion.run_ingestion()
        for filename in [CSV_HEADLINE, CSV_CATEGORIES, CSV_ANNUAL_PCT, CSV_MONTHLY_PCT, CSV_WEIGHTS]:
            assert (OUTPUTS_DIR / filename).exists(), f"Missing: {filename}"

    def test_latest_headline_cpi_is_plausible(self):
        """Latest CPI value must be in a plausible range (100–200)."""
        result = ingestion.run_ingestion()
        latest = result["headline"].iloc[-1]
        assert 100.0 < latest < 200.0, f"Implausible CPI value: {latest}"

    def test_annual_pct_first_12_rows_all_nan(self):
        """First 12 rows of annual_pct must be NaN (no prior year data)."""
        result = ingestion.run_ingestion()
        assert result["annual_pct"].iloc[:12].isna().all().all()