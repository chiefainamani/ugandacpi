"""Ingestion module for the Uganda CPI Intelligence System.

Responsibilities:
    1. Read the UBOS monthly CPI Excel workbook from data/raw/
    2. Validate the workbook structure and data integrity
    3. Extract the headline CPI series and all 13 category series
    4. Compute month-on-month and year-on-year percentage changes
    5. Write clean, analysis-ready CSV files to outputs/

This module intentionally contains no analytical logic — it only cleans
and structures the raw UBOS data.  All statistical computations live in
analysis.py.

Author : Ainamani Dickson & Lumu Richard
Project: Uganda CPI Intelligence System (BSE Final Year Project, UTAMU)
Version: 1.0.0
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

from pipeline.config import (
    UBOS_FILE_PATH,
    UBOS_SHEET_NAME,
    UBOS_HEADER_ROW,
    UBOS_FIRST_DATA_ROW,
    UBOS_LAST_DATA_ROW,
    UBOS_GRAND_TOTAL_ROW,
    UBOS_CATEGORY_COL,
    UBOS_WEIGHT_COL,
    UBOS_FIRST_VALUE_COL,
    CATEGORIES,
    CATEGORY_WEIGHTS,
    HEADLINE_LABEL,
    OUTPUTS_DIR,
    CSV_HEADLINE,
    CSV_CATEGORIES,
    CSV_ANNUAL_PCT,
    CSV_MONTHLY_PCT,
    CSV_WEIGHTS,
    MIN_MONTHS_FOR_ANNUAL,
    DATASET_MONTHS,
)

# Module-level logger — handlers are configured by the caller (or __main__)
logger = logging.getLogger(__name__)


# Raw extraction

def _load_raw_sheet(file_path: Path) -> pd.DataFrame:
    """
    Load the UBOS Division sheet as a raw DataFrame with no header parsing.

    Parameters
    ----------
    file_path : Path
        Absolute path to the UBOS Excel workbook.

    Returns
    -------
    pd.DataFrame
        Raw sheet contents with integer column indices.

    Raises
    ------
    FileNotFoundError
        If the workbook does not exist at the specified path.
    ValueError
        If the expected sheet name is not found in the workbook.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"UBOS workbook not found at: {file_path}\n"
            f"Place the Excel file in data/raw/ and update UBOS_FILENAME in config.py."
        )

    logger.info(f"Loading workbook: {file_path.name}")

    try:
        df = pd.read_excel(file_path, sheet_name=UBOS_SHEET_NAME, header=None)
    except Exception as exc:
        raise ValueError(
            f"Could not read sheet '{UBOS_SHEET_NAME}' from {file_path.name}. "
            f"Check UBOS_SHEET_NAME in config.py.\nOriginal error: {exc}"
        )

    logger.info(f"Raw sheet loaded — shape: {df.shape}")
    return df


# Validation

def _validate_structure(df: pd.DataFrame) -> None:
    """
    Validate that the raw sheet has the expected structure.

    Checks performed:
        - Minimum number of rows (at least GRAND_TOTAL_ROW + 1)
        - Minimum number of columns (header + weight + at least DATASET_MONTHS values)
        - Category count matches config expectation

    Parameters
    ----------
    df : pd.DataFrame
        Raw sheet DataFrame returned by _load_raw_sheet.

    Raises
    ------
    ValueError
        If any structural check fails.
    """
    min_rows = UBOS_GRAND_TOTAL_ROW + 1
    if df.shape[0] < min_rows:
        raise ValueError(
            f"Workbook has {df.shape[0]} rows; expected at least {min_rows}. "
            "The sheet structure may have changed — check UBOS_GRAND_TOTAL_ROW in config.py."
        )

    min_cols = UBOS_FIRST_VALUE_COL + DATASET_MONTHS
    if df.shape[1] < min_cols:
        raise ValueError(
            f"Workbook has {df.shape[1]} columns; expected at least {min_cols}. "
            f"Ensure the workbook covers {DATASET_MONTHS} monthly periods."
        )

    # Count non-null category names in the category column
    category_cells = df.iloc[
        UBOS_FIRST_DATA_ROW : UBOS_LAST_DATA_ROW + 1, UBOS_CATEGORY_COL
    ]
    actual_count = category_cells.notna().sum()
    expected_count = len(CATEGORIES)

    if actual_count != expected_count:
        raise ValueError(
            f"Found {actual_count} category rows; expected {expected_count}. "
            "Check UBOS_FIRST_DATA_ROW / UBOS_LAST_DATA_ROW in config.py."
        )

    logger.info("Structural validation passed.")


def _validate_weights(df: pd.DataFrame) -> None:
    """
    Validate that the workbook weights match the weights declared in config.py.

    A tolerance of 0.1 per category is allowed to account for rounding in
    the published workbook.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sheet DataFrame.

    Raises
    ------
    ValueError
        If any category weight differs from the config value by more than 0.1.
    """
    for i, category in enumerate(CATEGORIES):
        row_idx = UBOS_FIRST_DATA_ROW + i
        raw_weight = df.iloc[row_idx, UBOS_WEIGHT_COL]

        try:
            workbook_weight = float(raw_weight)
        except (TypeError, ValueError):
            raise ValueError(
                f"Non-numeric weight for '{category}' at row {row_idx}: {raw_weight!r}"
            )

        config_weight = CATEGORY_WEIGHTS[category]
        if abs(workbook_weight - config_weight) > 1.0:
            raise ValueError(
                f"Weight mismatch for '{category}': "
                f"workbook={workbook_weight}, config={config_weight}. "
                "Update CATEGORY_WEIGHTS in config.py if UBOS revised its basket."
            )

    logger.info("Weight validation passed.")


# Data extraction

def _extract_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Extract the monthly date index from the header row of the workbook.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sheet DataFrame.

    Returns
    -------
    pd.DatetimeIndex
        Sorted monthly period index covering all available CPI months.

    Raises
    ------
    ValueError
        If any date value in the header row cannot be parsed.
    """
    raw_dates = df.iloc[UBOS_HEADER_ROW, UBOS_FIRST_VALUE_COL:].tolist()

    try:
        dates = pd.to_datetime(raw_dates).to_period("M").to_timestamp()
    except Exception as exc:
        raise ValueError(
            f"Failed to parse dates from header row. "
            f"First few values: {raw_dates[:5]}\nOriginal error: {exc}"
        )

    logger.info(
        f"Date range extracted: {dates[0].strftime('%b %Y')} — {dates[-1].strftime('%b %Y')} "
        f"({len(dates)} months)"
    )
    return pd.DatetimeIndex(dates)


def _extract_series(df: pd.DataFrame, row_idx: int, dates: pd.DatetimeIndex) -> pd.Series:
    """
    Extract a single CPI value series from the specified workbook row.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sheet DataFrame.
    row_idx : int
        Zero-based row index of the series to extract.
    dates : pd.DatetimeIndex
        Date index to assign to the extracted series.

    Returns
    -------
    pd.Series
        Float series indexed by date, with NaN for any unparseable cells.
    """
    raw_values = df.iloc[row_idx, UBOS_FIRST_VALUE_COL:].tolist()
    series = pd.to_numeric(raw_values, errors="coerce")
    return pd.Series(series, index=dates, dtype=float)


def _extract_all_series(df: pd.DataFrame, dates: pd.DatetimeIndex) -> tuple[pd.Series, pd.DataFrame]:
    """
    Extract the headline CPI series and all 13 category series.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sheet DataFrame.
    dates : pd.DatetimeIndex
        Date index returned by _extract_dates.

    Returns
    -------
    headline : pd.Series
        Headline (All Items) CPI series.
    categories_df : pd.DataFrame
        DataFrame with one column per category, indexed by date.
    """
    # Headline series
    headline = _extract_series(df, UBOS_GRAND_TOTAL_ROW, dates)
    headline.name = HEADLINE_LABEL
    null_count = headline.isna().sum()
    if null_count > 0:
        logger.warning(f"Headline series has {null_count} null value(s).")

    # Category series
    category_dict: dict[str, pd.Series] = {}
    for i, category in enumerate(CATEGORIES):
        row_idx = UBOS_FIRST_DATA_ROW + i
        series = _extract_series(df, row_idx, dates)
        nulls = series.isna().sum()
        if nulls > 0:
            logger.warning(f"'{category}' has {nulls} null value(s).")
        category_dict[category] = series

    categories_df = pd.DataFrame(category_dict)

    logger.info(
        f"Extracted headline series and {len(CATEGORIES)} category series "
        f"({len(dates)} observations each)."
    )
    return headline, categories_df


# Percentage change computation

def _compute_monthly_pct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute month-on-month percentage change for all series.

    Formula: ((CPI_t / CPI_{t-1}) - 1) * 100

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of CPI values (headline + categories), indexed by date.

    Returns
    -------
    pd.DataFrame
        Month-on-month percentage changes; first row is NaN by definition.
    """
    return df.pct_change(periods=1) * 100


def _compute_annual_pct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute year-on-year percentage change for all series.

    Formula: ((CPI_t / CPI_{t-12}) - 1) * 100

    The first MIN_MONTHS_FOR_ANNUAL rows will be NaN because no 12-month
    prior observation exists.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of CPI values (headline + categories), indexed by date.

    Returns
    -------
    pd.DataFrame
        Annual percentage changes; first 12 rows are NaN by definition.
    """
    if len(df) < MIN_MONTHS_FOR_ANNUAL:
        logger.warning(
            f"Only {len(df)} observations available; annual % change requires "
            f"at least {MIN_MONTHS_FOR_ANNUAL}."
        )
    return df.pct_change(periods=12) * 100


# CSV output

def _save_csv(df: pd.DataFrame | pd.Series, filename: str) -> Path:
    """
    Save a DataFrame or Series to a CSV file in the outputs directory.

    Parameters
    ----------
    df : pd.DataFrame | pd.Series
        Data to persist.
    filename : str
        Output file name (e.g. "cpi_headline.csv").

    Returns
    -------
    Path
        Absolute path to the written file.
    """
    out_path = OUTPUTS_DIR / filename
    df.to_csv(out_path, index=True, index_label="date")
    logger.info(f"Saved: {out_path.relative_to(out_path.parent.parent)}")
    return out_path


def _save_weights() -> Path:
    """
    Persist the category weights dictionary from config.py as a CSV.

    Returns
    -------
    Path
        Absolute path to the written weights CSV.
    """
    weights_series = pd.Series(CATEGORY_WEIGHTS, name="weight")
    weights_series.index.name = "category"
    out_path = OUTPUTS_DIR / CSV_WEIGHTS
    weights_series.to_csv(out_path)
    logger.info(f"Saved: {out_path.name}")
    return out_path


# Public API

def run_ingestion() -> dict[str, pd.DataFrame | pd.Series]:
    """
    Execute the full ingestion pipeline.

    Steps:
        1. Load raw Excel sheet
        2. Validate structure and weights
        3. Extract dates, headline, and category series
        4. Compute monthly and annual percentage changes
        5. Write all outputs to CSV

    Returns
    -------
    dict with keys:
        "headline"      — pd.Series  : Headline CPI (All Items)
        "categories"    — pd.DataFrame : 13 category CPI series
        "monthly_pct"   — pd.DataFrame : Month-on-month % changes (headline + categories)
        "annual_pct"    — pd.DataFrame : Year-on-year % changes (headline + categories)

    Raises
    ------
    FileNotFoundError
        If the UBOS workbook is missing from data/raw/.
    ValueError
        If any validation check fails.
    """
    logger.info("=== Uganda CPI Intelligence — Ingestion Pipeline ===")

    # Step 1: Load
    raw_df = _load_raw_sheet(UBOS_FILE_PATH)

    # Step 2: Validate
    _validate_structure(raw_df)
    _validate_weights(raw_df)

    # Step 3: Extract
    dates = _extract_dates(raw_df)
    headline, categories_df = _extract_all_series(raw_df, dates)

    # Combine into one DataFrame for percentage change computation
    all_series_df = pd.concat([headline, categories_df], axis=1)

    # Step 4: Percentage changes
    monthly_pct_df = _compute_monthly_pct(all_series_df)
    annual_pct_df = _compute_annual_pct(all_series_df)

    # Step 5: Save CSVs
    _save_csv(headline, CSV_HEADLINE)
    _save_csv(categories_df, CSV_CATEGORIES)
    _save_csv(monthly_pct_df, CSV_MONTHLY_PCT)
    _save_csv(annual_pct_df, CSV_ANNUAL_PCT)
    _save_weights()

    logger.info("=== Ingestion complete. All outputs written to outputs/ ===")

    return {
        "headline": headline,
        "categories": categories_df,
        "monthly_pct": monthly_pct_df,
        "annual_pct": annual_pct_df,
    }


# Entry point for running the module directly

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    results = run_ingestion()

    print("\nSummary")
    print(f"  Headline observations : {len(results['headline'])}")
    print(f"  Categories            : {results['categories'].shape[1]}")
    print(f"  First month           : {results['headline'].index[0].strftime('%b %Y')}")
    print(f"  Last month            : {results['headline'].index[-1].strftime('%b %Y')}")
    print(f"  Latest headline CPI   : {results['headline'].iloc[-1]:.2f}")
    print(f"  Latest annual %       : {results['annual_pct'][results['annual_pct'].columns[0]].iloc[-1]:.2f}%")