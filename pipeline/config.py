"""
config.py
---------
Central configuration module for the Uganda CPI Intelligence System.

All hard-coded constants such as file paths, category definitions, expenditure
weights, and analytical parameters will be here so that the rest of the
codebase has a single source of truth.  When UBOS releases a new monthly
workbook the only file that will need editing is this one.

Author : Ainamani Dickson & Lumu Richard
Project: Uganda CPI Intelligence System (BSE Final Year Project, UTAMU)
Version: 1.0.0
"""

from pathlib import Path


# PROJECT PATHS
# Root of the repository (two levels up from this file: pipeline/ -> project/)
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# Raw data directory — the UBOS Excel workbook will be placed here
RAW_DATA_DIR: Path = ROOT_DIR / "data" / "raw"

# Processed CSV outputs written by the pipeline
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"

# Name of the current UBOS CPI Excel workbook
UBOS_FILENAME: str = "07_2026CPI_Excel_Tables_For_June_2026.xlsx"

# Full path to the source Excel file
UBOS_FILE_PATH: Path = RAW_DATA_DIR / UBOS_FILENAME

# Names of the output CSV files produced by the ingestion module
CSV_HEADLINE: str = "cpi_headline.csv"          # Headline (All Items) CPI series
CSV_CATEGORIES: str = "cpi_categories.csv"      # All 13 category CPI series
CSV_ANNUAL_PCT: str = "cpi_annual_pct.csv"      # Annual percentage changes
CSV_MONTHLY_PCT: str = "cpi_monthly_pct.csv"    # Month-on-month percentage changes
CSV_WEIGHTS: str = "cpi_weights.csv"            # Category weights (out of 1 000)



# UBOS WORKBOOK STRUCTURE

# These constants describe the layout of the UBOS "Division" sheet so that
# the ingestion module can locate data without relying on fragile cell
# addresses.

UBOS_SHEET_NAME: str = "Division "          # Note the trailing space in UBOS file
UBOS_HEADER_ROW: int = 0                    # Row index (0-based) containing dates
UBOS_FIRST_DATA_ROW: int = 1               # First category row index (0-based)
UBOS_LAST_DATA_ROW: int = 13               # Last category row index (inclusive, 0-based)
UBOS_GRAND_TOTAL_ROW: int = 14             # Headline / All-Items row index (0-based)
UBOS_CATEGORY_COL: int = 1                 # Column index for category names
UBOS_WEIGHT_COL: int = 2                   # Column index for national weights
UBOS_FIRST_VALUE_COL: int = 3             # Column index where monthly CPI values begin



# CATEGORY DEFINITIONS

# The 13 UBOS CPI expenditure categories exactly as they appear in the
# workbook.  Order matters — it matches the row order in the Excel sheet.

CATEGORIES: list[str] = [
    "Food and Non-Alcoholic Beverages",
    "Alcoholic Beverages, Tobacco and Narcotics",
    "Clothing and Footwear",
    "Housing, Water, Electricity, Gas and Other Fuels",
    "Furnishings, Household Equipment and Routine Household",
    "Health",
    "Transport",
    "Information and Communication",
    "Recreation, Sport and Culture",
    "Education Services",
    "Restaurants and Accommodation Services",
    "Insurance and Financial Services",
    "Personal Care, Social Protection and Miscellaneous Goods",
]

# Headline label used in the workbook Grand Total row
HEADLINE_LABEL: str = "All Items"



# EXPENDITURE WEIGHTS  (out of 1 000)

# National weights from the UBOS 2016/17 consumer basket, as published in
# the monthly CPI release.  These sum to 1 000.
# Source: Uganda Bureau of Statistics (2026). Consumer Price Index — June 2026.

CATEGORY_WEIGHTS: dict[str, float] = {
    "Food and Non-Alcoholic Beverages":                            270.5390004753476,
    "Alcoholic Beverages, Tobacco and Narcotics":                   38.795791342564634,
    "Clothing and Footwear":                                        69.77288896194038,
    "Housing, Water, Electricity, Gas and Other Fuels":            104.16172507951781,
    "Furnishings, Household Equipment and Routine Household":       48.36722367277888,
    "Health":                                                       47.46954703122525,
    "Transport":                                                   104.54790560057987,
    "Information and Communication":                                44.321641547949426,
    "Recreation, Sport and Culture":                                49.849448536995375,
    "Education Services":                                           57.955754225276586,
    "Restaurants and Accommodation Services":                       87.37912278394087,
    "Insurance and Financial Services":                             22.795280118963028,
    "Personal Care, Social Protection and Miscellaneous Goods":     54.04467062292019,
}

# Sanity-check: weights must sum to 1 000 (±0.5 to allow for rounding)
_weight_sum = sum(CATEGORY_WEIGHTS.values())
assert abs(_weight_sum - 1000.0) < 0.5, (
    f"Category weights sum to {_weight_sum:.1f}, expected 1 000. "
    "Check CATEGORY_WEIGHTS in config.py."
)



# ESSENTIAL vs DISCRETIONARY CLASSIFICATION

# Classification follows three converging authorities:
#   1. ILO COICOP framework (ILO, 2004) — six divisions cover basic human
#      needs: nutrition, clothing, shelter, health, learning, and mobility.
#   2. Uganda National Minimum Household Basket (UBOS UNHS 2019/20) —
#      these six are the primary expenditure heads households must fund
#      before any discretionary spending.
#   3. Empirical weight evidence — the six categories collectively account
#      for about 654.5 / 1 000 weight points, confirming their dominance in
#      the average Ugandan household's consumption pattern.

ESSENTIAL_CATEGORIES: list[str] = [
    "Food and Non-Alcoholic Beverages",
    "Clothing and Footwear",
    "Housing, Water, Electricity, Gas and Other Fuels",
    "Health",
    "Transport",
    "Education Services",
]

DISCRETIONARY_CATEGORIES: list[str] = [
    cat for cat in CATEGORIES if cat not in ESSENTIAL_CATEGORIES
]

# Pre-computed essential weight total (used in analysis module)
ESSENTIAL_WEIGHT_TOTAL: float = sum(
    CATEGORY_WEIGHTS[cat] for cat in ESSENTIAL_CATEGORIES
)  # Expected: about 654.5



# ANALYTICAL PARAMETERS


# Minimum number of months required before annual % change can be computed
MIN_MONTHS_FOR_ANNUAL: int = 13

# Number of standard deviations beyond which a month is flagged as anomalous
ANOMALY_Z_THRESHOLD: float = 2.0

# Number of months to use as the OLS training window for forecasting
OLS_TRAINING_WINDOW: int = 24

# Number of months ahead to forecast (3-month horizon for OLS accuracy)
FORECAST_HORIZON: int = 3

# Walk-forward validation: minimum training size before first validation step
WALK_FORWARD_MIN_TRAIN: int = 24

# CPI base period label (UBOS 2016/17 = 100)
BASE_PERIOD_LABEL: str = "2016/17"

# Dataset coverage
DATASET_START: str = "July 2017"
DATASET_END: str = "June 2026"
DATASET_MONTHS: int = 108



# API CONFIGURATION


API_VERSION: str = "v1"
API_PREFIX: str = f"/api/{API_VERSION}"
API_HOST: str = "0.0.0.0"
API_PORT: int = 5000
API_DEBUG: bool = False   # Set to False in production



# ENSURE OUTPUT DIRECTORY EXISTS
# 
# Called once at import time so downstream modules can write CSVs immediately.

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)