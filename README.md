# Uganda CPI Intelligence System

A system for Consumer Price Index analysis and forecasting,
built on  Uganda Bureau of Statistics (UBOS) monthly CPI data.

**Authors:** Ainamani Dickson · Lumu Richard 


## What the system does

The Uganda CPI Intelligence System transforms UBOS raw Excel workbooks into an
interactive, API-driven web dashboard. 

**Five analytical sections:**

| Section | What it shows |
|---|---|
| Overview | Headline CPI trend, annual inflation rate, month-on-month volatility, anomaly detection |
| Inflation Drivers | Weighted Laspeyres contribution ranking, cumulative growth by category |
| Household Calculator | UGX cost burden for a user-specified basket across any two months in the dataset |
| Essential vs Discretionary | Welch t-test comparison of essential vs discretionary category inflation |
| 3-Month Forecast | Machine-learning forecast (best of 4 models, chosen by walk-forward validation) with 90% prediction intervals; OLS kept as benchmark |

---

## Project structure

```
uganda_cpi_intelligence/
│
├── data/
│   └── raw/                          ← Place UBOS Excel workbook here
│
├── pipeline/
│   ├── config.py                     ← Central configuration (paths, weights, parameters)
│   ├── ingestion.py                  ← UBOS Excel parsing, validation, CSV output
│   └── analysis.py                   ← Statistical analysis (contributions, burden, t-test, anomalies)
│
├── forecasting/
│   ├── models.py                     ← OLS benchmark + walk-forward cross-validation
│   ├── ml_models.py                  ← features, 4 ML candidates (Lasso, RF, GB, SVR), ML forecast
│   └── model_selection.py            ← walk-forward comparison, ranking, DM tests, evidence files
│
├── api/
│   └── app.py                        ← Flask REST API (13 endpoints)
│
├── dashboard/
│   └── index.html                    ← Interactive HTML/JS dashboard (Chart.js)
│
├── tests/
│   ├── test_ingestion.py             ← 41 unit tests
│   ├── test_analysis.py              ← 49 unit tests
│   ├── test_forecasting.py           ← 32 unit tests (OLS benchmark)
│   ├── test_ml_forecasting.py        ← 41 unit tests (ML features, models, selection)
│   └── test_api.py                   ← 74 integration tests
│
├── outputs/                          ← Processed CSVs written by the pipeline
├── pyproject.toml
└── README.md
```

---

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (package manager)
- Git

---

## Setup

**1. Clone the repository:**
```bash
git clone https://github.com/chiefainamani/ugandacpi
cd ugandacpi
```

**2. Install dependencies:**
```bash
uv sync
```

**3. Add the UBOS data file:**

Copy the UBOS monthly CPI Excel workbook into `data/raw/`:
```
data/raw/07_2026CPI_Excel_Tables_For_June_2026.xlsx
```

If you use a different file, update `UBOS_FILENAME` in `pipeline/config.py`.

---

## Running the system

**Start the API server:**
```bash
uv run python -m api.app
```

The server starts on `http://localhost:5000`. You should see:
```
Running ingestion pipeline at startup...
Pipeline complete. API ready.
Starting Uganda CPI Intelligence API on 0.0.0.0:5000
```

**Open the dashboard:**

Open `dashboard/index.html` directly in your browser (double-click in File Explorer,
or use VS Code Live Server). The green "API connected" status confirms the
dashboard is communicating with the server.

---

## API endpoints

All endpoints are `GET` and return JSON. Base URL: `http://localhost:5000/api/v1`

| Endpoint | Description |
|---|---|
| `GET /` | API metadata and endpoint listing |
| `GET /summary` | Descriptive statistics (cumulative growth, peak inflation, etc.) |
| `GET /headline` | Full headline CPI series with monthly and annual % changes |
| `GET /categories` | All 13 category CPI series |
| `GET /categories/<name>` | Single category CPI, annual % change, and weight |
| `GET /weights` | National expenditure weights (out of 1,000) |
| `GET /drivers?period=` | Weighted contribution ranking for a given month |
| `GET /growth?base=` | Cumulative % growth from a configurable base period |
| `GET /burden?basket=&base=&target=` | UGX household cost burden for a configurable period |
| `GET /essential-vs-discretionary` | Welch t-test comparison of essential vs discretionary inflation |
| `GET /anomalies?threshold=` | Anomalous months by z-score |
| `GET /forecast?horizon=` | Forecast from the selected ML model (horizon 1–3) with 90% prediction intervals |
| `GET /forecast/validation` | Walk-forward metrics of the selected ML model |
| `GET /forecast/models` | Model-selection evidence: 4 ML models + baselines, ranking, winner |
| `GET /forecast/ols?horizon=&window=&confidence=` | OLS benchmark forecast |
| `GET /forecast/ols/validation` | OLS benchmark validation metrics |

**Example requests:**
```bash
# Household cost burden: UGX 500,000 basket, Jan 2020 to Jun 2026
curl "http://localhost:5000/api/v1/burden?basket=500000&base=2020-01-01&target=2026-06-01"

# Inflation drivers for October 2022 (peak inflation month)
curl "http://localhost:5000/api/v1/drivers?period=2022-10-01"

# Anomalies with a tighter threshold
curl "http://localhost:5000/api/v1/anomalies?threshold=1.5"
```

---

## Running the tests

```bash
# All 196 tests
uv run python -m pytest tests/ -v

# Individual test modules
uv run python -m pytest tests/test_ingestion.py -v    # 41 tests
uv run python -m pytest tests/test_analysis.py -v     # 49 tests
uv run python -m pytest tests/test_forecasting.py -v  # 32 tests
uv run python -m pytest tests/test_api.py -v          # 74 tests
```

Expected result: **196 passed**.

---

## Running individual modules

Each pipeline module can be run directly for manual verification:

```bash
# Ingestion — reads UBOS Excel, writes 5 CSVs to outputs/
uv run python -m pipeline.ingestion

# Analysis — runs all 5 analytical functions on real data
uv run python -m pipeline.analysis

# Model selection — trains/evaluates the 4 ML models and baselines, saves the winner
# (writes tables, figures and selection.json to outputs/model_selection/)
uv run python -m forecasting.model_selection
# add --with-sarima to include the SARIMA benchmark on the same origins (slow)
```

---

## Key figures (June 2026 data)

| Metric | Value |
|---|---|
| Dataset coverage | July 2017 – June 2026 (108 months) |
| Categories | 13 UBOS expenditure categories |
| Baseline CPI (Jul 2017) | 102.28 |
| Latest CPI (Jun 2026) | 141.97 |
| Cumulative growth | 38.8% |
| Peak annual inflation | 10.71% (October 2022) |
| Latest annual inflation | 3.70% (June 2026) |
| Essential category weight | 654.5 / 1,000 |
| Forecast model | Best of 4 ML models (see outputs/model_selection/selection.json) |
| Forecast MAPE (walk-forward) | see outputs/model_selection/metrics_overall.csv (below 2% target) |
| Total automated tests | 196 (all passing) |

---

## Technology stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data processing | pandas, NumPy |
| Statistical analysis | SciPy (Welch t-test, OLS via linregress) |
| Excel parsing | openpyxl |
| Web framework | Flask 3.0 |
| CORS | flask-cors |
| Frontend charting | Chart.js 4.4 |
| Frontend language | Vanilla JavaScript (ES6+) |
| Testing | pytest |
| Package management | uv |
| Version control | Git |

---

## Methodology notes

**Essential vs Discretionary classification** follows three converging authorities:
the ILO COICOP framework (ILO, 2004), Uganda's National Minimum Household Basket
(UBOS National Household Survey 2019/20), and empirical weight evidence —
the six essential categories (Food, Clothing and Footwear, Housing, Health, Transport,
Education) account for 654.5 / 1,000 weight points in the UBOS basket.

**Forecasting model choice:** four machine-learning models (Lasso, Random Forest,
Gradient Boosting, SVR) are trained to predict CPI *growth* over 1, 2 and 3 months
(direct multi-step) from lagged monthly changes, rolling statistics, the annual rate
and the calendar month. Hyper-parameters are tuned with forward-only cross-validation
using only data available at each forecast origin. All models, plus Naive-drift and
OLS benchmarks, are scored by the same walk-forward procedure (MAPE, RMSE, MAE, by
horizon and economic period, with Diebold-Mariano tests). The winner is chosen by
average rank across overall MAPE, RMSE and worst-period MAPE, and saved to
`outputs/model_selection/selection.json`, which the API reads at startup.

**Research framework:** Design Science Research (Hevner et al., 2004).  
**Development methodology:** Agile Scrum, 3 sprints (August – September 2026).

---

## Updating the data

When UBOS releases a new monthly workbook:

1. Place the new `.xlsx` file in `data/raw/`
2. Update `UBOS_FILENAME` in `pipeline/config.py`
3. Verify the category names and weights still match — update `CATEGORIES` and
   `CATEGORY_WEIGHTS` in `config.py` if UBOS revised the basket
4. Update `DATASET_END` and `DATASET_MONTHS` in `config.py`
5. Restart the API server — the pipeline runs automatically on startup

---

## License

This project was developed as a final year academic submission at UTAMU.
Data sourced from Uganda Bureau of Statistics (UBOS) — publicly available at
[https://www.ubos.org](https://www.ubos.org).
