# 🔌 New York City Electricity Consumption Forecasting

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Prophet](https://img.shields.io/badge/Facebook%20Prophet-Time%20Series-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Metrics-red?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![University](https://img.shields.io/badge/Pace%20University-CS675-blue)

> A time series forecasting project using **Facebook's Prophet** model to predict NYC electricity consumption across all five boroughs. Originally built for CS675: Introduction to Data Science at Pace University, and refreshed with NYC Open Data through October 2025.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Data Pipeline](#-data-pipeline)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Evaluation Approach](#-evaluation-approach)
- [Models & Results](#-models--results)
- [Borough-Level Forecasting](#-borough-level-forecasting)
- [Limitations & Next Steps](#-limitations--next-steps)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Findings](#-key-findings)
- [Technologies Used](#-technologies-used)
- [Author](#-author)

---

## 🗽 Overview

This project forecasts **New York City's daily, monthly, and yearly electricity consumption** using Facebook Prophet's additive time series model. The dataset spans **December 2009 to October 2025** and covers all five NYC boroughs: Brooklyn, Manhattan, Bronx, Queens, and Staten Island.

The notebook covers:
- ✅ Automatic time-unit detection (daily / monthly / yearly)
- ✅ Forecasting with three growth models: **Linear**, **Logistic**, and **Flat**
- ✅ Hyperparameter tuning via custom seasonality and trend changepoints
- ✅ Model evaluation using **MAE**, **MAPE**, and **R²**
- ✅ Independent borough-level forecasting (extra credit)

---

## 📊 Dataset

**Source:** [NYC Open Data — Electric Consumption And Cost (2010 – Sep 2025)](https://data.cityofnewyork.us/Housing-Development/Electric-Consumption-And-Cost-2010-Feb2023-/jr24-e7cr/about_data)

| Property | Details |
|---|---|
| Rows | 553,666 |
| Columns | 27 |
| Date Range | 2009-12-17 to 2025-10-13 |
| Boroughs | Brooklyn, Manhattan, Bronx, Queens, Staten Island |
| Target Variable | `Consumption (KWH)` |
| Granularity | Billing records (expanded to daily) |

The raw dataset contains billing records per meter, each covering a variable-length service period.

> The dataset is periodically re-published under an updated title; the link above resolves by its permanent ID (`jr24-e7cr`) regardless of the current name.

---

## 🔄 Data Pipeline

| Stage | Records | Operation |
|---|---|---|
| Raw billing records | 553,666 | As published by NYC Open Data |
| After cleaning | 362,215 | Removed zero-consumption and negative-day rows |
| After borough filter | 357,684 | Restricted to the 5 valid NYC boroughs |
| **Expanded to daily** | **10,860,703** | Each billing row split into one record per day (`KWH / days`) |
| Aggregated series | 5,095 daily · 191 monthly · 17 yearly | Summed by date, then resampled to monthly and yearly **means** |

Borough distribution of the cleaned billing records: Brooklyn 140,819 · Manhattan 93,128 · Bronx 83,216 · Queens 37,526 · Staten Island 2,995.

### 📏 A note on units

The daily series is the **total** KWH consumed per day. The monthly and yearly series are `resample().mean()` of that daily series, so they are **mean daily consumption** within each month or year, *not* monthly or yearly totals.

This is why MAE is directly comparable across all three tables below — every figure is on the same mean-daily-KWH scale. A monthly MAE of ~314K should be read as "the model misses average daily consumption by 314K KWH," not as a monthly-scale error.

> ⚠️ **Important:** Daily values are *interpolated*, not observed. Every day within a billing period is assigned the same `KWH ÷ days` value, so the daily series is considerably smoother than true daily meter reads would be. This inflates goodness-of-fit statistics on the daily dataset. See [Limitations](#-limitations--next-steps).

> ⚠️ **Calendar coverage is incomplete.** The 5,095 daily records span a 5,780-day window, so **685 days (11.9%) have no data at all**. Prophet fits straight across these gaps without flagging them.

---

## 📁 Project Structure
```
nyc-electricity-forecasting-prophet/
│
├── NYC_Electricity_Forecasting.ipynb   # Main Jupyter notebook
├── README.md                           # Project documentation
└── requirements.txt                    # Python dependencies
```

---

## 🔬 Methodology

### Step 1: Data Preprocessing
- Parsed billing records with variable-length service periods
- Removed invalid rows (zero consumption, negative days)
- Filtered to 5 valid NYC boroughs only
- Expanded each billing row into individual daily records (`KWH / days`)
- Aggregated to daily totals, then resampled to monthly and yearly means

### Step 2: Time-Unit Auto Detection
A `detect_frequency()` function identifies the granularity of any input dataset by measuring the median gap between consecutive dates:
```python
def detect_frequency(df):
    df_sorted = df.sort_values("ds")
    diffs = df_sorted["ds"].diff().dropna()
    median_diff = diffs.median().days

    if median_diff <= 1:
        return "daily"
    elif 25 <= median_diff <= 35:
        return "monthly"
    elif 360 <= median_diff <= 370:
        return "yearly"
    else:
        return "unknown"
```

### Step 3: Forecasting Horizons

| Dataset | Forecast Horizons |
|---|---|
| Daily | 100 days, 200 days, 365 days |
| Monthly | 1 month, 6 months, 9 months |
| Yearly | 1 year, 10 years, 20 years |

### Step 4: Model Configurations

| Parameter | Values Used |
|---|---|
| Growth | `linear`, `logistic`, `flat` |
| Seasonality period | 7, 365.25 days |
| Fourier order | 3, 5, 8, 10 |
| n_changepoints | 5, 10, 15, 25, 30 |
| changepoint_prior_scale | 0.1, 0.3, 0.5 |

> This is **not a grid search.** The values above are the union of parameters across five hand-picked configurations (tuned daily, alternative daily, tuned monthly, tuned yearly, and borough-level), each of which varies several parameters simultaneously. A proper sweep holding one axis constant at a time is listed in [Next Steps](#-limitations--next-steps).

---

## 📐 Evaluation Approach

**All metrics reported below are in-sample.** Each model is fitted on the full history, and predictions are then compared against the same historical dates used for fitting:

```python
model.fit(daily_agg)
forecast = model.predict(future)
merged = daily_agg.merge(forecast[["ds", "yhat"]], on="ds", how="inner")
evaluate_model(merged["y"], merged["yhat"], ...)
```

The inner join retains only dates the model was trained on, so the reported R² and MAE measure **how well each model fits observed history, not how accurately it forecasts unseen data**. Results should be read as fit quality and as a relative comparison between configurations, not as forecast accuracy.

This matters most for the tuning results. The tuned daily configuration raises flexibility on three axes at once (Fourier order 10, 30 changepoints, changepoint prior scale 0.5), and a more flexible model fits its training data better almost by construction. Without a holdout, the improvement from R² 0.631 to 0.722 cannot be separated from overfitting. Adding `Prophet.cross_validation()` is the first item in [Next Steps](#-limitations--next-steps).

### Metrics
- **MAE**, Mean Absolute Error, primary metric
- **R²**, Coefficient of Determination via scikit-learn
- **MAPE**, reported but **not usable on this dataset**. Near-zero consumption values in the earliest records drive percentage errors into the thousands — the first daily totals in December 2009 are under 1 KWH. See the borough table below.

---

## 📈 Models & Results

*All figures are in-sample fit metrics on the mean-daily-KWH scale. See [Evaluation Approach](#-evaluation-approach) and [A note on units](#-a-note-on-units).*

### Daily Dataset

| Model | MAE (KWH) | R² |
|---|---|---|
| Linear Growth | 289,490 | 0.631 |
| Logistic Growth | 308,038 | 0.537 |
| Flat Growth | 331,815 | 0.477 |
| **Custom Seasonality (tuned)** | **254,120** | **0.722** ✅ |
| Alternative Seasonality | 310,680 | 0.533 |

### Monthly Dataset

| Model | MAE (KWH) | R² |
|---|---|---|
| Linear Growth | 313,946 | 0.486 |
| Logistic Growth | 314,493 | 0.479 |
| Flat Growth | 343,453 | 0.460 |
| **Custom Seasonality (tuned)** | 322,904 | **0.517** ✅ |

Note that on the monthly series, tuning improves R² but *worsens* MAE, so no configuration dominates.

### Yearly Dataset

| Model | MAE (KWH) | MAPE | R² |
|---|---|---|---|
| Linear Growth | 592,484 | 31.93% | 0.040 |
| Logistic Growth | 602,763 | 33.25% | -0.032 |
| Flat Growth | 577,090 | 32.33% | -0.000 |
| Changepoint Tuned | 592,473 | 31.93% | 0.040 |

> Near-zero and negative R² on yearly data is expected. There are only 17 annual observations, 2009 covers just two weeks of December, and 2011 shows an anomalous drop (1.20M vs. 3.43M in 2010) that likely reflects incomplete reporting rather than a real consumption collapse.

---

## 🗺️ Borough-Level Forecasting

Independent Prophet models trained per borough on a **dedicated borough configuration**, distinct from the tuned citywide daily model:

| Parameter | Tuned daily (citywide) | Borough models |
|---|---|---|
| `n_changepoints` | 30 | 25 |
| `changepoint_prior_scale` | 0.5 | 0.3 |
| yearly `fourier_order` | 10 | 8 |
| weekly `fourier_order` | 5 | 3 |
| Forecast horizon | 365 days | 365 days |

| Borough | Daily Records | MAE (KWH) | R² | MAPE |
|---|---|---|---|---|
| Brooklyn | 5,027 | 83,628 | 0.670 | 6,995% |
| Manhattan | 5,080 | 91,102 | 0.700 | 362,780% |
| **Bronx** | 5,028 | **66,002** | **0.737** ✅ | 26,506% |
| Queens | 5,078 | 30,511 | 0.696 | 11,657% |
| Staten Island | 5,027 | 9,488 | 0.717 | 4,833% |

The MAPE column is included for transparency and should be disregarded. Those values are an artifact of near-zero denominators in early billing records, not a reflection of model quality. MAE and R² are the meaningful comparisons.

**Key observations:**
- 🏆 **Bronx** achieves the highest R² at 0.7367
- 🎯 **Staten Island** has the lowest MAE, consistent with its smaller footprint
- 📈 **Brooklyn** has the highest absolute consumption overall
- ☀️ All 5 boroughs show clear **summer consumption peaks**

> **How to read the borough MAE figures.** Borough MAE is much lower than the citywide 254,120, but this is arithmetic rather than evidence of a better model — each borough is a fraction of total consumption, so a proportionally smaller error is what a model of *identical* quality would produce. The scale-free comparison is R², and there the borough models straddle the citywide result of 0.722: Bronx (0.737) is better, Staten Island (0.717), Manhattan (0.700), Queens (0.696) and Brooklyn (0.670) are not. **This project does not demonstrate that per-borough modeling outperforms the aggregate.** Establishing that would require comparing borough forecasts against the citywide model's predictions decomposed to borough level, on a common holdout.

---

## ⚠️ Limitations & Next Steps

1. **No holdout validation.** Every metric in this README is in-sample. The immediate next step is `Prophet.cross_validation()` with a rolling origin, which would produce genuine out-of-sample forecast error. Expect the reported figures to fall.
2. **The tuning gain is unverified.** The daily improvement from R² 0.631 to 0.722 comes from raising model flexibility, which improves training fit by construction. It cannot be called a generalization gain until cross-validated.
3. **Daily values are interpolated.** Constant `KWH ÷ days` within each billing period produces an artificially smooth series, which inflates daily fit statistics relative to what true daily meter data would yield.
4. **11.9% of calendar days are missing.** 685 of the 5,780 days in the series window have no records. Prophet interpolates across these gaps silently, which further smooths an already-smoothed series.
5. **MAPE is unusable.** Near-zero consumption in early records makes percentage error meaningless. A symmetric alternative such as sMAPE, or simply excluding the 2009 to 2010 period, would give a usable percentage metric.
6. **Yearly series is too short.** 17 observations, one of them a two-week partial year, cannot support a meaningful trend model. The 10 and 20 year forecasts are illustrative only.
7. **Tuning is not a systematic sweep.** Five configurations were hand-picked, each varying multiple parameters at once, so no individual parameter's contribution is isolated. A one-axis-at-a-time sweep under cross-validation would identify which changes actually matter.
8. **Borough comparison is not like-for-like.** Borough models use their own configuration, so their results cannot be read as the tuned citywide model applied at borough scale.
9. **Coverage is not all of NYC.** The source data covers NYC Housing Authority developments and related facilities, not citywide residential and commercial consumption. Results describe that portfolio, not the city as a whole.
10. **The summary table is hardcoded.** The final evaluation table transcribes metrics as literals rather than reading the `results_*` lists the notebook builds. The values are correct for this run, but re-running on refreshed data would leave the table stale while the borough rows update. Wiring it to the computed results is a small, worthwhile fix.

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Clone & Setup
```bash
# Clone the repository
git clone https://github.com/krishnamaniyar2209/nyc-electricity-forecasting-prophet.git

# Navigate to the project folder
cd nyc-electricity-forecasting-prophet

# Install all dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook NYC_Electricity_Forecasting.ipynb
```

---

## 🚀 Usage

1. Download the dataset from [NYC Open Data](https://data.cityofnewyork.us/Housing-Development/Electric-Consumption-And-Cost-2010-Feb2023-/jr24-e7cr/about_data) and export it as a `.csv` file
2. Open `NYC_Electricity_Forecasting.ipynb` in Jupyter or Google Colab
3. Update the `file_path` variable in **Cell 2** to point to your downloaded CSV file:
```python
file_path = "/your/path/to/electric_consumption.csv"
```
4. Run all cells sequentially from top to bottom
5. All plots, forecast tables, and metrics are generated automatically

> **Note on reproducibility:** NYC Open Data refreshes this dataset periodically. A freshly downloaded copy will contain more records than the 553,666 analyzed here, and every figure in this README will shift accordingly.

---

## 💡 Key Findings

- **Linear growth** consistently outperforms logistic and flat on both the daily and monthly series
- **Custom seasonality tuning** (yearly `fourier_order=10`, weekly `fourier_order=5`, `n_changepoints=30`) improved in-sample daily R² from 0.631 to **0.722** and cut MAE by 12%, from 289K to 254K KWH
- **Borough-level models** fit their own series well (R² 0.670 to 0.737), bracketing the citywide 0.722 — no borough model clearly beats the aggregate, and the lower absolute errors reflect smaller series rather than better modeling
- Prophet's additive model captures **NYC's strong summer electricity spikes** across all five boroughs
- The yearly series is inherently unmodellable at 17 observations, and wide confidence intervals are the correct output rather than a failure
- **Data quality dominates model choice here.** Interpolated daily values, an 11.9% calendar gap, near-zero early records, and a partial first year shaped the results more than any hyperparameter did

---

## 🛠️ Technologies Used

| Tool | Version | Purpose |
|---|---|---|
| [Python](https://python.org) | 3.10+ | Core language |
| [Facebook Prophet](https://facebook.github.io/prophet/) | 1.1+ | Time series forecasting |
| [pandas](https://pandas.pydata.org/) | 1.5+ | Data manipulation |
| [NumPy](https://numpy.org/) | 1.23+ | Numerical operations |
| [Matplotlib](https://matplotlib.org/) | 3.6+ | Visualization |
| [scikit-learn](https://scikit-learn.org/) | 1.1+ | MAE, MAPE, R² metrics |
| [Jupyter Notebook](https://jupyter.org/) | Latest | Development environment |

---

## 👤 Author

**Krishna Maniyar**, Data Analyst
- 🎓 Pace University, Seidenberg School of CSIS, MS in Data Science
- 📘 Originally built for CS675: Introduction to Data Science; refreshed with NYC Open Data through October 2025
- 📧 maniyarkrishnakm22@gmail.com
- 🔗 [GitHub](https://github.com/krishnamaniyar2209) · [LinkedIn](https://www.linkedin.com/in/krishnamaniyar/) · [Portfolio](https://krishnamaniyar2209.github.io/)

---

<p align="center">
  Made with ❤️ for CS675 @ Pace University
  <br><br>
  <img src="https://img.shields.io/badge/Pace%20University-Seidenberg%20School%20of%20CSIS-blue" />
</p>
