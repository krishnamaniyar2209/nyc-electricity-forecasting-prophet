# 🔌 New York City Electricity Consumption Forecasting

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Prophet](https://img.shields.io/badge/Facebook%20Prophet-Time%20Series-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Metrics-red?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![University](https://img.shields.io/badge/Pace%20University-CS675-blue)

> A comprehensive time series forecasting project using **Facebook's Prophet** model to predict NYC electricity consumption across all five boroughs — built for CS675: Introduction to Data Science at Pace University.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Models & Results](#-models--results)
- [Borough-Level Forecasting](#-borough-level-forecasting)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Findings](#-key-findings)
- [Technologies Used](#-technologies-used)
- [Author](#-author)
- [License](#-license)

---

## 🗽 Overview

This project forecasts **New York City's daily, monthly, and yearly electricity consumption** using Facebook Prophet's additive time series model. The dataset spans from **2009 to September 2025** and covers all five NYC boroughs: Brooklyn, Manhattan, Bronx, Queens, and Staten Island.

The notebook covers:
- ✅ Automatic time-unit detection (daily / monthly / yearly)
- ✅ Forecasting with three growth models: **Linear**, **Logistic**, and **Flat**
- ✅ Hyperparameter tuning via custom seasonality and trend changepoints
- ✅ Model evaluation using **MAE**, **MAPE**, and **R²**
- ✅ Independent borough-level forecasting (extra credit)

---

## 📊 Dataset

**Source:** [NYC Open Data — Electric Consumption and Cost (2010–2025)](https://data.cityofnewyork.us/Housing-Development/Electric-Consumption-And-Cost-2010-Feb2023-/jr24-e7cr/about_data)

| Property | Details |
|---|---|
| Rows | ~553,000 |
| Columns | 27 |
| Date Range | December 2009 – September 2025 |
| Boroughs | Brooklyn, Manhattan, Bronx, Queens, Staten Island |
| Target Variable | `Consumption (KWH)` |
| Granularity | Billing records (expanded to daily) |

The raw dataset contains billing records per meter. Each record is expanded into true daily consumption by dividing total KWH by billing period length, then re-aggregated to produce clean daily, monthly, and yearly time series.

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

### Step 1 — Data Preprocessing
- Parsed billing records with variable-length service periods
- Removed invalid rows (zero consumption, negative days)
- Filtered to 5 valid NYC boroughs only
- Expanded each billing row into individual daily records (`KWH / days`)
- Aggregated to daily totals, then resampled to monthly and yearly means

### Step 2 — Time-Unit Auto Detection
A `detect_frequency()` function automatically identifies the time granularity of any input dataset by measuring the median gap between consecutive dates:
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

### Step 3 — Forecasting Horizons

| Dataset | Forecast Horizons |
|---|---|
| Daily | 100 days, 200 days, 365 days |
| Monthly | 1 month, 6 months, 9 months |
| Yearly | 1 year, 10 years, 20 years |

### Step 4 — Model Tuning

| Parameter | Values Tested |
|---|---|
| Growth | `linear`, `logistic`, `flat` |
| Seasonality period | 7, 365.25 days |
| Fourier order | 3, 5, 8, 10 |
| n_changepoints | 5, 10, 15, 25, 30 |
| changepoint_prior_scale | 0.1, 0.3, 0.5 |

### Step 5 — Evaluation Metrics
- **MAE** — Mean Absolute Error
- **MAPE** — Mean Absolute Percentage Error
- **R²** — Coefficient of Determination (via scikit-learn)

> ⚠️ **Note on MAPE:** Daily and monthly MAPE values are artificially inflated due to near-zero consumption records from early 2009–2010 billing data. MAE and R² are the primary evaluation metrics for those datasets.

---

## 📈 Models & Results

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
| **Custom Seasonality (tuned)** | **322,904** | **0.517** ✅ |

### Yearly Dataset

| Model | MAE (KWH) | MAPE | R² |
|---|---|---|---|
| Linear Growth | 592,484 | 31.93% | 0.040 |
| Logistic Growth | 602,763 | 33.25% | -0.032 |
| Flat Growth | 577,090 | 32.33% | -0.000 |
| Changepoint Tuned | 592,473 | 31.93% | 0.040 |

> Low R² on yearly data is expected — only 17 data points with high inter-year variance (COVID dip in 2018/2019, spikes in 2020).

---

## 🗺️ Borough-Level Forecasting

Independent Prophet models trained and evaluated for each of the 5 NYC boroughs (365-day forecast):

| Borough | Daily Records | MAE (KWH) | R² |
|---|---|---|---|
| Brooklyn | 5,027 | 83,628 | 0.670 |
| Manhattan | 5,080 | 91,102 | 0.700 |
| **Bronx** | 5,028 | **66,002** | **0.737** ✅ |
| Queens | 5,078 | 30,511 | 0.696 |
| Staten Island | 5,027 | 9,488 | 0.717 |

**Key observations:**
- 🏆 **Bronx** achieves the highest R² at 0.7367
- 🎯 **Staten Island** has the lowest MAE, consistent with its smaller size
- 📈 **Brooklyn** has the highest absolute consumption overall
- ☀️ All 5 boroughs show clear **summer consumption peaks** (July–August)

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

---

## 💡 Key Findings

- **Linear growth** consistently outperforms logistic and flat for both daily and monthly datasets
- **Custom seasonality tuning** (yearly fourier_order=10, weekly fourier_order=5, n_changepoints=30) improved daily R² from 0.63 → **0.72**
- The yearly dataset is inherently difficult to model with only 17 observations — wide confidence intervals are expected and normal
- Prophet's additive model captures **NYC's strong summer electricity spikes** effectively across all boroughs
- **Borough-level models** outperform the city-wide aggregate in precision, confirming that localized consumption patterns exist
- **Data quality matters** — early billing records with near-zero values affect MAPE; always check your target distribution before forecasting

---

## 🛠️ Technologies Used

| Tool | Version | Purpose |
|---|---|---|
| [Python](https://python.org) | 3.10+ | Core language |
| [Facebook Prophet](https://facebook.github.io/prophet/) | Latest | Time series forecasting |
| [pandas](https://pandas.pydata.org/) | Latest | Data manipulation |
| [NumPy](https://numpy.org/) | Latest | Numerical operations |
| [Matplotlib](https://matplotlib.org/) | Latest | Visualization |
| [scikit-learn](https://scikit-learn.org/) | Latest | MAE, MAPE, R² metrics |
| [Jupyter Notebook](https://jupyter.org/) | Latest | Development environment |

---

## 👤 Author

**Krishna Maniyar**
- 🎓 Pace University — Seidenberg School of CSIS
- 📘 CS675: Introduction to Data Science (Fall 2024)
- 🔗 [GitHub](https://github.com/krishnamaniyar2209)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ for CS675 @ Pace University
  <br><br>
  <img src="https://img.shields.io/badge/Pace%20University-Seidenberg%20School%20of%20CSIS-blue" />
</p>
