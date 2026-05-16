# Tokyo Weather Bayesian Forecaster

**Scientific ML project exploring uncertainty estimation in time series forecasting using deep learning and ensemble methods.**

A reproducible pipeline for short-range temperature forecasting that outputs both predictions and uncertainty estimates. Processes 55 years of hourly reanalysis data via PySpark, trains a deep ensemble in PyTorch with Gaussian NLL loss to estimate prediction uncertainty, and validates interval calibration on a held-out chronological test set. Containerized via Docker Hub for zero-setup reproducibility.

Dataset sourced from Open-Meteo Archive API — `https://archive-api.open-meteo.com/v1/archive`.

Dataset description from Open-Meteo:
The Open-Meteo Archive API provides historical weather data from 1940 to present with hourly resolution. Data is derived from reanalysis models combining observations with numerical weather prediction, offering consistent long-term records suitable for climate analysis and machine learning applications.

The dataset is ideal for practicing time series forecasting, uncertainty quantification, probabilistic forecasting, and scientific ML workflows. It allows exploration of how prediction confidence varies across seasons, weather regimes, and forecast horizons.

## Architecture

```text
Open-Meteo API -> fetch_tokyo_data.py -> CSV (raw)
                                          |
                                  spark_etl.py (PySpark)
                                          |
                          tokyo_features.csv (engineered)
                                          |
                          bayesian_forecasting.py (PyTorch)
                                          |
                          |- Test metrics: RMSE, MAE, 95% coverage
                          |- Live forecast: next 24h hourly + 7d daily
                          +- visualize_results.py (Matplotlib)
```

## Tech Stack

* **Language:** Python 3.12
* **Data Engineering:** PySpark 3.5, pandas 2.2, numpy 1.26
* **Modeling:** PyTorch 2.2, scikit-learn 1.4
* **Uncertainty:** Deep ensembles + Gaussian NLL loss + empirical interval calibration
* **Visualization:** Matplotlib 3.8
* **Containers:** Docker + Docker Hub (`yotane/tokyo-weather-bayesian-forecast`)
* **Data Source:** Open-Meteo Archive API (REST, hourly resolution, 1970 to 2024)

## Prerequisites

* Docker Desktop
* Docker Compose (optional, for local rebuilds)

## Installation

```bash
# Clone repository
git clone https://github.com/Yotane/tokyo-weather-bayesian-forecast.git
cd tokyo-weather-bayesian-forecast

# Configure Python environment (local run only)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Running the Project

### Option 1: Docker (Zero-Setup, Pre-Built Image)

```bash
# Pull and run from Docker Hub (fetches data automatically on first run)
docker run --rm -v .//app/data -v ./figures:/app/figures yotane/tokyo-weather-bayesian-forecast
```

Requires Docker Desktop. The `-v` mounts ensure:

* Downloaded data persists in your local `data/` folder
* Generated plots save to your local `figures/` folder

> **Note:** First run automatically fetches historical weather data from the Open-Meteo API. Subsequent runs reuse the local `data/` folder if mounted with `-v`.

### Option 2: Build Locally with Docker

```bash
docker build -t tokyo-weather-bayesian-forecast .
docker run --rm -v .//app/data -v ./figures:/app/figures tokyo-weather-bayesian-forecast
```

### Option 3: Local Run (No Docker)

```bash
# 1. Fetch historical weather data
python fetch_tokyo_data.py

# 2. Generate engineered features
python spark_etl.py

# 3. Train and forecast
python bayesian_forecasting.py

# 4. Generate plots
python visualize_results.py
```

## Forecast Analysis: January 2025

### Recent Conditions (Late December 2024)

The model was trained on hourly weather data through December 31, 2024 using:

* temperature
* precipitation
* humidity
* sea-level pressure
* wind vector components

The final days of 2024 showed relatively mild daytime temperatures with colder overnight lows, creating strong diurnal temperature swings that the model learned through engineered lag and rolling-statistics features.

### 7-Day Forecast (January 1–7, 2025)

The forecasting system predicts a cool and moderately variable start to 2025.

**Key Findings:**

* **Day 1 (Jan 1):** Daily mean 6.9°C, ranging from 3.5°C to 11.2°C
* **Days 2–7:** Gradual cooling and stabilization around 5.6–6.1°C mean temperatures
* **Temperature range:** Overnight lows near 2°C–3°C with afternoon highs near 10°C–11°C
* **Uncertainty:** Prediction intervals widen from approximately ±1.9°C on Day 1 to ±2.7°C by Day 7, reflecting increasing forecast uncertainty over longer horizons

**Interpretation:**

The model captures realistic winter diurnal structure in Tokyo, including colder nighttime temperatures and milder afternoon warming. Forecast uncertainty increases smoothly with forecast horizon, which is consistent with atmospheric predictability limits. The probabilistic forecasts remain well-calibrated on held-out chronological evaluation data.

## Feature Engineering

`spark_etl.py` generates engineered forecasting features from the raw hourly observations.

### Lag Features

```text
temperature_2m_lag_1h
temperature_2m_lag_24h
precipitation_lag_1h
precipitation_lag_24h
```

### Rolling Statistics

```text
temperature_2m_24h_mean
temperature_2m_24h_std
humidity_24h_mean
pressure_msl_24h_mean
pressure_msl_24h_std
precipitation_24h_sum
wind_speed_24h_mean
```

### Cyclical Time Features

```text
hour_sin
hour_cos
month_sin
month_cos
```

### Additional Features

```text
wind_speed
precipitation_log
is_weekend
```

## Example Outputs

### Next 24 Hours (Hourly Forecast)

![24-Hour Forecast](figures/forecast_24h.png)

*Hourly temperature predictions with 95% confidence intervals for the next day.*

### Next 7 Days (Daily Overview)

![7-Day Forecast](figures/forecast_7d.png)

*Daily mean, max, and min temperatures with uncertainty bands for the next week.*

### Sample Console Output

```text
============================================================
FORECASTING RESULTS
============================================================
Test Set Evaluation (Held-Out Chronological Data)
   RMSE (Next 24h): 1.92C | MAE: 1.45C
   95% Interval Coverage: 93.2%

Next 24 Hours (Hourly Forecast)
Hour Ahead   Pred Temp (C)   Uncertainty (sigma)
---------------------------------------------
1            5.6             +/-1.2 C
2            5.1             +/-1.4 C
3            4.7             +/-1.5 C
4            4.4             +/-1.6 C
5            4.1             +/-1.7 C
6            3.7             +/-1.7 C
7            3.5             +/-1.7 C
8            4.0             +/-1.6 C
9            5.4             +/-1.6 C
10           7.2             +/-1.7 C
11           8.8             +/-1.8 C
12           9.9             +/-2.0 C
13           10.6            +/-2.1 C
14           11.0            +/-2.3 C
15           11.2            +/-2.3 C
16           11.0            +/-2.3 C
17           10.4            +/-2.3 C
18           9.3             +/-2.2 C
19           8.0             +/-2.2 C
20           6.9             +/-2.1 C
21           6.1             +/-2.1 C
22           5.5             +/-2.2 C
23           5.0             +/-2.2 C
24           4.5             +/-2.2 C

Next 7 Days (Daily Overview)
Day    Condition       Pred Mean  Pred Max   Pred Min   Uncertainty (sigma)
---------------------------------------------------------------------------
Day 1   Partly Cloudy   6.9        11.2       3.5        +/-1.9 C
Day 2   Partly Cloudy   6.1        10.6       2.4        +/-2.4 C
Day 3   Variable        6.1        10.6       2.4        +/-2.6 C
Day 4   Variable        6.1        10.5       2.5        +/-2.7 C
Day 5   Variable        5.9        10.3       2.4        +/-2.7 C
Day 6   Variable        5.6        9.9        2.2        +/-2.6 C
Day 7   Variable        5.6        9.8        2.1        +/-2.6 C
```


## Project Structure

```text
tokyo-weather-bayesian-forecast/
├── fetch_tokyo_data.py       # Open-Meteo API client, hourly CSV download
├── spark_etl.py              # PySpark feature engineering pipeline
├── bayesian_forecasting.py   # PyTorch deep ensemble forecasting
├── visualize_results.py      # Matplotlib forecast visualization
├── requirements.txt          # Python dependencies
├── Dockerfile                # Reproducible container build
├── .dockerignore
├── data/
│   ├── tokyo_weather_1970_2024.csv
│   ├── tokyo_features.csv
│   └── latest_forecast.csv
├── models/
│   ├── ensemble_model_0.pth
│   ├── ensemble_model_1.pth
│   ├── ensemble_model_2.pth
│   ├── scaler_X.npy
│   └── scaler_y.npy
└── figures/
```

## Key Results

| Metric                | Value | Interpretation                                              |
| --------------------- | ----- | ----------------------------------------------------------- |
| RMSE (Next 24h)       | 1.92C | Average short-range hourly temperature forecasting error    |
| MAE (Next 24h)        | 1.45C | Average absolute forecast deviation                         |
| 95% Interval Coverage | 93.2% | Empirical calibration of probabilistic prediction intervals |

## Uncertainty Estimation

`bayesian_forecasting.py` implements uncertainty-aware forecasting via:

* **Gaussian NLL loss:** Model predicts both mean and variance for every forecast timestep
* **Deep ensembles:** Multiple independently trained models estimate epistemic uncertainty
* **Variance decomposition:** Final uncertainty combines aleatoric and epistemic components
* **Calibration validation:** Empirical interval coverage measured on held-out chronological test data

All forecasting outputs include uncertainty estimates. Prediction intervals default to 95% confidence intervals using:

```text
mean ± 1.96 × standard deviation
```

## Docker Hub Integration

Pre-built image published to:

```text
https://hub.docker.com/r/yotane/tokyo-weather-bayesian-forecast
```

Anyone with Docker Desktop can run the full forecasting pipeline in one command:

```bash
docker run --rm -v .//app/data -v ./figures:/app/figures yotane/tokyo-weather-bayesian-forecast
```

No Python, Java, PySpark, or PyTorch installation required.

## Scope and Limitations

* **Temperature-focused forecasting:** Weather conditions are generated using lightweight rule-based classification rather than full precipitation forecasting
* **Single-location forecasting:** Model is trained only on Tokyo weather data
* **Tabular architecture:** Uses engineered features rather than sequence-aware architectures like LSTMs or Transformers
* **No spatial weather fields:** Forecasting is based on point observations without radar or satellite inputs

## Future Roadmap

* **Sequence modeling:** Replace tabular inputs with LSTM or Transformer architectures
* **Multi-variable forecasting:** Joint temperature and precipitation forecasting
* **Multi-location inference:** Scale forecasting across multiple cities or weather stations
* **Advanced uncertainty methods:** Conformal prediction and probabilistic calibration refinement
* **Operational monitoring:** CI/CD integration and forecasting observability dashboards

## Technical Stack Summary

| Component        | Technology                    |
| ---------------- | ----------------------------- |
| Language         | Python 3.12                   |
| Data Engineering | PySpark 3.5, pandas 2.2       |
| Modeling         | PyTorch 2.2, scikit-learn 1.4 |
| Uncertainty      | Deep ensembles + Gaussian NLL |
| Visualization    | Matplotlib 3.8                |
| Containers       | Docker + Docker Hub           |
| Data Source      | Open-Meteo Archive API        |

## License

This project is for educational purposes. Dataset sourced from Open-Meteo Archive API under their open data policy.

## Author

Matt Raymond Ayento
Nagoya University
G30, 3rd Year Automotive Engineering (Electrical, Electronics, Information Engineering)
