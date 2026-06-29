# ATMOS — Weather Prediction

7-day weather forecasting using a Random Forest model trained on 8 years of historical data. Includes rain classification, multi-model comparison, and multi-city comparison.

## Tech Stack
- Python 3.12
- scikit-learn (Random Forest Regressor + Classifier, Linear Regression)
- XGBoost
- Open-Meteo API (free, no API key required)
- FastAPI + vanilla JS frontend
- Streamlit (alternate UI)
- joblib

## Features
- Fetches live 7-day weather data for any city via Open-Meteo
- Predicts next 7 days max temperature (Random Forest)
- Rain classifier — predicts whether it will rain tomorrow (79.8% accuracy)
- Feature importance chart
- Prediction confidence range per day
- CSV export of predictions
- Multi-city comparison with side-by-side metrics and temperature trend chart

## Project Structure

```
weather/
├── api.py                            # FastAPI backend
├── app.py                            # Streamlit UI
├── frontend/                         # Vanilla JS frontend
├── src/
│   ├── data/
│   │   └── fetch_historical.py       # Fetch 8-year historical data
│   └── models/
│       ├── train.py                  # Temperature model training
│       ├── predict.py                # Prediction interface
│       ├── fetch_weather.py          # Live weather fetch
│       ├── rain_classifier.py        # Rain classifier training
│       ├── compare_models.py         # LR / RF / XGBoost comparison
│       └── cross_validate.py         # 5-fold time-series cross-validation
├── models/                           # Saved models and metrics
├── data/
│   └── weather_history.csv           # Historical training data
└── tests/
    └── test_api.py                   # API integration tests
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Step 1 — Fetch historical data:
```bash
python src/data/fetch_historical.py
```

Step 2 — Train the temperature model:
```bash
python src/models/train.py
```

Step 3 — Train the rain classifier:
```bash
python src/models/rain_classifier.py
```

Step 4 — Run the API:
```bash
uvicorn api:app --reload
```

Or run the Streamlit UI:
```bash
streamlit run app.py
```

Optional — compare models and run cross-validation:
```bash
python src/models/compare_models.py
python src/models/cross_validate.py
```

## Tests

```bash
pytest tests/test_api.py -v
```

## Model Details

### Temperature Forecast
- Algorithm: Random Forest Regressor (100 trees)
- Features: temperature_2m_max, temperature_2m_min, precipitation_sum, windspeed_10m_max, temp_max_lag1, temp_min_lag1
- Target: next day maximum temperature (°C)
- Training data: 2015–2022 (Open-Meteo Archive API)
- MAE: 0.84°C, RMSE: 1.16°C

### Rain Classifier
- Algorithm: Random Forest Classifier (100 trees)
- Target: will it rain tomorrow? (precipitation > 1mm)
- Accuracy: 79.8% | Precision: 0.792 | Recall: 0.792

### Model Comparison (train/test split)

| Model | MAE | RMSE |
|---|---|---|
| Linear Regression | 0.85 | 1.19 |
| Random Forest | 0.84 | 1.16 |
| XGBoost | 0.91 | 1.27 |

### 5-Fold Time-Series Cross-Validation

| Model | MAE Mean | MAE Std |
|---|---|---|
| Linear Regression | 0.839 | 0.047 |
| Random Forest | 0.853 | 0.035 |
| XGBoost | 0.921 | 0.036 |

Linear Regression edges Random Forest on CV mean MAE; Random Forest wins on the held-out test split and has lower variance.

## Data Source
Open-Meteo (https://open-meteo.com) — free, no API key required.
Historical archive: archive-api.open-meteo.com
Live forecast: api.open-meteo.com
