# Weather Prediction ML System

Real-time weather forecasting app using a Random Forest model trained on 8 years of historical data, with rain classification, multi-model comparison, and multi-city support.

## Tech Stack
- Python 3.12
- scikit-learn (Random Forest Regressor + Classifier, Linear Regression)
- XGBoost
- Open-Meteo API (free, no API key required)
- Streamlit (web UI)
- joblib (model serialization)

## Features
- Fetches real-time 7-day weather data for any city worldwide
- Predicts next **7 days** max temperature using ML (Random Forest)
- Rain classifier — predicts whether it will rain tomorrow (79.8% accuracy)
- Displays feature importance chart
- Shows prediction confidence range (±MAE) and weather condition badge per day
- CSV export of predictions
- Multi-city comparison — side-by-side metrics and temperature trend chart for two cities
- Sidebar displays model metrics (MAE, RMSE), rain classifier stats, and model comparison table

## Project Structure

```
weather/
├── app.py                        # Streamlit web app
├── src/
│   ├── data/
│   │   └── fetch_historical.py   # Fetch 8-year historical data
│   └── models/
│       ├── train.py              # Model training script (Random Forest)
│       ├── predict.py            # Temperature prediction interface
│       ├── fetch_weather.py      # Live weather fetch
│       ├── rain_classifier.py    # Rain classification training script
│       ├── compare_models.py     # LR / RF / XGBoost comparison
│       └── cross_validate.py     # 5-fold time-series cross-validation
├── models/
│   ├── weather_model.pkl         # Trained temperature model
│   ├── rain_classifier.pkl       # Trained rain classifier
│   ├── feature_importance.json   # Feature importance scores
│   ├── metrics.json              # MAE and RMSE
│   ├── rain_metrics.json         # Rain classifier accuracy/precision/recall
│   ├── model_comparison.json     # LR vs RF vs XGBoost results
│   └── cv_results.json           # 5-fold CV results
├── data/
│   └── weather_history.csv       # Historical training data
└── README.md
```

## Setup

```bash
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

Step 4 — Run the app:
```bash
streamlit run app.py
```

Optional — compare models:
```bash
python src/models/compare_models.py
python src/models/cross_validate.py
```

## Model Details

### Temperature Forecast (Random Forest)
- **Algorithm:** Random Forest Regressor (100 trees)
- **Features:** temperature_2m_max, temperature_2m_min, precipitation_sum, windspeed_10m_max, temp_max_lag1, temp_min_lag1
- **Target:** Next day maximum temperature (°C)
- **Training data:** 2015–2022 (Open-Meteo Archive API)
- **MAE:** 0.84°C, **RMSE:** 1.16°C

### Rain Classifier (Random Forest)
- **Algorithm:** Random Forest Classifier (100 trees)
- **Target:** Will it rain tomorrow? (precipitation > 1mm)
- **Accuracy:** 79.8% | **Precision:** 0.792 | **Recall:** 0.792

### Model Comparison (single train/test split)

| Model | MAE | RMSE |
|---|---|---|
| Linear Regression | 0.85 | 1.19 |
| **Random Forest** | **0.84** | **1.16** |
| XGBoost | 0.91 | 1.27 |

### 5-Fold Time-Series Cross-Validation

| Model | MAE Mean | MAE Std |
|---|---|---|
| **Linear Regression** | **0.839** | 0.047 |
| Random Forest | 0.853 | 0.035 |
| XGBoost | 0.921 | 0.036 |

> Linear Regression edges RF on CV mean MAE; RF is more consistent (lower std) and wins on the held-out test split.

## Data Source
Open-Meteo (https://open-meteo.com) — free, no API key required.  
Historical archive: archive-api.open-meteo.com  
Live forecast: api.open-meteo.com
