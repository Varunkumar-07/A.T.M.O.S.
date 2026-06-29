import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBRegressor
from src.constants import FEATURE_COLS

# 1. Load data
df = pd.read_csv("data/weather_history.csv")

# 2. Lag features
df["temp_max_lag1"] = df["temperature_2m_max"].shift(1)
df["temp_min_lag1"] = df["temperature_2m_min"].shift(1)
df.dropna(inplace=True)

X = df[FEATURE_COLS]
y = df["temperature_2m_max"].shift(-1).dropna()
X = X.iloc[:-1]

# 3. Cross-validation
tscv = TimeSeriesSplit(n_splits=5)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
}

# 4. Evaluate
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
    mae_mean = round(float(-scores.mean()), 3)
    mae_std = round(float(scores.std()), 3)
    results[name] = {"MAE_mean": mae_mean, "MAE_std": mae_std}

# 5. Save results
with open("models/cv_results.json", "w") as f:
    json.dump(results, f, indent=2)

# 6. Print table
print(f"\n{'Model':<20} {'MAE Mean':>10} {'MAE Std':>9}")
print("-" * 42)
for name, m in results.items():
    print(f"{name:<20} {m['MAE_mean']:>10.3f} {m['MAE_std']:>9.3f}")
print()
print("✅ Results saved to models/cv_results.json")
