import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from src.constants import FEATURE_COLS

# 1. Load data
df = pd.read_csv("data/weather_history.csv")

# 2. Lag features
df["temp_max_lag1"] = df["temperature_2m_max"].shift(1)
df["temp_min_lag1"] = df["temperature_2m_min"].shift(1)
df.dropna(inplace=True)

available_features = [col for col in FEATURE_COLS if col in df.columns]
X = df[available_features]
y = df["temperature_2m_max"].shift(-1).dropna()
X = X.iloc[:-1]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. Train and evaluate models
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = round(float(mean_absolute_error(y_test, y_pred)), 2)
    rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2)
    results[name] = {"MAE": mae, "RMSE": rmse}

# 5. Save results
with open("models/model_comparison.json", "w") as f:
    json.dump(results, f, indent=2)

# 6. Print comparison table
print(f"\n{'Model':<20} {'MAE':>6} {'RMSE':>7}")
print("-" * 36)
for name, m in results.items():
    print(f"{name:<20} {m['MAE']:>6.2f} {m['RMSE']:>7.2f}")
print()
print("Results saved to models/model_comparison.json")
