# src/models/train.py
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
from src.constants import FEATURE_COLS

# 1. Load data
df = pd.read_csv("data/weather_history.csv")

# 2. Lag features
df["temp_max_lag1"] = df["temperature_2m_max"].shift(1)
df["temp_min_lag1"] = df["temperature_2m_min"].shift(1)
df.dropna(inplace=True)

# 3. Features (X) and Target (y)
available_features = [col for col in FEATURE_COLS if col in df.columns]
X = df[available_features]

# Target: tomorrow's max temp
y = df["temperature_2m_max"].shift(-1).dropna()
X = X.iloc[:-1]  # align with y

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 5. Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"✅ Model trained with features: {available_features}")
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# 7. Save model & feature list
joblib.dump({"model": model, "features": available_features}, "models/weather_model.pkl")
print("💾 Model + features saved to models/weather_model.pkl")

# 8. Save feature importance
importance = dict(zip(available_features, model.feature_importances_))
with open("models/feature_importance.json", "w") as f:
    json.dump(importance, f, indent=2)
print("📊 Feature importance saved to models/feature_importance.json")

# 9. Save metrics
metrics = {"MAE": round(mae, 2), "RMSE": round(rmse, 2)}
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("📈 Metrics saved to models/metrics.json")
