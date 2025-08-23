# src/models/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# 1. Load data
df = pd.read_csv("data/weather_history.csv")

# 2. Features (X) and Target (y)
X = df[["temperature_2m_max", "temperature_2m_min", "precipitation_sum"]]
y = df["temperature_2m_max"].shift(-1).dropna()
X = X.iloc[:-1]  # align with y

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"✅ Model trained!")
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# 6. Save model
joblib.dump(model, "models/weather_model.pkl")
print("💾 Model saved to models/weather_model.pkl")
