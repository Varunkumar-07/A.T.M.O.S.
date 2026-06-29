import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from src.constants import FEATURE_COLS

# 1. Load data
df = pd.read_csv("data/weather_history.csv")

# 2. Lag features
df["temp_max_lag1"] = df["temperature_2m_max"].shift(1)
df["temp_min_lag1"] = df["temperature_2m_min"].shift(1)
df.dropna(inplace=True)

# 3. Binary target
df["rain_tomorrow"] = (df["precipitation_sum"].shift(-1) > 1.0).astype(int)
df.dropna(inplace=True)

# 4. Features
X = df[FEATURE_COLS]
y = df["rain_tomorrow"]

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 6. Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 7. Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

print(f"✅ Accuracy: {accuracy:.3f}")
print(classification_report(y_test, y_pred))

# 8. Save classifier
joblib.dump({"model": clf, "features": FEATURE_COLS}, "models/rain_classifier.pkl")
print("💾 Classifier saved to models/rain_classifier.pkl")

# 9. Save metrics
metrics = {
    "accuracy": round(accuracy, 3),
    "rain_precision": round(report["1"]["precision"], 3),
    "rain_recall": round(report["1"]["recall"], 3),
}
with open("models/rain_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("📈 Metrics saved to models/rain_metrics.json")
