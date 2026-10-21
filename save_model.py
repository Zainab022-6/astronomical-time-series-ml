import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load features extracted from your earlier run
df = pd.read_csv("extracted_features.csv")
df["is_variable"] = df["label"].apply(lambda x: 0 if x == "NON_VAR" else 1)

feature_cols = ["period_days", "peak_power", "amplitude", "flux_std", "skewness", "kurtosis"]
X = df[feature_cols]
y = df["is_variable"]

# Train production model and serialize
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "stellar_classifier.joblib")
print("Saved model file: stellar_classifier.joblib")