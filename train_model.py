import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import LeaveOneOut, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Load data
df = pd.read_csv("extracted_features.csv")

# 2. Map labels to binary research question:
# Variable (RR_LYR, EB) -> 1, Non-Variable (NON_VAR) -> 0
df["is_variable"] = df["label"].apply(lambda x: 0 if x == "NON_VAR" else 1)

feature_cols = ["period_days", "peak_power", "amplitude", "flux_std", "skewness", "kurtosis"]
X = df[feature_cols]
y = df["is_variable"]

print(f"Total samples: {len(df)}")
print(f"Variable stars: {sum(y == 1)}, Stable stars: {sum(y == 0)}\n")

# 3. Define the 3 models
models = {
    "Baseline (Dummy - Most Frequent)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression (Standardized)": make_pipeline(StandardScaler(), LogisticRegression()),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# 4. Evaluate using Leave-One-Out Cross-Validation (LOOCV)
loo = LeaveOneOut()
results = []

print("=" * 60)
print(f"{'Model':<35} | {'Accuracy':<10} | {'F1-Score':<10}")
print("=" * 60)

for name, model in models.items():
    cv_scores = cross_validate(model, X, y, cv=loo, scoring=["accuracy", "f1_macro"])
    acc = cv_scores["test_accuracy"].mean()
    f1 = cv_scores["test_f1_macro"].mean()
    results.append({"Model": name, "Accuracy": acc, "F1-Score": f1})
    print(f"{name:<35} | {acc:<10.3f} | {f1:<10.3f}")

print("=" * 60)

# 5. Fit Random Forest to analyze Feature Importances
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

print("\n--- Random Forest Feature Importances ---")
for feat, imp in importances.items():
    print(f"{feat:<15}: {imp:.4f}")

# 6. Generate and save Plots for your Research Report
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot A: Feature Importance
sns.barplot(x=importances.values, y=importances.index, ax=axes[0], hue=importances.index, palette="viridis", legend=False)
axes[0].set_title("A. Feature Importance (Random Forest)")
axes[0].set_xlabel("Gini Importance")

# Plot B: Cross-Validated Confusion Matrix for Random Forest
from sklearn.model_selection import cross_val_predict
y_pred_rf = cross_val_predict(rf, X, y, cv=loo)
cm = confusion_matrix(y, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stable", "Variable"])
disp.plot(ax=axes[1], cmap="Blues", colorbar=False)
axes[1].set_title("B. Confusion Matrix (LOOCV)")

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=200)
print("\nEvaluation plot saved as 'model_evaluation.png'")