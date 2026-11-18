"""
Diabetes Risk Assessment - Machine Learning Project
Dataset: Pima Indians Diabetes (OpenML). Falls back to synthetic data if offline.

Run:  pip install pandas numpy scikit-learn matplotlib seaborn joblib
      python diabetes_risk_assessment.py
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)

warnings.filterwarnings("ignore")
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
SEED = 42

# ---------------------------------------------------------------- 1. LOAD DATA
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigree", "Age"]


def load_data():
    if os.path.exists("diabetes.csv"):           # local copy (Kaggle format)
        df = pd.read_csv("diabetes.csv")
        df.columns = FEATURES + ["Outcome"]
        return df
    try:
        from sklearn.datasets import fetch_openml
        raw = fetch_openml("diabetes", version=1, as_frame=True, parser="auto").frame
        raw.columns = FEATURES + ["Outcome"]
        raw["Outcome"] = (raw["Outcome"] == "tested_positive").astype(int)
        return raw
    except Exception:
        print("[!] Could not download dataset - using synthetic data instead.")
        rng = np.random.default_rng(SEED)
        n = 768
        df = pd.DataFrame({
            "Pregnancies": rng.poisson(3.8, n),
            "Glucose": rng.normal(121, 32, n).clip(50, 200),
            "BloodPressure": rng.normal(69, 19, n).clip(30, 120),
            "SkinThickness": rng.normal(20, 16, n).clip(0, 99),
            "Insulin": rng.gamma(2, 40, n),
            "BMI": rng.normal(32, 7.8, n).clip(18, 67),
            "DiabetesPedigree": rng.gamma(2, 0.25, n),
            "Age": rng.gamma(6, 5.5, n).clip(21, 81),
        })
        z = (0.035 * (df.Glucose - 121) + 0.07 * (df.BMI - 32) + 0.02 * (df.Age - 33)
             + 0.9 * df.DiabetesPedigree - 0.9)
        df["Outcome"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
        return df


df = load_data()
print("Shape:", df.shape)
print(df.head(), "\n")

# -------------------------------------------------------------- 2. CLEAN / EDA
# In this dataset a 0 in these columns is a missing value, not a real reading.
for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    df[col] = df[col].replace(0, np.nan)
print("Missing values:\n", df.isna().sum(), "\n")
print("Class balance:\n", df["Outcome"].value_counts(normalize=True).round(3), "\n")

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation")
plt.tight_layout(); plt.savefig(f"{OUT}/correlation_heatmap.png", dpi=120); plt.close()

df.drop(columns="Outcome").hist(figsize=(11, 8), bins=25)
plt.tight_layout(); plt.savefig(f"{OUT}/feature_distributions.png", dpi=120); plt.close()

# ------------------------------------------------------------ 3. TRAIN MODELS
X, y = df[FEATURES], df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)


def make(model):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", model)])


models = {
    "Logistic Regression": make(LogisticRegression(max_iter=1000, class_weight="balanced")),
    "Random Forest": make(RandomForestClassifier(n_estimators=300, max_depth=6,
                                                 class_weight="balanced", random_state=SEED)),
    "Gradient Boosting": make(GradientBoostingClassifier(random_state=SEED)),
}

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
results, fitted = [], {}
plt.figure(figsize=(7, 6))
for name, pipe in models.items():
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    pipe.fit(X_train, y_train)
    fitted[name] = pipe
    pred, prob = pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1]
    results.append({"Model": name, "CV AUC": cv_auc,
                    "Accuracy": accuracy_score(y_test, pred),
                    "Precision": precision_score(y_test, pred),
                    "Recall": recall_score(y_test, pred),
                    "F1": f1_score(y_test, pred),
                    "Test AUC": roc_auc_score(y_test, prob)})
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={results[-1]['Test AUC']:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves"); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/roc_curves.png", dpi=120); plt.close()

res = pd.DataFrame(results).set_index("Model").round(3)
print("=== Model comparison ===\n", res, "\n")
res.to_csv(f"{OUT}/model_comparison.csv")

# ------------------------------------------------------------ 4. BEST MODEL
best_name = res["Test AUC"].idxmax()
best = fitted[best_name]
print(f"Best model: {best_name}\n")
print(classification_report(y_test, best.predict(X_test),
                            target_names=["No Diabetes", "Diabetes"]))

cm = confusion_matrix(y_test, best.predict(X_test))
plt.figure(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout(); plt.savefig(f"{OUT}/confusion_matrix.png", dpi=120); plt.close()

# Feature importance
inner = best.named_steps["model"]
imp = inner.feature_importances_ if hasattr(inner, "feature_importances_") else np.abs(inner.coef_[0])
imp = pd.Series(imp, index=FEATURES).sort_values()
imp.plot(kind="barh", figsize=(7, 4), title="Feature Importance")
plt.tight_layout(); plt.savefig(f"{OUT}/feature_importance.png", dpi=120); plt.close()

joblib.dump(best, f"{OUT}/diabetes_model.joblib")


# ------------------------------------------------------------ 5. RISK ASSESSMENT
def assess_risk(pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, pedigree, age):
    """Return risk probability and category for one patient."""
    patient = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                             insulin, bmi, pedigree, age]], columns=FEATURES)
    # 0 means "unknown" for these fields -> let the pipeline impute the median
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        patient[col] = patient[col].replace(0, np.nan)
    p = best.predict_proba(patient)[0, 1]
    level = "LOW" if p < 0.30 else "MODERATE" if p < 0.60 else "HIGH"
    return p, level


# ------------------------------------------------------------ 6. INTERACTIVE MODE
PROMPTS = [
    # (label, min, max)
    ("Number of pregnancies (0 if male)", 0, 20),
    ("Glucose level, mg/dL (e.g. 70-200)", 40, 300),
    ("Diastolic blood pressure, mm Hg (e.g. 60-90)", 30, 150),
    ("Skin-fold thickness, mm (e.g. 10-40, 0 if unknown)", 0, 100),
    ("Insulin level, mu U/ml (e.g. 15-300, 0 if unknown)", 0, 900),
    ("BMI (e.g. 18-45)", 10, 70),
    ("Diabetes pedigree function (family history score, e.g. 0.1-1.5)", 0.0, 2.5),
    ("Age in years", 1, 120),
]


def ask(label, lo, hi):
    while True:
        raw = input(f"  {label}: ").strip()
        try:
            v = float(raw)
            if lo <= v <= hi:
                return v
            print(f"    Please enter a value between {lo} and {hi}.")
        except ValueError:
            print("    Please enter a number.")


def advice(level):
    return {
        "LOW": "Keep up a healthy diet, regular exercise and routine check-ups.",
        "MODERATE": "Consider a check-up and an HbA1c / fasting glucose test. "
                    "Watch weight, diet and activity.",
        "HIGH": "Please consult a doctor for proper testing soon. "
                "Early diagnosis and lifestyle changes make a big difference.",
    }[level]


def interactive():
    print("\n" + "=" * 55)
    print(" DIABETES RISK ASSESSMENT - enter patient data")
    print("=" * 55)
    while True:
        vals = [ask(*p) for p in PROMPTS]
        p, level = assess_risk(*vals)
        print("\n" + "-" * 55)
        print(f"  Estimated diabetes risk: {p:.1%}  ->  {level} RISK")
        print(f"  {advice(level)}")
        print("-" * 55)
        print("  Disclaimer: educational tool, not a medical diagnosis.\n")
        if input("Assess another person? (y/n): ").strip().lower() != "y":
            break
    print("Goodbye!")


print(f"\nModel ready ({best_name}). Plots and metrics saved in ./{OUT}/")
interactive()
