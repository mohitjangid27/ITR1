"""Train the heart disease model and save it to model.joblib.
Data: UCI Cleveland Heart Disease (downloads automatically), or put a local
'heart.csv' (UCI column order, 13 features + target) next to this file.
Falls back to synthetic data if offline.   Run:  python train_model.py
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal"]
URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "heart-disease/processed.cleveland.data")
SEED = 42


def load():
    try:
        if os.path.exists("heart.csv"):
            df = pd.read_csv("heart.csv", header=None if "age" not in open("heart.csv").readline().lower() else 0)
            df.columns = COLS + ["target"]
        else:
            df = pd.read_csv(URL, header=None, names=COLS + ["target"], na_values="?")
        df["target"] = (df["target"] > 0).astype(int)
        return df
    except Exception as e:
        print("[!] Dataset unavailable (%s) - using synthetic data." % e)
        r = np.random.default_rng(SEED); n = 600
        df = pd.DataFrame({
            "age": r.normal(54, 9, n).clip(29, 77), "sex": r.integers(0, 2, n),
            "cp": r.integers(1, 5, n), "trestbps": r.normal(131, 17, n),
            "chol": r.normal(246, 51, n), "fbs": r.integers(0, 2, n),
            "restecg": r.integers(0, 3, n), "thalach": r.normal(150, 22, n),
            "exang": r.integers(0, 2, n), "oldpeak": r.gamma(1.2, 0.9, n),
            "slope": r.integers(1, 4, n), "ca": r.integers(0, 4, n),
            "thal": r.choice([3, 6, 7], n)})
        z = (0.04 * (df.age - 54) + 0.8 * (df.cp == 4) + 0.9 * df.exang + 0.6 * df.oldpeak
             + 0.8 * df.ca - 0.03 * (df.thalach - 150) + 0.9 * (df.thal == 7) - 0.6)
        df["target"] = (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)
        return df


df = load()
X, y = df[COLS], df["target"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def pipe(m):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), ("m", m)])


models = {"Logistic Regression": pipe(LogisticRegression(max_iter=2000)),
          "Random Forest": pipe(RandomForestClassifier(300, max_depth=5, random_state=SEED))}
best, best_auc, best_name = None, -1, ""
for name, m in models.items():
    cv = cross_val_score(m, Xtr, ytr, cv=5, scoring="roc_auc").mean()
    m.fit(Xtr, ytr)
    auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    print(f"{name}: CV AUC={cv:.3f}  Test AUC={auc:.3f}  "
          f"Acc={accuracy_score(yte, m.predict(Xte)):.3f}")
    if auc > best_auc:
        best, best_auc, best_name = m, auc, name

print(f"\nBest: {best_name}\n", classification_report(yte, best.predict(Xte)))
joblib.dump(best, "model.joblib")
print("Saved model.joblib")
