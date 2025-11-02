# linear_regression_classifier_tuning.py
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LinearRegression
from src.models.preprocessors import DropColumnsTransformer, IQRClippingTransformer, ToStringTransformer
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
import os
from itertools import product

# -------------------------------------------------
# 1. MLflow Setup
# -------------------------------------------------
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("linear_regression_as_classifier")

# -------------------------------------------------
# 2. Load & Prepare Data (same as original)
# -------------------------------------------------
df = pd.read_csv("data/raw/work_absenteeism_original.csv")
target_col = "absenteeism_time_in_hours"
df.columns = df.columns.str.lower().str.replace(r"\s+", "_", regex=True)

cols_drop = ["id", "body_mass_index"]
num_cols = [
    "transportation_expense", "distance_from_residence_to_work", "service_time",
    "age", "work_load_average/day", "hit_target", "son", "pet", "weight",
    "height", "education",
]
cat_cols = [
    "disciplinary_failure", "social_drinker", "social_smoker",
    "month_of_absence", "day_of_the_week", "seasons", "reason_for_absence",
]

X = df.drop(columns=[target_col])
y = df[target_col]
median_thr = y.median()
y_bin = (y > median_thr).astype(int)

X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_bin)
_, _, y_train_bin, y_test_bin = train_test_split(X, y_bin, test_size=0.2, random_state=42, stratify=y_bin)

# -------------------------------------------------
# 3. Classifier Wrapper
# -------------------------------------------------
class LinearRegressionClassifier(LinearRegression):
    def __init__(self, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
    def predict(self, X):
        return (super().predict(X) > self.threshold).astype(int)

# -------------------------------------------------
# 4. Preprocessing Builder
# -------------------------------------------------
def build_preprocess(scaler="standard", iqr_factor=1.5):
    scalers = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "none": "passthrough"}
    return Pipeline([
        ("drop", DropColumnsTransformer(cols_drop)),
        ("transform", ColumnTransformer([
            ("num", Pipeline([
                ("iqr", IQRClippingTransformer(factor=iqr_factor)),
                ("scale", scalers[scaler])
            ]), num_cols),
            ("cat", Pipeline([
                ("str", ToStringTransformer()),
                ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
            ]), cat_cols),
        ], remainder="passthrough"))
    ])

# -------------------------------------------------
# 5. Parameter Grid
# -------------------------------------------------
param_dist = {
    "preprocess": [build_preprocess(s, f) for s, f in product(
        ["standard", "minmax", "none"], [1.0, 1.5, 2.0]
    )],
    "clf__fit_intercept": [True, False],
    "clf__positive": [True, False],
    "clf__n_jobs": [1, -1],
    "clf__threshold": [0.4, 0.45, 0.5, 0.55, 0.6],
}

# -------------------------------------------------
# 6. Tuning Pipeline
# -------------------------------------------------
pipe = Pipeline([
    ("preprocess", build_preprocess()),
    ("clf", LinearRegressionClassifier())
])

search = RandomizedSearchCV(
    pipe, param_dist, n_iter=50, cv=5, scoring="f1", random_state=42, n_jobs=-1, verbose=1
)

# -------------------------------------------------
# 7. Run with MLflow
# -------------------------------------------------
with mlflow.start_run(run_name="LinearRegression_Classifier_Tuning"):
    search.fit(X_train, y_train_bin)
    best = search.best_estimator_
    pred = best.predict(X_test)

    f1 = f1_score(y_test_bin, pred)
    acc = accuracy_score(y_test_bin, pred)

    # Extract config
    proc = best.named_steps["preprocess"]
    iqr = proc.named_steps["transform"].named_transformers_["num"].named_steps["iqr"].factor
    scaler = proc.named_steps["transform"].named_transformers_["num"].named_steps["scale"]
    scaler_name = scaler.__class__.__name__ if scaler != "passthrough" else "none"
    thr = best.named_steps["clf"].threshold

    # Log
    mlflow.log_param("scaler", scaler_name)
    mlflow.log_param("iqr_factor", iqr)
    mlflow.log_param("threshold", thr)
    mlflow.log_params({k.replace("clf__", ""): v for k, v in search.best_params_.items() if k.startswith("clf__")})
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("accuracy", acc)

    # Confusion matrix
    cm = confusion_matrix(y_test_bin, pred)
    plt.figure(figsize=(5,4))
    ConfusionMatrixDisplay(cm, display_labels=["Low","High"]).plot(cmap="Blues")
    plt.title(f"LinearReg Classifier (F1={f1:.3f})")
    plt.tight_layout()
    plt.savefig("cm.png")
    plt.close()
    mlflow.log_artifact("cm.png")

    mlflow.sklearn.log_model(best, "model")
    mlflow.set_tag("best", "true")

    print(f"\nBest F1: {f1:.4f} | Acc: {acc:.4f} | Threshold: {thr} | Scaler: {scaler_name}")

# -------------------------------------------------
# 8. Save locally
# -------------------------------------------------
joblib.dump(best, "best_linear_regression_classifier.pkl")