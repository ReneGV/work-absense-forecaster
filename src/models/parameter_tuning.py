import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from src.models.preprocessors import DropColumnsTransformer, IQRClippingTransformer, ToStringTransformer
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
import os
from itertools import product
import itertools
import random
from sklearn.model_selection import ParameterSampler

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
# 4. Preprocessing Builder
# -------------------------------------------------
def build_preprocess():
    return Pipeline([
        ("drop", DropColumnsTransformer(cols_drop)),
        ("transform", ColumnTransformer([
            ("num", Pipeline([
                ("iqr", IQRClippingTransformer()),
                ("scale", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("str", ToStringTransformer()),
                ("ohe", OneHotEncoder(sparse=False, handle_unknown="ignore"))
            ]), cat_cols),
        ], remainder="passthrough"))
    ])

# -------------------------------------------------
# 5. Parameter Grid
# -------------------------------------------------
param_dist = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__fit_intercept': [True, False],
    'clf__class_weight': [None, 'balanced']
}
# -------------------------------------------------
# 6. Tuning Pipeline
# -------------------------------------------------
pipe = Pipeline([
    ("preprocess", build_preprocess()),
    ("clf", LogisticRegression(max_iter=1000))
])

search = GridSearchCV(
    pipe, param_dist, cv=5, scoring="f1", n_jobs=-1, verbose=1
)
# -------------------------------------------------
# 7. Run with MLflow
# -------------------------------------------------
param_grid = list(ParameterSampler(param_dist, n_iter=16, random_state=42))

with mlflow.start_run(run_name="LogisticRegressionClassifier_Tuning") as parent_run:
    best_f1 = -1.0
    best_params = None
    best_model = None

    for idx, params in enumerate(param_grid, 1):
        # ---- child run for each trial ----
        with mlflow.start_run(run_name=f"Trial_{idx:02d}", nested=True) as child_run:
            # Log the sampled hyper-parameters
            mlflow.log_params(params)

            # Fit the pipeline with the current set of parameters
            pipe.set_params(**params)
            pipe.fit(X_train, y_train_bin)

            # ---- predictions & metrics ----
            y_pred_train = pipe.predict(X_train)
            y_pred_test  = pipe.predict(X_test)

            train_f1 = f1_score(y_train_bin, y_pred_train)
            test_f1  = f1_score(y_test_bin, y_pred_test)

            train_acc = accuracy_score(y_train_bin, y_pred_train)
            test_acc  = accuracy_score(y_test_bin, y_pred_test)

            # Log metrics
            mlflow.log_metric("train_f1", train_f1)
            mlflow.log_metric("test_f1",  test_f1)
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("test_accuracy",  test_acc)

            # ---- confusion matrix (as artifact) ----
            cm = confusion_matrix(y_test_bin, y_pred_test)
            cm_fig, ax = plt.subplots()
            ConfusionMatrixDisplay(cm, display_labels=["≤ median", "> median"]).plot(ax=ax, cmap="Blues")
            cm_path = f"confusion_matrix_trial_{idx}.png"
            cm_fig.savefig(cm_path)
            plt.close(cm_fig)
            mlflow.log_artifact(cm_path, artifact_path="confusion_matrices")
            os.remove(cm_path)   # clean up

            # ---- log the model itself ----
            mlflow.sklearn.log_model(pipe, artifact_path="model")

            # ---- keep track of the best model ----
            if test_f1 > best_f1:
                best_f1 = test_f1
                best_params = params
                best_model = pipe
                mlflow.set_tag("best_trial", child_run.info.run_id)

    # -------------------------------------------------
    # 8. Register the best model
    # -------------------------------------------------
    mlflow.log_metric("best_test_f1", best_f1, step=0)
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

    # Log the final best model under a dedicated name
    model_name = "LogisticRegressionClassifier_Best"
    mlflow.sklearn.log_model(best_model, artifact_path="best_model")
    # Optional: register it in the Model Registry
    # result = mlflow.register_model(f"runs:/{parent_run.info.run_id}/best_model", model_name)

    print("\n=== Tuning finished ===")
    print(f"Best test F1 : {best_f1:.4f}")
    print(f"Best params  : {best_params}")