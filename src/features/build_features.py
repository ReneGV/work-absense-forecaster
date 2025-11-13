import argparse
import logging
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Cargando: {path}")
    df = pd.read_csv(path)
    df.columns = [x.lower().replace(" ", "_").replace("/", "_") for x in df.columns]
    logger.info(f"Shape: {df.shape}")
    return df

# TRANSFORMERS
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols): self.cols = cols
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        drop = [c for c in self.cols if c in X.columns]
        return X.drop(drop, axis=1) if drop else X
    def get_feature_names_out(self, input_features=None):
        if input_features is None: raise ValueError("input_features required")
        return [f for f in input_features if f not in self.cols]

class IQRClippingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self): self.bounds_ = {}
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            q1, q3 = X[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.bounds_[col] = (q1 - 1.5*iqr, q3 + 1.5*iqr)
        return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col, (lo, hi) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lo, hi)
        return X.values
    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else list(self.bounds_.keys())

def main(args):
    df = load_data(args.input)

    # TARGET: Guardamos antes
    target = df['absenteeism_time_in_hours']
    theta = target.median()
    y = (target > theta).astype(int)

    # COLUMNAS
    drop_cols = ['id', 'body_mass_index', 'absenteeism_time_in_hours']  # ← DROP TARGET
    num_cols  = ['transportation_expense', 'distance_from_residence_to_work', 'service_time', 'age',
                 'work_load_average_day', 'hit_target', 'son', 'pet', 'weight', 'height']
    cat_cols  = ['reason_for_absence', 'month_of_absence', 'day_of_the_week', 'seasons',
                 'disciplinary_failure', 'education', 'social_drinker', 'social_smoker']

    # PIPELINE SIN REMAINDER
    pipeline = Pipeline([
        ('drop', DropColumnsTransformer(drop_cols)),
        ('preprocess', ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('clip', IQRClippingTransformer()),
                ('scaler', StandardScaler())
            ]), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ]), cat_cols)
        ], remainder='drop'))  # ← NADA PASA
    ])

    X = pipeline.fit_transform(df)
    ohe = pipeline.named_steps['preprocess'].named_transformers_['cat']['ohe']
    cat_features = ohe.get_feature_names_out(cat_cols)
    all_features = num_cols + list(cat_features)

    df_ready = pd.DataFrame(X, columns=all_features)
    df_ready['high_absenteeism'] = y.values

    logger.info(f"¡PERFECTO! {df_ready.shape[1]-1} features + target")
    logger.info(f"Positivos: {y.mean():.2%}")
    logger.info(f"Missing: {df_ready.isna().sum().sum()}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_ready.to_csv(args.output, index=False)
    logger.info(f"GUARDADO → {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed/features_ready.csv")
    args = parser.parse_args()
    main(args)