import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from src.models.preprocessors import DropColumnsTransformer, IQRClippingTransformer, ToStringTransformer


# ============================
# Carga de datos y preprocesamiento  
# ============================
df = pd.read_csv('data/raw/work_absenteeism_original.csv')

target_column = 'absenteeism_time_in_hours'
df.columns = df.columns.str.lower().str.replace("[ ]", "_", regex=True)

columns_to_drop = ['id', 'body_mass_index']
numerical_columns = [
    'transportation_expense', 'distance_from_residence_to_work', 'service_time',
    'age', 'work_load_average/day', 'hit_target', 'son', 'pet', 'weight',
    'height', 'education'
]
categorical_columns = [
    'disciplinary_failure', 'social_drinker', 'social_smoker',
    'month_of_absence', 'day_of_the_week', 'seasons', 'reason_for_absence'
]

X = df.drop(target_column, axis=1)
y = df[target_column]

median_absentism_value = y.median()
print(f"Median of clipped training target: {median_absentism_value}")
y = (y > median_absentism_value).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================
# Pipeline de preprocesamiento
# ============================
preprocess_pipeline = Pipeline([
    ('drop_columns', DropColumnsTransformer(columns_to_drop)),
    ('preprocess', ColumnTransformer(
        transformers=[
            ('numerical', Pipeline([
                ('iqr_clipping', IQRClippingTransformer()),
                ('scaling', StandardScaler())
            ]), numerical_columns),
            ('categorical', Pipeline([
                ('to_string', ToStringTransformer()),
                ('one_hot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ]), categorical_columns)
        ],
        remainder='passthrough'
    ))
])

# ============================
# Entrenamiento de modelo y evaluación
# ============================
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest Classifier': RandomForestClassifier(random_state=42, n_estimators=100),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                                    alpha=0.01, max_iter=1000, random_state=42, early_stopping=True)
}

best_model = None
best_accuracy = 0

for model_name, model in models.items():
    full_pipeline = Pipeline([
        ('preprocess', preprocess_pipeline),
        ('regressor', model)
    ])

    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nResults for {model_name}:")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'High'])
    disp.plot(cmap='Blues')
    plt.title(f'{model_name}: Confusion Matrix')
    plt.tight_layout()
    plt.show()

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = full_pipeline

# ============================
# Guardamos el mejor modelo
# ============================
joblib.dump(best_model, 'best_absenteeism_model.pkl')
print(f"\n✅ Best model saved with accuracy: {best_accuracy:.2f}")