import pandas as pd
import joblib
import mlflow
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score
from src.models.preprocessors import DropColumnsTransformer, IQRClippingTransformer, ToStringTransformer
import matplotlib.pyplot as plt


# ============================
# Configurar MLflow
# ============================
mlflow.set_experiment("absenteeism_predictions")

with mlflow.start_run():  # iniciar run

    # ============================
    # Cargar dataset original y generar datos de prueba
    # ============================
    df = pd.read_csv('data/raw/work_absenteeism_original.csv')
    df.columns = df.columns.str.lower().str.replace("[ ]", "_", regex=True)

    # Guardar columna real para evaluación
    df_new = df.drop(columns=['id', 'body_mass_index'])
    df_new['absenteeism_real'] = df_new['absenteeism_time_in_hours']  # columna para evaluación
    df_new = df_new.drop(columns=['absenteeism_time_in_hours'])

    # Tomar 10 muestras aleatorias como nuevos datos
    df_new = df_new.sample(25, random_state=42)
    df_new.to_csv('data/raw/new_absenteeism_data.csv', index=False)

    # ============================
    # Cargar modelo
    # ============================
    model_path = 'src/models/best_absenteeism_model.pkl'
    model = joblib.load(model_path)

    # ============================
    # Hacer predicciones
    # ============================
    new_data = pd.read_csv('data/raw/new_absenteeism_data.csv')
    predictions = model.predict(new_data.drop(columns=['absenteeism_real'], errors='ignore'))  # No usar columna real
    prediction_labels = ['High' if p == 1 else 'Low' for p in predictions]
    new_data['predicted_absenteeism'] = prediction_labels

    # ============================
    # Guardar predicciones
    # ============================
    output_path = 'data/predictions/absenteeism_predictions.csv'
    new_data.to_csv(output_path, index=False)
    mlflow.log_artifact(output_path)
    print("\n Predicciones guardadas y logueadas en MLflow")

    # ============================
    # Evaluación si existe columna real
    # ============================
    if 'absenteeism_real' in new_data.columns:
        median_value = new_data['absenteeism_real'].median()
        y_true = (new_data['absenteeism_real'] > median_value).astype(int)

        acc = accuracy_score(y_true, predictions)
        mlflow.log_metric("accuracy", acc)

        f1 = f1_score(y_true, predictions)
        mlflow.log_metric("f1_score", f1)

        recall = recall_score(y_true, predictions)
        mlflow.log_metric("recall", recall)

        print(f"\nAccuracy: {acc:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_true, predictions))

        # Matriz de confusión
        cm = confusion_matrix(y_true, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'High'])
        disp.plot(cmap='Blues')
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")
