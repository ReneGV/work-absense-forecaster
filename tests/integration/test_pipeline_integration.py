"""
Integration tests for the complete ML pipeline.

Tests the integration of:
- data_utils: Data loading and preparation
- preprocessors: Custom transformers
- train_model: Model training functions
- predict_model: Prediction pipeline
- evaluation: Metrics and reports
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

from src.data.data_utils import load_and_prepare_data
from src.models.preprocessors import (
    DropColumnsTransformer, 
    IQRClippingTransformer, 
    ToStringTransformer
)
from src.models.train_model import (
    load_and_prepare_data as train_load_data,
    create_preprocessing_pipeline,
    get_default_models,
    train_and_evaluate_model,
    train_models,
    save_model
)
from src.models.predict_model import (
    load_model,
    prepare_test_data,
    make_predictions,
    convert_to_labels,
    convert_to_binary,
    save_predictions
)
from src.models.evaluation import (
    evaluate_predictions,
    get_classification_report,
    create_confusion_matrix
)
from src.data.constants import NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

class TestCompleteEndToEndScenario:
    """Test a complete realistic end-to-end scenario."""
    
    def test_realistic_ml_workflow(self, sample_data_file, tmp_path):
        """
        Test realistic ML workflow:
        1. Load and prepare training data
        2. Train a model with preprocessing
        3. Save the model
        4. Load the model back
        5. Load new data for prediction
        6. Make predictions
        7. Evaluate if ground truth available
        """
        # ==== TRAINING PHASE ====
        # Step 1: Load training data
        df_full = load_and_prepare_data(sample_data_file, normalize_columns=True)
        
        # Step 2: Split into train/test
        X_train, X_test, y_train, y_test, median = train_load_data(
            sample_data_file,
            test_size=0.3,
            random_state=42
        )
        
        # Step 3: Create preprocessing pipeline
        columns_to_drop = ['id', 'body_mass_index']
        
        preprocess_pipeline = create_preprocessing_pipeline(
            columns_to_drop, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS
        )
        
        # Step 4: Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        full_pipeline = Pipeline([
            ('preprocess', preprocess_pipeline),
            ('regressor', model)
        ])
        full_pipeline.fit(X_train, y_train)
        
        # Step 5: Save model
        model_path = tmp_path / "deployed_model.pkl"
        save_model(full_pipeline, str(model_path))
        
        # ==== DEPLOYMENT/PREDICTION PHASE ====
        # Step 6: Load model (simulating production environment)
        loaded_model = load_model(str(model_path))
        
        # Step 7: Prepare new data for prediction
        new_data = prepare_test_data(
            df_full,
            sample_size=15,
            random_state=123
        )
        
        # Step 8: Make predictions
        predictions = make_predictions(loaded_model, new_data)
        prediction_labels = convert_to_labels(predictions)
        
        # Step 9: Save predictions
        new_data['predicted_absenteeism'] = prediction_labels
        predictions_path = tmp_path / "final_predictions.csv"
        save_predictions(new_data, str(predictions_path))
        
        # ==== EVALUATION PHASE ====
        # Step 10: Evaluate predictions (if ground truth available)
        if 'absenteeism_real' in new_data.columns:
            y_true = convert_to_binary(new_data['absenteeism_real'])
            
            # Calculate metrics
            metrics = evaluate_predictions(y_true, predictions)
            assert all(metric_name in metrics for metric_name in ['accuracy', 'f1_score', 'recall'])
            
            # Generate report
            report = get_classification_report(y_true, predictions)
            assert len(report) > 0
            
            # Create confusion matrix
            cm_path = tmp_path / "final_confusion_matrix.png"
            create_confusion_matrix(y_true, predictions, str(cm_path))
            assert os.path.exists(cm_path)
        
        # ==== FINAL VERIFICATION ====
        # Verify all artifacts were created
        assert os.path.exists(model_path)
        assert os.path.exists(predictions_path)
        
        # Verify predictions file content
        saved_predictions = pd.read_csv(predictions_path)
        assert len(saved_predictions) == 15
        assert 'predicted_absenteeism' in saved_predictions.columns
        assert all(label in ['High', 'Low'] for label in saved_predictions['predicted_absenteeism'])

