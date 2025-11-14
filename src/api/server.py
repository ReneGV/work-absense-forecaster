from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import uvicorn
import os

# Get the absolute path to the model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_absenteeism_model.pkl")

# Load the saved model
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Warning: Model not found at {MODEL_PATH}")
    model = None

# Define the input data format for prediction
class AbsenteeismData(BaseModel):
    """Input data schema for absenteeism prediction."""
    reason_for_absence: int = Field(..., description="Reason code for absence (1-28)")
    month_of_absence: int = Field(..., ge=0, le=12, description="Month of absence (0-12)")
    day_of_the_week: int = Field(..., ge=2, le=6, description="Day of the week (2-6)")
    seasons: int = Field(..., ge=1, le=4, description="Season (1-4)")
    transportation_expense: int = Field(..., description="Transportation expense")
    distance_from_residence_to_work: int = Field(..., description="Distance from residence to work (km)")
    service_time: int = Field(..., description="Service time (years)")
    age: int = Field(..., description="Age of employee")
    work_load_average_day: float = Field(..., alias="work_load_average/day", description="Work load average per day")
    hit_target: int = Field(..., ge=0, le=100, description="Hit target percentage")
    disciplinary_failure: int = Field(..., ge=0, le=1, description="Disciplinary failure (0 or 1)")
    education: int = Field(..., description="Education level (1-4)")
    son: int = Field(..., ge=0, description="Number of sons")
    social_drinker: int = Field(..., ge=0, le=1, description="Social drinker (0 or 1)")
    social_smoker: int = Field(..., ge=0, le=1, description="Social smoker (0 or 1)")
    pet: int = Field(..., ge=0, description="Number of pets")
    weight: int = Field(..., description="Weight (kg)")
    height: int = Field(..., description="Height (cm)")
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "reason_for_absence": 23,
                "month_of_absence": 7,
                "day_of_the_week": 3,
                "seasons": 1,
                "transportation_expense": 289,
                "distance_from_residence_to_work": 36,
                "service_time": 13,
                "age": 33,
                "work_load_average/day": 239.554,
                "hit_target": 97,
                "disciplinary_failure": 0,
                "education": 1,
                "son": 2,
                "social_drinker": 1,
                "social_smoker": 0,
                "pet": 1,
                "weight": 90,
                "height": 172
            }
        }

# Initialize FastAPI app
app = FastAPI(
    title="Absenteeism Prediction API",
    description="API for predicting work absenteeism based on employee data",
    version="1.0.0"
)

# Define prediction endpoint
@app.post("/predict", summary="Predict absenteeism level")
def predict(data: AbsenteeismData):
    """
    Predict whether an employee will have high or low absenteeism.
    
    Returns:
        - prediction: Binary prediction (0 = Low, 1 = High)
        - prediction_label: Human-readable label ('Low' or 'High')
        - confidence: Prediction probability (if available)
    """
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please ensure the model file exists."
        )
    
    try:
        # Convert input data to DataFrame
        input_dict = data.dict(by_alias=True)
        input_df = pd.DataFrame([input_dict])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_label = "High" if prediction == 1 else "Low"
        
        # Get probability if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_df)[0]
            confidence = float(proba[prediction])
        
        response = {
            "prediction": int(prediction),
            "prediction_label": prediction_label,
        }
        
        if confidence is not None:
            response["confidence"] = round(confidence, 4)
        
        return response
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

# Health check endpoint
@app.get("/health")
def health_check():
    """Check if the API and model are ready."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }

# Define a root endpoint
@app.get("/")
def read_root():
    """Root endpoint with API information."""
    return {
        "message": "Work Absenteeism Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
