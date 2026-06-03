from fastapi import FastAPI
import mlflow.pyfunc

import pandas as pd
import os

# Import from the same directory
try:
    from .pydantic_models import CreditRiskRequest,CreditRiskResponse
except ImportError:
    from src.api.pydantic_models import CreditRiskRequest, CreditRiskResponse

app = FastAPI(title="Credit Risk API")

# Configuration
MODEL_NAME = "CreditRiskBestModel"
# If running locally without an MLflow server, change this to a local path!
MODEL_URI = f"models:/{MODEL_NAME}/latest" 


model = mlflow.pyfunc.load_model(

    r"notebooks/mlruns/1/models/m-743f98a7f3dc4fb2adac03fd108e1f32/artifacts"

)

@app.get("/")
def home():
    return {"message": "Credit Risk Model API Running"}

@app.post("/predict", response_model=CreditRiskResponse)
def predict(request: CreditRiskRequest):
    # Convert Pydantic model to DataFrame
    # model_dump() is the modern version of .dict()
    data = pd.DataFrame([request.model_dump()])

    # Get probability from the model
    probability = float(model.predict(data)[0])

    # Classify based on 0.5 threshold
    prediction = 1 if probability >= 0.5 else 0

    return CreditRiskResponse(
        risk_probability=probability,
        prediction=prediction
    )