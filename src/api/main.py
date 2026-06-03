from fastapi import FastAPI
import mlflow.pyfunc

from pydantic_models import (
    CreditRiskRequest,
    CreditRiskResponse
)

import pandas as pd


app = FastAPI(
    title="Credit Risk API"
)


MODEL_NAME = "CreditRiskBestModel"

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/latest"
)


@app.get("/")
def home():

    return {
        "message":
        "Credit Risk Model API Running"
    }


@app.post(
    "/predict",
    response_model=CreditRiskResponse
)
def predict(
    request: CreditRiskRequest
):

    data = pd.DataFrame(
        [request.dict()]
    )

    probability = float(
        model.predict(data)[0]
    )

    prediction = (
        1
        if probability >= 0.5
        else 0
    )

    return CreditRiskResponse(
        risk_probability=probability,
        prediction=prediction
    )