from pydantic import BaseModel


class CreditRiskRequest(BaseModel):

    Amount: float
    Value: float
    PricingStrategy: int
    FraudResult: int

    Recency: float
    Frequency: float
    Monetary: float


class CreditRiskResponse(BaseModel):

    risk_probability: float
    prediction: int