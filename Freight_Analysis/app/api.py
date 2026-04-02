from fastapi import FastAPI,HTTPException
from app.schemas import FreightRequest, PredictionResponse
from app.engine import LogisticEngine
import os

app = FastAPI()
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
model_path = os.path.join(project_root,"models", "logistic-model")

engine = LogisticEngine(model_uri = model_path)

@app.post("/predict",response_model = PredictionResponse)
async def predict_freight(payload: FreightRequest):
    pred, drift = engine.predict(payload.model_dump())
    return {
        "is_late_prediction": pred,
        "model_version": "v1.0-balanced",
        "drift_warning": drift
    }
