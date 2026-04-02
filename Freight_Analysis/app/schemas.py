from pydantic import BaseModel, Field

class FreightRequest(BaseModel):
    is_holiday: int = Field(..., ge=0, le=1)
    freight_value: float = Field(..., ge=0)
    product_weight_g: float =  Field(..., gt=0, description="weight must be positive number")
    
    

class PredictionResponse(BaseModel):
    is_late_prediction: int
    model_version: str
    drift_warning: bool