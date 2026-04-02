from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_late_prediction_valid_input():
    payload = {"is_holiday": 0,"freight_value": 200,"product_weight_g": 500}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "is_late_prediction" in data
