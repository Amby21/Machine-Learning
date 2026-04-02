import mlflow.sklearn
import pandas as pd

class LogisticEngine:
    def __init__(self,model_uri):
        self.model = mlflow.sklearn.load_model(model_uri)
        self.baseline_weigh_mean = 2100.0

    def predict(self,data:dict):
        df = pd.DataFrame([data])

        drift = data['product_weight_g'] > (self.baseline_weigh_mean* 5)
        
        prediction = self.model.predict(df)[0]
        return int(prediction),drift