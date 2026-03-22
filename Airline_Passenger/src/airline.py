#%%
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error
def load_data():
    url = "s3://ml-lstm-demo-skandan/airline_passenger.csv"
    df = pd.read_csv(url)
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month',inplace=True)
    return df

df = load_data()

scaler = MinMaxScaler()
#%%
data = scaler.fit_transform(df.values)
#%%
def create_sequences(data, seq_length = 12):
    X, y = [],[]
    print(len(data),seq_length)
    for i in range(len(data)- seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X),np.array(y)

X,y = create_sequences(data)

#%% Train,test split
split = int(len(X)*0.8)
X_train,X_test = X[:split],X[split:]
y_train,y_test = y[:split],y[split:]


#%%
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size = 1, hidden_size=50,num_layers=2):
        super(LSTMModel,self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size,num_layers,batch_first = True)
        self.fc = nn.Linear(hidden_size,1)
    
    def forward(self,x):
        out, _ = self.lstm(x)
        out = out[:,-1,:]
        out = self.fc(out)
        return out
    
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

#%% Training ML Tracking
import mlflow
import mlflow.pytorch
# ARTIFACT_ROOT = "./mlruns"
# mlflow.set_tracking_uri(f"file:{ARTIFACT_ROOT}")
import os
import mlflow

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

mlflow.set_tracking_uri(f"file:{os.path.join(PROJECT_ROOT, 'mlruns')}")
mlflow.set_experiment("LSTM_Timeseries")

X_train_t = torch.tensor(X_train,dtype=torch.float32)
y_train_t = torch.tensor(y_train,dtype=torch.float32)


with mlflow.start_run():
    print("RUN STARTED")
    for epoch in range(20):
        model.train()

        outputs = model(X_train_t)
        loss = criterion(outputs,y_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch},Loss:{loss.item()}")
        mlflow.log_metric("loss",loss.item(),step=epoch)
    mlflow.pytorch.log_model(model,"lstm_model")
 
model.eval()
X_test_t = torch.tensor(X_test,dtype=torch.float32)
preds = model(X_test_t).detach().numpy().reshape(-1, 1)

preds = scaler.inverse_transform(preds)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y_test_inv,preds))

print("RMSE",rmse)


# %%
