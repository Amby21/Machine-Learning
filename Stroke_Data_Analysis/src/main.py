#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import mlflow
import os
from sklearn.metrics import recall_score, f1_score, roc_auc_score

# %%
data = pd.read_csv('stroke_data.csv')

# %%
data.describe()
data.head(5)
# %%
sns.countplot(data=data,x='stroke')
#Unbalanced data, accuracy may not be the right metric, F-score or Recall
#%%
data['Residence_type'].value_counts()
data['work_type'].value_counts()
#%%
data.isnull().sum()
sns.heatmap(data=data.corr(numeric_only=True),annot=True,cmap='viridis')

# %%
#  Binary mapping for already binary columns
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
data['ever_married'] = data['ever_married'].map({'Yes': 1, 'No': 0})
data['Residence_type'] = data['Residence_type'].map({'Urban': 1, 'Rural': 0})

# One-hot encoding for multi-category columns only
data = pd.get_dummies(
    data, 
    columns=['smoking_status', 'work_type'],  # only these
    drop_first=True
)

# Optional: convert any boolean dummies to int
bool_cols = data.select_dtypes(include='bool').columns
data[bool_cols] = data[bool_cols].astype(int)
# %%
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
#%%
X = data.drop('stroke',axis=1)
y = data['stroke']
#%%
scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
X_train_scaled = scaler.fit_transform(X_train)
# y_train_scaled = scaler.fit_transform(y_train)
X_test_scaled = scaler.transform(X_test)

#%%
mlflow.set_tracking_uri(f"file:{os.path.join(PROJECT_ROOT, 'mlruns')}")
mlflow.set_experiment("stroke prediction")

def train_and_log(model,name,X_tr,X_te):
    with mlflow.start_run(run_name=name):
        print("RUN STARTED")
        model.fit(X_tr,y_train)

        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:,1]

        recall = recall_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        roc_auc = roc_auc_score(y_test,y_pred)

        #Log Parameter
        mlflow.log_param("model",name)

        #Log Metrics
        mlflow.log_metric("recall",recall)
        mlflow.log_metric("f1_score",f1)
        mlflow.log_metric("roc_auc",roc_auc)

        #log model
        mlflow.sklearn.log_model(model,name)

        print(f"{name} done")

#%%
#Logistic Regression
lr = LogisticRegression(class_weight="balanced",max_iter = 1000)
train_and_log(lr,"Logistic Regression",X_train_scaled,X_test_scaled)
#%%
# RandomForest Regression
rf = RandomForestClassifier(n_estimators=100,class_weight="balanced",random_state=42)
train_and_log(lr,"Random Forest",X_train,X_test)
