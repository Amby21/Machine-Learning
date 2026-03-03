#%% Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score
#%% Load dataset
df = pd.read_csv("housing.csv")
print(df.columns)
print(df.head(5))
#%% Normalizing columns -  
binary_cols = ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]

for col in binary_cols:
    df[col] = df[col].str.strip().str.lower().map({"yes": 1,"no":0})

#feature engineering
df["area_per_bedroom"] = df["area"]/df["bedrooms"]
df["bathroom_ratio"] = df["bathrooms"]/df["bedrooms"]
df["rooms_total"] = df["bedrooms"] + df["bathrooms"] + df["stories"]
df["amenity_score"] = df[["guestroom","basement","hotwaterheating","airconditioning"]].sum(axis=1)
df["log_area"] = np.log(df["area"])
df["log_price"] = np.log(df["price"])

#%%Features and target

X = df.drop(["log_price","price"],axis=1)
y = df["log_price"]

numeric_features = ["area","bedrooms","bathrooms","stories","parking","area_per_bedroom",
              "bathroom_ratio","rooms_total","amenity_score","log_area"]

categorical_features = ["furnishingstatus"]

preprocess = ColumnTransformer(
    transformers=[("num",StandardScaler(),numeric_features),("cat",OneHotEncoder(drop="first"),categorical_features)],remainder="passthrough"
)

model = Pipeline(steps=[("preprocess",preprocess),("regressor",RandomForestRegressor(random_state=42))
                        ])

#%%Hyperparameter search space

param_dist = {
    "regressor__n_estimators": [200,300,500,800],
    "regressor__max_depth": [10,20,30],
    "regressor__min_samples_split": [2,5,10],
    "regressor__min_samples_leaf": [1,2,4],
    "regressor__max_features":["sqrt",0.5]
}

search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter = 20,
    cv = 5,
    scoring = "neg_root_mean_squared_error",
    random_state = 42,
    n_jobs = 1,
    error_score='raise'
)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
#%%
search.fit(X_train,y_train)

best_model = search.best_estimator_
preds = best_model.predict(X_test)

rmse = mean_squared_error(y_test,preds,squared=False)
r2 = r2_score(y_test,preds)

print("Best Params:",search.best_params_)
print("RMSE:" ,rmse)
print("R2 Score", r2)


# %%
