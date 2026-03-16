#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%% hyperparameter tuning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
import joblib
import mlflow   
import mlflow.sklearn
#%%
# stores.csv

# This file contains anonymized information about the 45 stores, indicating the type and size of store.

# train.csv

# This is the historical training data, which covers to 2010-02-05 to 2012-11-01. Within this file you will find the following fields:

# Store - the store number
# Dept - the department number
# Date - the week
# Weekly_Sales -  sales for the given department in the given store
# IsHoliday - whether the week is a special holiday week
# Type A: Sizes from 39.690 to 219.622
# Type B: Sizes from 34.875 to 140.167
# Type C: Sizes from 39.690 to 42.988
# test.csv

# This file is identical to train.csv, except we have withheld the weekly sales. You must predict the sales for each triplet of store, department, and date in this file.

# features.csv

# This file contains additional data related to the store, department, and regional activity for the given dates. It contains the following fields:

# Store - the store number
# Date - the week
# Temperature - average temperature in the region
# Fuel_Price - cost of fuel in the region
# MarkDown1-5 - anonymized data related to promotional markdowns that Walmart is running. MarkDown data is only available after Nov 2011, and is not available for all stores all the time. Any missing value is marked with an NA.
# CPI - the consumer price index
# Unemployment - the unemployment rate
# IsHoliday - whether the week is a special holiday week

# %% EDA
# df_feat.head(5)
# df_stores.head(5)
# df_test.head(5)
# df_train.head(5)

#%%

# # %%
# plt.figure(figsize=(12,8))
# data_train.groupby('Date')["Weekly_Sales"].sum().plot()
# plt.show()

# # %%
# sns.boxplot(x='IsHoliday_x',y='Weekly_Sales',data=data_train)
# plt.title("Sales Distribution on Holiday vs Non-Holiday")
# plt.show()
# # %%
# plt.figure(figsize=(10, 8))
# sns.boxplot(data=data_train, x='Type', y='Weekly_Sales')

# plt.title('Sales Distribution Spread by Store Type')

# # %%
# plt.scatter(data_train["Temperature"], data_train["Weekly_Sales"], alpha=0.6)
# plt.xlabel("Temperature")
# plt.ylabel("Weekly Sales")
# plt.title("Temperature vs Sales")
# plt.show()
# #%%
# plt.figure(figsize=(10, 8))
# sns.barplot(y=data_train.groupby('Dept')['Weekly_Sales'].mean(),x='Dept', data=data_train)
# plt.show()

# %%
def preprocess(df):
    # 1. Convert the column to datetime objects
    df["Date"] = pd.to_datetime(df["Date"])

    # 2. Now you can use the .dt accessor
    df["week"] = df["Date"].dt.isocalendar().week
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
 # %%
    df.isna().sum()
 # %%
    df['MarkDown1'].value_counts()

# %%
    markdown_cols = ["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5"]
#%%
    df["total_markdown"] = df[markdown_cols].fillna(0).sum(axis=1)
    dummies = pd.get_dummies(df['Type'],drop_first=True)
    df = pd.concat([df.drop(['Type'],axis=1),dummies],axis=1)
    df = df['IsHoliday_x'].map({'True':1,'False':0})
# %%
    df["IsHoliday"] = df["IsHoliday_x"]
    df.drop(["IsHoliday_x","IsHoliday_y"], axis=1, inplace=True)
    return df



# %% Lag Features of Random Forest Regressor
def lag_features(df):
    df.sort_values(['Store','Dept','Date'])

    df['lag1'] = df.groupby(['Store','Dept'])['Weekly_Sales'].shift(1)
    df['lag2'] = df.groupby(['Store','Dept'])['Weekly_Sales'].shift(2)
    df['lag4'] = df.groupby(['Store','Dept'])['Weekly_Sales'].shift(4)
    df['lag12'] = df.groupby(['Store','Dept'])['Weekly_Sales'].shift(12)
    df['rolling_mean_4'] = df.groupby(['Store','Dept'])['Weekly_Sales'].rolling(4).mean().reset_index(level=[0,1],drop=True)
    df["rolling_std_4"] = df.groupby(["Store","Dept"])["Weekly_Sales"].rolling(4).std().reset_index(level=[0,1],drop=True)
    df = df.fillna(0)
    df = df.dropna()
    return df


#%%

#%%
#%%
def train_model():
    df_feat = pd.read_csv('features - Walmart Sales Forecast.csv')
    df_stores = pd.read_csv('stores - Walmart Sales Forecast.csv')
    df_train = pd.read_csv('train - Walmart Sales Forecast.csv')
#%%
    data_train = df_train.merge(df_feat,on=['Store','Date'],how='left')
    data_train = data_train.merge(df_stores,on="Store",how='left')
    df_train = preprocess(df_train)
    df_train = lag_features(df_train)
#%%
    features = ['Store', 'Dept', 'Weekly_Sales', 'Temperature', 'Fuel_Price','CPI',
        'Unemployment', 'Size', 'week', 'month', 'year', 'lag1', 'lag2', 'lag4',
        'lag12', 'rolling_mean_4', 'rolling_std_4', 'total_markdown', 'B', 'C',
        'IsHoliday']
    X = data_train[features]
    y = data_train['Weekly_Sales']

    param_grid = {"n_estimators":[100,200,500],
              "max_depth":[8,12,15],
              "min_samples_split":[2,5],
              "min_smaples_leaf": [1,2]
              }
    rf = RandomForestRegressor(random_state = 42,n_jobs=-1)
    tscv = TimeSeriesSplit(n_splits=5)
    random_search = RandomizedSearchCV(rf,param_distributions=param_grid,vebose=1,cv=tscv,scoring='neg_mean_absolute_error',n_iter=10)
    random_search.fit(X,y)
    best_model = random_search.best_estimator_
    # Final Model
    final_model = RandomForestRegressor(**best_model.get_params())
    final_model.fit(X, y)
    joblib.dump(final_model,"final_sales_model.pkl")
    print("Model training complete and saved as final_sales_model.pkl")
    #%%
    # Log with mlflow
    mlflow.start_run()
    mlflow.sklearn.log_model(best_model,"rf_sales_model")
    mlflow.log_params(best_model,get_params())
    mlflow.end_run()
#%%
def predict_model():
    final_model = joblib.load("final_sales_model.pkl")
    df_test = pd.read_csv('test - Walmart Sales Forecast.csv')
    df_feat = pd.read_csv('features - Walmart Sales Forecast.csv')
    df_stores = pd.read_csv('stores - Walmart Sales Forecast.csv')
    data_test = df_test.merge(df_feat,on=['Store','Date'],how='left')
    data_test = data_test.merge(df_stores,on="Store",how='left')

    data_test = preprocess(data_test)
    data_train, data_test = data_train.align(data_test, join="left", axis=1, fill_value=0)
# %%
    features = ['Store', 'Dept', 'Weekly_Sales', 'Temperature', 'Fuel_Price','CPI',
        'Unemployment', 'Size', 'week', 'month', 'year', 'lag1', 'lag2', 'lag4',
        'lag12', 'rolling_mean_4', 'rolling_std_4', 'total_markdown', 'B', 'C',
        'IsHoliday']
#%%
# Predict on test data
    X_test = data_test[features]
    predictions = final_model.predict(X_test)

    df_test["Weekly_Sales"] = predictions

    submission = df_test[["Store","Dept","Date","Weekly_Sales"]]

    submission.to_csv("submission.csv", index=False)

#%%

# mae_scores = []
#%%
    # for train_idx, val_idx in tscv.split(X):
    #     X_train, X_val = X.iloc[train_idx],X.iloc[val_idx]
    #     y_train, y_val = y.iloc[train_idx],y.iloc[val_idx]

    #     model = RandomForestRegressor(n_estimators=200,max_depth=15,random_state=42,n_jobs=-1)
        
    #     model.fit(X_train,y_train)
    #     preds = model.predict(X_val)
    #     mae = mean_absolute_error(preds,y_val)
    #     mae_scores.append(mae)
    #     print("Average MAE", np.mean(mae_scores))



