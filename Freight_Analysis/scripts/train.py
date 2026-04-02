#%%
import pandas as pd
import holidays
import os
import mlflow
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
#%%
#Load order table
# Use a relative path that works on any computer
# 1. Get the directory where THIS script is saved
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Go UP one level to the project root (Freight_Analysis)
project_root = os.path.dirname(current_dir)
br_holidays = holidays.BR()
# base_path = os.path.dirname(__file__) if "__file__" in locals() else "."
order_path = os.path.join(project_root, "data", "olist_orders_dataset.csv")
prod_path = os.path.join(project_root,"data","olist_products_dataset.csv")
item_path = os.path.join(project_root,"data","olist_order_items_dataset.csv")


df_orders = pd.read_csv(order_path)
df_orders['order_purchase_timestamp'] = pd.to_datetime(df_orders['order_purchase_timestamp'])
df_orders['is_holiday'] = df_orders['order_purchase_timestamp'].apply(lambda x: 1 if x in br_holidays else 0)
# Convert to datetime
df_orders['order_delivered_customer_date'] = pd.to_datetime(df_orders['order_delivered_customer_date'])
df_orders['order_estimated_delivery_date'] = pd.to_datetime(df_orders['order_estimated_delivery_date'])

# Target: 1 if late, 0 if on time
df_orders['is_late'] = (df_orders['order_delivered_customer_date'] > df_orders['order_estimated_delivery_date']).astype(int)
# Minimal merge for high-performance training
df_master = df_orders[['order_id', 'customer_id', 'is_late','is_holiday']].merge(
    pd.read_csv(item_path)[['order_id', 'product_id', 'freight_value']], on='order_id').merge(
    pd.read_csv(prod_path)[['product_id', 'product_weight_g']], on='product_id')
#%%
df_master.info

# %%
df_master.describe()
# %%
df_master.isna().sum()
# %%
df_master.dropna(subset=['product_weight_g'])
#%%
df_master['is_late'].value_counts()
cols_to_drop = ['order_id', 'customer_id', 'product_id', 'is_late']
# %%
mlflow.set_experiment("Freight Analysis")

with mlflow.start_run():


    X = df_master.drop(cols_to_drop,axis=1)
    y = df_master['is_late']

    X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    model = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=42)
    model.fit(X_train,y_train)

    preds = model.predict(X_test)
    mlflow.sklearn.log_model(model,'logistic-model')
    target_path = os.path.join(os.path.dirname(__file__), "..", "models", "logistic-model")
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    mlflow.sklearn.save_model(model, target_path)
    mlflow.log_param("n_estimators",100)
    print(f" Model saved to permanent location: {target_path}")    

#%%
# Load your model from the folder where you saved it
model = mlflow.sklearn.load_model("./models/logistic-model")

# This attribute holds the exact list and order of features
print("The model expects these features in this order:")
print(model.feature_names_in_)
# %%
