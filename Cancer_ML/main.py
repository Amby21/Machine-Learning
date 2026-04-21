#%%
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score,recall_score,precision_recall_curve
#%%
df = pd.read_csv("cancer_classification.csv")
# %%
df.describe()
df.isna().sum()
# %%
df.corr()
# %%
sns.countplot(data=df,x='benign_0__mal_1')
# %%
df.value_counts('benign_0__mal_1')
# %%
X = df.drop('benign_0__mal_1',axis=1)
y = df['benign_0__mal_1']
models = {
    "Logistic": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ]),
    "RandomForest": RandomForestClassifier()
}


#%%
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)
#%%
for name,model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="recall")
    print(name, scores.mean())
    

# %%
final_model = LogisticRegression()
final_model.fit(X_train, y_train)
preds = final_model.predict(X_test)

final_recall = recall_score(y_test,preds)
print("Test Recall:", final_recall)
# %%
