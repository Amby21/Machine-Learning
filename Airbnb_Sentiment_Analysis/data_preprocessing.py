#%%
import nltk
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from textblob import TextBlob
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder


from sklearn.svm import SVC
#%% 
from xgboost import XGBClassifier
#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

import re
import string
import joblib
import warnings
# warning.filterwarnings('ignore')
#%%

#%%
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('wordnet', quiet=True)
#%%
df = pd.read_csv('Dataset/reviews.txt')
#%%
df.head(5)
df.shape
df.info()
# %%
df.isna().sum()
# %%
df.columns
# ID columns are not required as they do not provide any substantial information
# df.drop('')
# %% Exploratory Data Analysis

reviewer_freq = df['reviewer_name'].value_counts()
print(reviewer_freq)
# %%
plt.figure(figsize=(10,6))
sns.distplot(reviewer_freq, kde=False)
plt.title ("Distribution of Reviewer Frequency")
plt.xlabel('Review Frequency')
plt.ylabel('Count')
plt.show()
# %%
df['comment_length'] = df['cleaned_comments'].apply(lambda x: len(x.split()))
plt.figure(figsize=(10,6))
sns.histplot(df['comment_length'], bins=30, kde=True)
plt.title('Distribution of Comment Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.show()
# %%
# Sentiment Visualization
plt.figure(figsize=(10,6))
sns.histplot(df['sentiment'],bins=30, kde=True)
plt.title('Sentiment Distribution of Comments')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()
# %%
# Comment length vs Sentiment
plt.figure(figsize=(10,6))
sns.scatterplot(x='comment_length', y='sentiment', data=df)
plt.title('Comment Length vs Sentiment')
plt.xlabel('Comment Length')
plt.ylabel('Sentiment Score')
plt.show()
# %%
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_comments'])
#%%
X_train,X_test,y_train,y_test = train_test_split(X, df['sentiment'],test_size=0.2,random_state=42)
#%%
lgc = LogisticRegression()
lgc.fit(X_train, y_train)
lgc_pred = lgc.predict(X_test)
lgc_acc = accuracy_score(y_test, lgc_pred)
lgc_prec = precision_score(y_test, lgc_pred, average='weighted')
lgc_rec = recall_score(y_test, lgc_pred, average='weighted')
lgc_f1 = f1_score(y_test, lgc_pred, average='weighted')
print(f"Model Type: {lgc}")
print(f"Accuracy Score:  {(lgc_acc):%}")
print(f"Precision Score: {(lgc_prec):%}")
print(f"Recall Score:    {(lgc_rec):%}")
print(f"F1-Score:        {(lgc_f1):%}")
# %%
print("Classification Report for Logistic Regression")
print(classification_report(y_test, lgc_pred))
#save the logistic regression model
joblib.dump(lgc,'Models/logistic_regression_model.pkl')
# %% ADA Boost
print("Evaluation for: Adaboost Classifier")
abc = AdaBoostClassifier()
abc.fit(X_train,y_train)
abc_pred = abc.predict(X_test)
abc_acc = accuracy_score(y_test,abc_pred)
abc_prec = precision_score(y_test, abc_pred,average='weighted')
abc_rec = recall_score(y_test,abc_pred,average='weighted')
abc_f1 = f1_score(y_test,abc_pred,average='weighted')
print(f"Accuracy Score:  {(abc_acc):%}")
print(f"Precision Score: {(abc_prec):%}")
print(f"Recall Score:    {(abc_rec):%}")
print(f"F1-Score:        {(abc_f1):%}")
# %% KNN

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)

knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)
knn_prec = precision_score(y_test,knn_pred,average='weighted')
knn_recall = recall_score(y_test,knn_pred,average='weighted')
knn_f1 = f1_score(y_test,knn_pred,average='weighted')
print(f"Model Type: {knn}")
print(f"Prediction: {knn_pred}")
#save the knn model
joblib.dump(knn_pred, 'Models/knn_model.pkl')

#%% Random Forest
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
rfc_acc = accuracy_score(y_test, rfc_pred)
rfc_prec = precision_score(y_test, rfc_pred,average='weighted')
rfc_recall = recall_score(y_test, rfc_pred, average="weighted")
rfc_f1 = f1_score(y_test, rfc_pred, average="weighted")
print(f"Accuracy Score:  {(rfc_acc):%}")
print(f"Precision Score: {(rfc_prec):%}")
print(f"Recall Score:    {(rfc_recall):%}")
print(f"F1-Score:        {(rfc_f1):%}")
# save the random forest model
joblib.dump(rfc, 'Models/random_forest_model.pkl')
#%%
lgc_scores = [lgc_acc*100, lgc_prec*100, lgc_rec*100, lgc_f1*100]
abc_scores = [abc_acc*100, abc_prec*100,
              abc_rec*100, abc_f1*100]
knn_scores = [knn_acc*100, knn_prec*100,
              knn_recall*100, knn_f1*100]

rfc_scores = [rfc_acc*100, rfc_prec*100,
              rfc_recall*100, rfc_f1*100]

x_grid = ["Accuracy", "Precision", "Recall", "F1-Score"]
y_grid = [value for value in range(1, 120, 10)]
xpos = np.arange(len(x_grid))
bar_width = .10

plt.figure(figsize=(20, 10), dpi=700, facecolor='w', edgecolor='k')
plt.title("Model Evaluation Scores", fontsize=20)
# Function to add value labels on the bars


def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 f'{height:.2f}%', ha='center', va='bottom', rotation=90)


bars_lgc = plt.bar(xpos, lgc_scores, width=bar_width,
                   label="Logistic Regression")
add_labels(bars_lgc)

bars_abc = plt.bar(xpos + bar_width, abc_scores,
                   width=bar_width, label="AdaBoost")
add_labels(bars_abc)

bars_knn = plt.bar(xpos + bar_width*2, knn_scores,
                   width=bar_width, label="K-Neighbors")
add_labels(bars_knn)

bars_rfc = plt.bar(xpos + bar_width*4, rfc_scores,
                   width=bar_width, label="Random Forest")
add_labels(bars_rfc)

# %%
