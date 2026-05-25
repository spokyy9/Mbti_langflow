import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression 
from xgboost import XGBClassifier
import scipy.sparse
import shap
import joblib
import streamlit as st
import numpy as np
import scipy.sparse
df = pd.read_csv('mbti_1.csv')
print(df.info())
print("-------------------------------------")
print(df.describe())
print("-------------------------------------")
print(df.shape)
print("-------------------------------------")
print(df.isnull().sum())
print("-------------------------------------")
print(df.duplicated().sum())
print("-------------------------------------")
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text=re.sub(r'[^a-zA-Z ]', '', text)
    text=text.lower()
    return text
print("-------------------------------------")
df['posts']=df['posts'].apply(clean_text)
print(df['posts'].head())
print("-------------------------------------")
le = LabelEncoder()
df['target'] = le.fit_transform(df['type'])
print(df[['type', 'target']].head())
print("-------------------------------------")
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1,2),
    stop_words='english'
)
X = tfidf.fit_transform(df['posts'])
print(X.shape)
print("-------------------------------------")
y = df['target']
print("-------------------------------------")
print(df['type'].value_counts())
#df.to_csv('mbti_cleaned.csv', index=False)
print("-------------------------------------")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("-------------------------------------")
#model = RandomForestClassifier(class_weight='balanced', random_state=42) 
#model.fit(X_train, y_train)   
print("-------------------------------------")
model = LogisticRegression(max_iter=10000, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
print("-------------------------------------")
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print("-------------------------------------")
#model = XGBClassifier(n_estimators=200, random_state=42, eval_metric='mlogloss')
#X_train_dense = X_train.toarray()
#X_test_dense = X_test.toarray()
#model.fit(X_train_dense, y_train)
#y_pred = model.predict(X_test_dense)
#print(accuracy_score(y_test, y_pred))
print("-------------------------------------")
#explainer = shap.LinearExplainer(model, X_train)
#shap_values = explainer.shap_values(X_test)
print("-------------------------------------")
joblib.dump(model, 'model.pkl')
joblib.dump(tfidf, 'tfidf.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("-------------------------------------")

X_train_sample = X_train[:100]
scipy.sparse.save_npz('X_train_sample.npz', X_train_sample)
print("-------------------------------------")
