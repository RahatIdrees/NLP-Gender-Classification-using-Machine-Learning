# -*- coding: utf-8 -*-


Import the Dataset
"""

import pandas as pd

df = pd.read_excel("/content/B9AI006 - BLOG GENDER BALANCED.xlsx")

print(df.head())
print(df.shape)
print(df['GENDER'].value_counts())

"""Check the null value"""

print(df.isnull().sum())

"""**Data Preprocessing**

Clean the Text
"""

import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

df['BLOG'] = df['BLOG'].fillna('') # Fill NaN values with empty string
df['BLOG'] = df['BLOG'].apply(clean_text)
print(df[['BLOG']].head(10))

"""Convert the gender to numbers"""

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['GENDER'] = le.fit_transform(df['GENDER'])
print(df.head(10))

"""Split data into training and testing"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df['BLOG'], df['GENDER'],
    test_size=0.2,
    random_state=42
)

"""**CountVectorizer**"""

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(ngram_range=(1,2), max_features=5000)

X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)
print("Shape of BOW matrix:", X_train_cv.shape)
print("First 10 feature names:", cv.get_feature_names_out()[:10])

"""**TF-IDF**"""

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print("Shape of TF-IDF matrix:", X_train_tfidf.shape)
print("First 10 features:", tfidf.get_feature_names_out()[:10])
print("Sample TF-IDF vector:\n", X_train_tfidf[0].toarray())

"""**TF-IDF + PCA**"""

from sklearn.decomposition import PCA

pca = PCA(n_components=300)

X_train_pca = pca.fit_transform(X_train_tfidf.toarray())
X_test_pca = pca.transform(X_test_tfidf.toarray())
print("Shape of X_train_pca after PCA:", X_train_pca.shape)
print("Shape of X_test_pca after PCA:", X_test_pca.shape)

"""## **MODELS**

**LR+ TF-IDF**
"""

from sklearn.linear_model import LogisticRegression

model_lrt = LogisticRegression()
model_lrt.fit(X_train_tfidf, y_train)

pred_lrt = model_lrt.predict(X_test_tfidf)

"""**SVM + TF-IDF**"""

from sklearn.svm import SVC

model_svmt = SVC()
model_svmt.fit(X_train_tfidf, y_train)

pred_svmt = model_svmt.predict(X_test_tfidf)

"""**RF + TF-IDF**"""

from sklearn.ensemble import RandomForestClassifier

model_rft = RandomForestClassifier()
model_rft.fit(X_train_tfidf, y_train)

pred_rft = model_rft.predict(X_test_tfidf)

"""**LR + TFIDF-PCA**"""

from sklearn.linear_model import LogisticRegression

model_lrpca = LogisticRegression()
model_lrpca.fit(X_train_pca, y_train)

pred_lrpca = model_lrpca.predict(X_test_pca)

"""SVM + TFIDF-PCA**"""

from sklearn.svm import SVC

model_svmpca = SVC()
model_svmpca.fit(X_train_pca, y_train)

pred_svmpca = model_svmpca.predict(X_test_pca)

"""**RF + TFIDF-PCA**"""

from sklearn.ensemble import RandomForestClassifier

model_rfpca = RandomForestClassifier()
model_rfpca.fit(X_train_pca, y_train)

pred_rfpca = model_rfpca.predict(X_test_pca)

"""**LR + CV**"""

from sklearn.linear_model import LogisticRegression

model_lrcv = LogisticRegression()
model_lrcv.fit(X_train_cv, y_train)

pred_lrcv = model_lrcv.predict(X_test_cv)

"""**SVM + CV**"""

from sklearn.svm import SVC

model_svmcv = SVC()
model_svmcv.fit(X_train_cv, y_train)

pred_svmcv = model_svmcv.predict(X_test_cv)

"""**RF + CV**"""

from sklearn.ensemble import RandomForestClassifier

model_rfcv = RandomForestClassifier()
model_rfcv.fit(X_train_cv, y_train)

pred_rfcv = model_rfcv.predict(X_test_cv)

"""**Evaluation Results**"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

results = []


results.append([
    "TF-IDF + LR",
    accuracy_score(y_test, pred_lrt),
    precision_score(y_test, pred_lrt),
    recall_score(y_test, pred_lrt),
    f1_score(y_test, pred_lrt)
])


results.append([
    "TF-IDF + SVM",
    accuracy_score(y_test, pred_svmt),
    precision_score(y_test, pred_svmt),
    recall_score(y_test, pred_svmt),
    f1_score(y_test, pred_svmt)
])


results.append([
    "TF-IDF + RF",
    accuracy_score(y_test, pred_rft),
    precision_score(y_test, pred_rft),
    recall_score(y_test, pred_rft),
    f1_score(y_test, pred_rft)
])
results.append([
    "TF-IDF-PCA + LR",
    accuracy_score(y_test, pred_lrpca),
    precision_score(y_test, pred_lrpca),
    recall_score(y_test, pred_lrpca),
    f1_score(y_test, pred_lrpca)
])


results.append([
    "TF-IDF-PCA + SVM",
    accuracy_score(y_test, pred_svmpca),
    precision_score(y_test, pred_svmpca),
    recall_score(y_test, pred_svmpca),
    f1_score(y_test, pred_svmpca)
])


results.append([
    "TF-IDF-PCA + RF",
    accuracy_score(y_test, pred_rfpca),
    precision_score(y_test, pred_rfpca),
    recall_score(y_test, pred_rfpca),
    f1_score(y_test, pred_rfpca)
])
results.append([
    "CV + LR",
    accuracy_score(y_test, pred_lrcv),
    precision_score(y_test, pred_lrcv),
    recall_score(y_test, pred_lrcv),
    f1_score(y_test, pred_lrt)
])


results.append([
    "CV + SVM",
    accuracy_score(y_test, pred_svmcv),
    precision_score(y_test, pred_svmcv),
    recall_score(y_test, pred_svmcv),
    f1_score(y_test, pred_svmcv)
])


results.append([
    "CV + RF",
    accuracy_score(y_test, pred_rfcv),
    precision_score(y_test, pred_rfcv),
    recall_score(y_test, pred_rfcv),
    f1_score(y_test, pred_rfcv)
])

import pandas as pd

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
)

print(results_df)

"""**Comparison Of models Using BAR Graph**"""

import matplotlib.pyplot as plt

plt.figure()
plt.bar(results_df["Model"], results_df["Accuracy"])

plt.xticks(rotation=45)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

plt.show()