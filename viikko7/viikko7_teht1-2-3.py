import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import seaborn as sns
import matplotlib.pyplot as plt
import pickle #save encoder

# 1

df = pd.read_csv('titanic-class-age-gender-survived.csv')

# 2
# X = df.iloc[:,[1]]
# X = df.iloc[:,[1,2]]
X = df.iloc[:,[0,1,2]]
y = df.iloc[:,[-1]]

# 3 

X_org = X
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), ['Gender'])], remainder='passthrough')
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), ['Gender','PClass'])], remainder='passthrough')
X = ct.fit_transform(X)

# Scaler
# scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.fit_transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train) # X_train_scaled if scaler is used

y_pred = model.predict(X_test) # X_test_scaled if scaler is used
y_pred_prob = model.predict_proba(X_test) # X_test_scaled if scaler is used

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f'acc: {acc}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

sns.heatmap(cm, annot=True, fmt='g')
plt.show()

Xnew = pd.read_csv('titanic-new.csv')
# X_new = Xnew.iloc[:,[1]]
# X_new = Xnew.iloc[:,[1,2]]
X_new = Xnew.iloc[:,[0,1,2]]
X_new = ct.transform(X_new)

y_pred_new = model.predict(X_new)
X_pred_new = model.predict_proba(X_new)

tn, fp, fn, tp = cm.ravel()