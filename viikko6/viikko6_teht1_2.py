import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# 1

df = pd.read_csv('startup.csv')

# 2

# X = df.loc[:,['R&D Spend', 'Administration', 'Marketing Spend', 'State']]
# y = df.loc[:,['Profit']]

X_1 = df.iloc[:, :-1]
y_1 = df.iloc[:, [-1]]
X_2 = df.iloc[:, :-1]
y_2 = df.iloc[:, [-1]]

# 3

dummies_state = pd.get_dummies(X_1['State'], drop_first=True)
X_1 = X_1.join(dummies_state)
X_1.drop('State', inplace=True, axis=1)

X_2_org = X_2
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), ['State'])], remainder='passthrough')
X_2 = ct.fit_transform(X_2)

# 4

X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(X_1, y_1,test_size=0.2,
                                                    random_state=0)

X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2, y_2,test_size=0.2,
                                                    random_state=0)


# Skaalataan data

scaler_x_1 = StandardScaler()
X_1_train = scaler_x_1.fit_transform(X_1_train)
X_1_test = scaler_x_1.transform(X_1_test)

scaler_x_2 = StandardScaler()
X_2_train = scaler_x_2.fit_transform(X_2_train)
X_2_test = scaler_x_2.transform(X_2_test)

scaler_y_1 = StandardScaler()
y_1_train = scaler_y_1.fit_transform(y_1_train)

scaler_y_2 = StandardScaler()
y_2_train = scaler_y_2.fit_transform(y_2_train)

# 5

model_1 = LinearRegression()
model_2 = LinearRegression()

model_1.fit(X_1_train, y_1_train)
model_2.fit(X_2_train, y_2_train)

# 6
y_1_pred = scaler_y_1.inverse_transform(model_1.predict(X_1_test).reshape(-1,1))
y_2_pred = scaler_y_2.inverse_transform(model_2.predict(X_2_test).reshape(-1,1))

coef = model_1.coef_
inter = model_1.intercept_
print(f'Suoran model_1 yhtälö on: \n y ={coef[0]} * x + {inter} \n')

coef = model_2.coef_
inter = model_2.intercept_
print(f'Suoran model_2 yhtälö on: \n y ={coef[0]} * x + {inter} \n')

# 7

# y_1_pred = model_1.predict(X_1_test)
# y_2_pred = model_2.predict(X_2_test)

print(f'Pred 1:\n {y_1_pred} \n Pred 2:\n {y_2_pred}')

# 8

r2_1 = r2_score(y_1_test, y_1_pred)
mae_1 = mean_absolute_error(y_1_test, y_1_pred)
mse_1 = mean_squared_error(y_1_test, y_1_pred)
rmse_1 = np.sqrt(mse_1)

r2_2 = r2_score(y_2_test, y_2_pred)
mae_2 = mean_absolute_error(y_2_test, y_2_pred)
mse_2 = mean_squared_error(y_2_test, y_2_pred)
rmse_2 = np.sqrt(mse_2)

print('\nModel_1: \n')
print(f'r2 = {r2_1}')
print(f'mae = {mae_1}')
print(f'mse = {mse_1}')
print(f'rmse = {rmse_1} \n')

print('Model_2: \n')
print(f'r2 = {r2_2}')
print(f'mae = {mae_2}')
print(f'mse = {mse_2}')
print(f'rmse = {rmse_2}')

print('\nMallit suoriutuivat hyvin, koska r2 on lähellä arvoa 1.0')

# # Tehtävä 2.

y_1_pred_comp = model_1.predict([[165349.20, 136897.80, 471784.10, 0, 1]])

print(f'Yrityksen voitto: {y_1_pred_comp}')

df_2 = pd.read_csv('new_company.csv')
# ???
dummies_df_2 = pd.get_dummies(df_2, drop_first=True)

df_3 = pd.read_csv('new_company_ct.csv')
# ???
df_3_org = df_3
ct_df_3 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), ['State'])], remainder='passthrough')
df_3 = ct_df_3.fit_transform(df_3)


