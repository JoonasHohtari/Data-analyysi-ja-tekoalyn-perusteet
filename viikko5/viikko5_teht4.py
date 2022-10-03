import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import seaborn as sns
import scipy.stats as stats

# T4

# 1
df = pd.read_csv('housing.csv')

# 2 mediaaniarvo ja kotitalouden vuositulot

plt.scatter(df.median_income, df.median_house_value)
plt.show()

# 3 median_income median_house_value

X = df.loc[:,['median_income']]
y = df.loc[:,['median_house_value']]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,
                                                    random_state=0)
# 4

model = LinearRegression()
model.fit(X_train, y_train)

# 5

coef = model.coef_
inter = model.intercept_
print(f'Suoran yhtälö on: y ={coef[0]} * x + {inter}')

# 6

y_pred = model.predict(X_test)

plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.show()

# 7

plt.hist(y_pred, alpha=0.45, color='red')
plt.hist(y_test, alpha=0.45, color='blue')
plt.legend(['y_pred', 'y_test'])
plt.show()

# 8

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'r2 = {r2}')
print(f'mae = {mae}')
print(f'mse = {mse}')
print(f'rmse = {rmse}')
print('Malli epätarkka, koska pelkkä mae arvo on todella suuri.')
print('R2 arvo kertoo myös, että sovitus on huono, koska arvo on jopa alle 0.5')

# 9
y_pred_val = model.predict([[3.0]])
print(f'Ennustettu kotitalouden arvo, kun vuositulot 30 000 = {y_pred_val}')