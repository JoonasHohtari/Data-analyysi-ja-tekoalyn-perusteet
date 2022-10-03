import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import seaborn as sns
import scipy.stats as stats

# # T3

# 1
df = pd.read_csv('salary.csv')

plt.scatter(df.YearsExperience, df.Salary)
plt.show()

# 2

cor_cols = ['YearsExperience', 'Salary']
df_corr = df.loc[:, cor_cols]
corr = df_corr.corr()
sns.heatmap(corr, annot=True)
plt.show()

pearsonr_result = stats.pearsonr(df['YearsExperience'], df['Salary'])
print('PearsonR Tulokset')
print(f'{pearsonr_result}')
print(f'{pearsonr_result.confidence_interval()}')

# 3

X = df.loc[:,['YearsExperience']] #
y = df.loc[:,['Salary']] #

# 4

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,
                                                    random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)

# 5

coef = model.coef_
inter = model.intercept_
print(f'Suoran yhtälö on: y = {coef[0]} * YearsExperience + {inter}')

# 6

y_pred = model.predict(X_test)

# 7

plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.show()

# 8

sns.regplot(x=X_test, y=y_test, data=df, truncate=False)

# 9

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'r2 = {r2}')
print(f'mae = {mae}')
print(f'mse = {mse}')
print(f'rmse = {rmse}')

# 0 

y_pred_val = model.predict([[7]])
print(f'Ennustettu palkka, kun työkokemus 7v = {y_pred_val}')
