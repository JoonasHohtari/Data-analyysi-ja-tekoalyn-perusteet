import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# T1
# y = 2x + 3, x = [1,2,3,4,6,7,8]

x = pd.Series([1,2,3,4,6,7,8])
y = 2 * x + 3

df = pd.DataFrame({'x':x, 'y':y})

plt.scatter(df.x, df.y)
plt.plot(df.x, df.y)
plt.show()

# T2

X = df.loc[:,['x']] #
y = df.loc[:,['y']] #

model = LinearRegression()
model.fit(X,y)

y_pred = model.predict([[5]])

plt.scatter(df.x, df.y)
plt.scatter(5, y_pred, c='r')
plt.plot(df.x, df.y)
plt.show()

coef = model.coef_
inter = model.intercept_
print(f'Suoran yhtälö on: y = {coef[0]} * x + {inter}')

