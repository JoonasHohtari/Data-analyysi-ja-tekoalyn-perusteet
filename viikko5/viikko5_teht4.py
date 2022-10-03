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

# 3

X = df.loc[:,['median_income']]
y = df.loc[:,['median_house_value']]