import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_emp = pd.read_csv('emp-dep.csv')

# T1

xdata_sc = df_emp.age
ydata_sc = df_emp.salary
plt.scatter(xdata_sc, ydata_sc)
plt.show()

df_emp.groupby('dname')['dep'].count().plot(kind='bar')
df_emp.groupby('dep')['dname'].count().plot(kind='bar')

# T2

df_emp.groupby('age_group')['age'].count().plot(kind='bar')

# T3

emp_Count = df_emp.shape[0]
m_Count = (df_emp.gender == 0).sum()
f_Count = (df_emp.gender == 1).sum()
m_Prec = round((m_Count/emp_Count) * 100, 1)
f_Prec = round((f_Count/emp_Count) * 100, 1)
y = np.array([m_Prec,f_Prec])
labels = 'Miehet', 'Naiset'
plt.pie(y,labels=labels,autopct='%.1f%%')
plt.show()

