import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

# Tehtävä 1

df_Emp = pd.read_csv('employees.csv', dtype={'phone1':str,'phone2':str})
df_Dep = pd.read_csv('departments.csv')

df_Merge = pd.merge(df_Emp, df_Dep, how='inner', on='dep')

df_Merge.drop('image', inplace=True, axis='columns')

# Tehtävä 2

emp_Count = df_Merge.shape[0]
print(f'Työntekijöitä {emp_Count}')

m_Count = (df_Merge.gender == 0).sum()
f_Count = (df_Merge.gender == 1).sum()
print(f'Male = {m_Count} Female = {f_Count}')

m_Prec = round((m_Count/emp_Count) * 100, 1)
f_Prec = round((f_Count/emp_Count) * 100, 1)
print(f'Male Precentage = {m_Prec}%, Female Precentage = {f_Prec}%')

s_Min = df_Merge.salary.min()
s_Max = df_Merge.salary.max()
salary_Avg = df_Merge.salary.mean()

print(f'Minimi: {s_Min}, Maksimi: {s_Max}, Keskiarvo {round(salary_Avg)}')

tk_Avg = df_Merge.loc[df_Merge.dep == 4].salary.mean()

print(f'Tuotekehitys osaston keskipalkka: {tk_Avg}')

noPh2 = df_Merge.phone2.isna().sum()

print(f'{noPh2}:llä ei ole kakkos puhelinta')

datetime.now() - pd.to_datetime(df_Merge.bdate)

df_Merge['age'] = (datetime.now() - pd.to_datetime(df_Merge.bdate)) // timedelta(365.2425)

df_Merge.loc[df_Merge['age']<=15, 'age_group'] = '20'
df_Merge.loc[df_Merge['age'].between(20,24), 'age_group'] = '25'
df_Merge.loc[df_Merge['age'].between(25,29), 'age_group'] = '30'
df_Merge.loc[df_Merge['age'].between(30,34), 'age_group'] = '35'
df_Merge.loc[df_Merge['age'].between(35,39), 'age_group'] = '40'
df_Merge.loc[df_Merge['age'].between(40,44), 'age_group'] = '45'
df_Merge.loc[df_Merge['age'].between(45,49), 'age_group'] = '50'
df_Merge.loc[df_Merge['age'].between(50,54), 'age_group'] = '55'
df_Merge.loc[df_Merge['age'].between(55,59), 'age_group'] = '60'
df_Merge.loc[df_Merge['age'].between(60,64), 'age_group'] = '65'
df_Merge.loc[df_Merge['age'].between(65,69), 'age_group'] = '70'
df_Merge.loc[df_Merge['age']>70, 'age_group'] = '75'


df_s_a_g = df_Merge.loc[:, ['salary', 'age', 'gender']]
corr = df_s_a_g.corr()
sns.heatmap(corr, annot=True)
plt.show()