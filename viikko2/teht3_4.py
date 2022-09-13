import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

df_tnames = pd.read_csv('Titanic_names.csv')
df_tdata = pd.read_csv('Titanic_data.csv')

# 3

print(df_tnames.info())
print(df_tnames.describe())

print(df_tdata.info())
print(df_tdata.describe())
print("")
print("")
df_tdata.hist(bins=4)

df_tmerge = pd.merge(df_tnames, df_tdata,how='inner',on='id')

p_Count = df_tmerge.shape[0]
print(f'Henkilöitä: {p_Count}')

m_Count = (df_tmerge.GenderCode == 0).sum()
f_Count = (df_tmerge.GenderCode == 1).sum()
print(f'Male = {m_Count} Female = {f_Count}')

age_Avg = df_tmerge.Age.mean()
print(f'Keski-ikä: {round(age_Avg)}')

no_Age = (df_tmerge.Age == 0).sum()
print(f'{no_Age} values where age is 0')

# 4

age_Ravg = df_tmerge.Age.loc[df_tmerge.Age != 0].mean()
print(f'Oikea keski-ikä: {round(age_Ravg)}')

df_tmerge.loc[df_tmerge['Age']==0,'Age'] = age_Ravg

u_Vals = df_tmerge['PClass'].unique()
print(f'Unique values from PClass: {u_Vals}')

s_Count = (df_tmerge.Survived == 1).sum()
ns_Count = (df_tmerge.Survived == 0).sum()

s_Prec = round((s_Count/p_Count) * 100, 1)
ns_Prec = round((ns_Count/p_Count) * 100, 1)

print(f'Selviytyneitä = {s_Count}, prosentti = {s_Prec}%')
print(f'Ei selvinneitä = {ns_Count}, prosentti = {ns_Prec}%')


ms_Count = ((df_tmerge.Survived == 1) & (df_tmerge.GenderCode == 0)).sum()
fs_Count = ((df_tmerge.Survived == 1) & (df_tmerge.GenderCode == 1)).sum()

mns_Count = ((df_tmerge.Survived == 0) & (df_tmerge.GenderCode == 0)).sum()
fns_Count = ((df_tmerge.Survived == 0) & (df_tmerge.GenderCode == 1)).sum()

print(f'Selviytyneet miehet {ms_Count}, ei selvinneet {mns_Count}')
print(f'Selviytyneet naiset {fs_Count}, ei selvinneet {fns_Count}')