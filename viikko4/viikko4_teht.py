import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

df = pd.read_excel('tt.xlsx')

print(df.describe())
print(df.info())

cols = ['sukup', 'ikä', 'koulutus']
df[cols].hist() # df[['sukup', 'ikä', 'koulutus']].hist()
# Listataan koulutus tasot
koulutus = ['Peruskoulu', '2. Aste', 'Korkeakoulu', 'Ylempi korkeakoulu']

print(df['palkka'].nlargest(5))
print(df['ikä'].nsmallest(5))

edu_df = pd.crosstab(index=df['koulutus'], columns='Lukumäärä')
# Käytetään koulutus-nimiä indexissä
edu_df.index = koulutus

# Luodaan uusi indexi
edu_df = edu_df.reset_index()

# Uudelleen nimetään sarakkeet
edu_df.columns = ['Koulutus', 'Lukumäärä']

# Lasketaan prosenttiosuudet
tot = edu_df['Lukumäärä'].sum()
edu_df['%'] = round(edu_df['Lukumäärä']/tot * 100, 2)

sns.barplot(x='Lukumäärä', y='Koulutus', data=edu_df)
plt.show()

gedu_df = pd.crosstab(index=df['koulutus'], columns=df['sukup'])
gedu_df.index = koulutus
gedu_df = gedu_df.reset_index()
gedu_df.columns = ['Koulutus', 'Miehet', 'Naiset']


# # T3
# Apulista
sukupuolet = ['Miehet', 'Naiset']

p = stats.chi2_contingency(gedu_df[sukupuolet])[1]

if p>0.05:
    print(f'Riippuvuus ei ole tilastollisesti merkitsevä')
else:
    print(f'Riippuvuus on tilastollisesti merkitsevä')


# # T2 Kuvaajat
# Matplotlib
gedu_df.plot(kind='barh')
plt.show()

# Seaborn
gedu_df = pd.melt(gedu_df, id_vars='Koulutus', 
             var_name='Sukupuoli', value_name='Lukumäärä')
sns.barplot(x='Lukumäärä', y='Koulutus', 
            hue='Sukupuoli',data=gedu_df)
plt.show()

# # T4
cor_cols = ['sukup','ikä','perhe','koulutus','palkka']
df_corr = df.loc[:, cor_cols]
corr = df_corr.corr()
sns.heatmap(corr, annot=True)
plt.show()