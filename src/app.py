from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd

df = pd.read_csv('../data/raw/AB_NYC_2019.csv')


df.info()
df.shape

df.drop(['id', 'name', 'host_name', 'last_review'], axis = 1, inplace = True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.head()


#VISUALIZACIÓN DE DATOS

import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['price'], bins=50)
plt.xlabel('Precio ($)')
plt.ylabel('Frecuencia')
plt.show()

sns.countplot(x='neighbourhood_group', data=df)
plt.title('Cantidad de alojamientos por barrio')
plt.xlabel('Barrio')
plt.ylabel('Cantidad')
plt.show()


plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Mapa de correlación entre variables')
plt.show()


from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)


df_train.to_csv('../data/processed/airbnb_ny_train.csv', index=False)
df_test.to_csv('../data/processed/airbnb_ny_test.csv', index=False)

