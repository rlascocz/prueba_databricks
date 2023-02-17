# Databricks notebook source
!pip install prince
%pip install bamboolib  

# COMMAND ----------

import re #libreria expresiones regulares
import prince
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bamboolib as bam #from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import split
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import col
from prince import MCA #reduir dimensionalidad

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# COMMAND ----------

df = pd.read_csv(r'/dbfs/FileStore/shared_uploads/rlascocz@gmail.com/diabetic_data.csv', sep=',', decimal='.')
# Step: Drop duplicates based on ['encounter_id']
df = df.drop_duplicates(subset=['encounter_id'], keep='first')

# columnas eliminadas porque estan fuera de contexto del problema
df = df.drop(columns=['encounter_id','patient_nbr',"examide",'examide', 'weight','citoglipton',"payer_code"])

#columnas  eliminadas poque existe desbalanceo de clases
df = df.drop(columns=["medical_specialty","max_glu_serum","A1Cresult"])

columna_a_excluir ='readmitted'

# Separar las variables predictoras y la variable objetivo
X = df.drop(columna_a_excluir, axis=1)  # características
y = df[columna_a_excluir]  # variable objetivo

X_model, X_etl, y_model, y_etl = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)


# COMMAND ----------

conteo_val = y_etl.value_counts(normalize=True)*100
print("para evaluacion se tienen {} valores, y se distribuyen \n{}".format(y_etl.count(),conteo_val))
conteo_train = y_model.value_counts(normalize=True)*100
print("para entrenamiento se tienen {} valores, y se distribuyen \n{}".format(y_model.count(),conteo_train))

# COMMAND ----------

df = pd.concat([X_model, y_model], axis=1)

# COMMAND ----------

# Calcular la correlación de Spearman
corr, p = spearmanr(df)
# Crear un dataframe con los resultados
corr_df = pd.DataFrame(corr, index=df.columns, columns=df.columns)

# COMMAND ----------

importancia_variables=corr_df["readmitted"]
importancia_variables =importancia_variables.abs().sort_values(ascending=False)*100
importancia_variables=importancia_variables.reset_index()
importancia_variables= importancia_variables.loc[importancia_variables["readmitted"] > 0.5]
columnas_importancia = importancia_variables.rename(columns={'index': 'columna'})['columna']
print(columnas_importancia)

# COMMAND ----------

# Graficar la serie
importancia_variables.plot()
#fig, ax = plt.subplots()
#ax.plot(x, y)

# Rota las etiquetas del eje x 90 grados
plt.xticks(rotation=90)
# Mostrar la gráfica
plt.show()

# COMMAND ----------

# filtrar el DataFrame para incluir solo las columnas en col_names
df_filtrado = df.loc[:, columnas_importancia]


# COMMAND ----------

numeric_cols = df_filtrado.select_dtypes(include=['int', 'float']).columns.tolist()
categorical_cols = df_filtrado.select_dtypes(include=['object']).columns.tolist()
df_num=df_filtrado[numeric_cols]
df_categ=df_filtrado[categorical_cols]

# COMMAND ----------

summary =  df_categ.describe()
print(summary)

# COMMAND ----------

# procesamiento de la columna diag 3
#eliminar puntos
columna='diag_3'
df_categ[columna] = df_categ[columna].apply(lambda x: x.split(".")[0])

#eliminar caracteres alfabeticos
df_categ[columna] = df_categ[columna].apply(lambda x: re.sub(r'[a-zA-Z]+', '', x))


# Contar los valores únicos en la columna "column_name"
percentages=  df_categ[columna].value_counts()/ len(df_categ) * 100
acumulada=percentages.cumsum()
percentages = pd.concat([percentages, acumulada], axis=1)
percentages = pd.concat([percentages,df_categ[columna].value_counts() ], axis=1)
percentages=percentages.reset_index()

print(percentages)


# COMMAND ----------

# procesamiento de la columna diag 3
#eliminar puntos
columna="diag_2"
df_categ[columna] = df_categ[columna].apply(lambda x: x.split(".")[0])

#eliminar caracteres alfabeticos
df_categ[columna] = df_categ[columna].apply(lambda x: re.sub(r'[a-zA-Z]+', '', x))


# Contar los valores únicos en la columna "column_name"
percentages=  df_categ[columna].value_counts()/ len(df_categ) * 100
acumulada=percentages.cumsum()
percentages = pd.concat([percentages, acumulada], axis=1)
percentages = pd.concat([percentages,df_categ[columna].value_counts() ], axis=1)
percentages=percentages.reset_index()

# Mostrar gráfico
plt.show()
print(percentages)

# COMMAND ----------

# procesamiento de la columna diag 3https://adb-5459650832056091.11.azuredatabricks.net/?o=5459650832056091#
#eliminar puntos
columna="diag_1"
df_categ[columna] = df_categ[columna].apply(lambda x: x.split(".")[0])

#eliminar caracteres alfabeticos
df_categ[columna] = df_categ[columna].apply(lambda x: re.sub(r'[a-zA-Z]+', '', x))


# Contar los valores únicos en la columna "column_name"
percentages=  df_categ[columna].value_counts()/ len(df_categ) * 100
acumulada=percentages.cumsum()
percentages = pd.concat([percentages, acumulada], axis=1)
percentages = pd.concat([percentages,df_categ[columna].value_counts() ], axis=1)
percentages=percentages.reset_index()

# Mostrar gráfico
plt.show()
print(percentages)

# COMMAND ----------

print(df_categ[["diag_1","diag_2","diag_3"]])

# COMMAND ----------

# cargar los datos
data = df_categ[["diag_1","diag_2","diag_3"]]

# ajustar MCA
mca = prince.MCA(n_components=100)
mca.fit(data)
# calcular la inercia acumulada
explained_inertia = mca.explained_inertia_
cumulative_inertia = np.array(explained_inertia).cumsum()

# graficar la inercia acumulada
plt.plot(range(1, len(cumulative_inertia)+1), cumulative_inertia, '-o')
plt.xlabel('Número de componentes')
plt.ylabel('Inercia acumulada')
plt.show()

# COMMAND ----------

# Creamos un ejemplo de DataFrame con una columna categórica
columna="diag_1"
df =df_categ[columna]

# Calculamos la frecuencia de cada categoría en la columna
frequencies = df.value_counts()
# Calculamos el total de valores
n = frequencies.sum()

# Calculamos el coeficiente de Gini
gini = 1 - sum([(fi / n)**2 for fi in frequencies])

print(gini)

# COMMAND ----------

# Creamos un ejemplo de DataFrame con una columna categórica
columna="diag_2"
df =df_categ[columna]

# Calculamos la frecuencia de cada categoría en la columna
frequencies = df.value_counts()
# Calculamos el total de valores
n = frequencies.sum()

# Calculamos el coeficiente de Gini
gini = 1 - sum([(fi / n)**2 for fi in frequencies])

print(gini)

# COMMAND ----------

# Creamos un ejemplo de DataFrame con una columna categórica
columna="diag_3"
df =df_categ[columna]

# Calculamos la frecuencia de cada categoría en la columna
frequencies = df.value_counts()
# Calculamos el total de valores
n = frequencies.sum()

# Calculamos el coeficiente de Gini
gini = 1 - sum([(fi / n)**2 for fi in frequencies])

print(gini)

# COMMAND ----------

columna="diag_3"
df =df_categ[columna]
print("Numero de categorias originales {} ".format(len(df.value_counts())))
cantidad =180
cond = df.value_counts() < cantidad
df = df.replace(cond.index[cond], 'Otros')


# Calculamos la frecuencia de cada categoría en la columna
frequencies = df.value_counts()
# Calculamos el total de valores
n = frequencies.sum()

# Calculamos el coeficiente de Gini
gini = 1 - sum([(fi / n)**2 for fi in frequencies])

percentages = frequencies / len(df) * 100

print("Numero de categorias reconstruidas {} ".format(len(percentages)))
print("-------------------------")
print("valor gini {} ".format(gini))
print("-------------------------")     
print(percentages)


# COMMAND ----------

#remplazar la columan original
columna="diag_3"
#df =df_categ[columna]
cantidad =180
cond = df_categ[columna].value_counts() < cantidad
df_categ[columna] = df_categ[columna].replace(cond.index[cond], 'Otros')
print(df_categ[columna])

# COMMAND ----------

columna="diag_2"
df =df_categ[columna]
print("Numero de categorias originales {} ".format(len(df.value_counts())))
cantidad=160
cond = df.value_counts() < cantidad
df = df.replace(cond.index[cond], 'Otros')


# Calculamos la frecuencia de cada categoría en la columna
frequencies = df.value_counts()
# Calculamos el total de valores
n = frequencies.sum()

# Calculamos el coeficiente de Gini
gini = 1 - sum([(fi / n)**2 for fi in frequencies])

percentages = frequencies / len(df) * 100

print("Numero de categorias reconstruidas {} ".format(len(percentages)))
print("-------------------------")
print("valor gini {} ".format(gini))
print("-------------------------")     
print(percentages)

# COMMAND ----------

#remplazar la columan original
columna="diag_2"
#df =df_categ[columna]
cantidad =160
cond = df_categ[columna].value_counts() < cantidad
df_categ[columna] = df_categ[columna].replace(cond.index[cond], 'Otros')
print(df_categ[columna])

# COMMAND ----------

columna="diag_1"
df =df_categ[columna]
print("Numero de categorias originales {} ".format(len(df.value_counts())))
cantidad = 85
cond = df.value_counts() < cantidad
df = df.replace(cond.index[cond], 'Otros')


# Calculamos la frecuencia de cada categoría en la columna
frequencies = df.value_counts()
# Calculamos el total de valores
n = frequencies.sum()

# Calculamos el coeficiente de Gini
gini = 1 - sum([(fi / n)**2 for fi in frequencies])

percentages = frequencies / len(df) * 100

print("Numero de categorias reconstruidas {} ".format(len(percentages)))
print("-------------------------")
print("valor gini {} ".format(gini))
print("-------------------------")     
print(percentages)

# COMMAND ----------

#remplazar la columan original
columna="diag_1"
#df =df_categ[columna]
cantidad =85
cond = df_categ[columna].value_counts() < cantidad
df_categ[columna] = df_categ[columna].replace(cond.index[cond], 'Otros')
print(df_categ[columna])

# COMMAND ----------

df_categ.info()

# COMMAND ----------

columna_a_excluir ='readmitted'
# variable objetivo
y_model= df_categ[columna_a_excluir]  
# Separar las variables predictoras y la variable objetivo
df_categ = df_categ.drop(columna_a_excluir, axis=1)  # características


# COMMAND ----------

df_categ.info()

# COMMAND ----------

df_categ[["diag_1","diag_2","diag_3"]].describe()

# COMMAND ----------

one_hot_encoded = pd.get_dummies(df_categ)

# COMMAND ----------

print(len(one_hot_encoded))
print(len(df_num))
print(len(df_modelado))

# COMMAND ----------

df_modelado =pd.concat([df_num,one_hot_encoded],axis=1)

# COMMAND ----------

df_modelado.columns

# COMMAND ----------



X = df_modelado  # características
y = y_model  # variable objetivo

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# COMMAND ----------

conteo_val = y_val.value_counts(normalize=True)*100
print("para evaluacion se tienen {} valores, y se distribuyen {} \n".format(y_val.count(),conteo_val))
conteo_train = y_train.value_counts(normalize=True)*100
print("para entrenamiento se tienen {} valores, y se distribuyen {} \n".format(y_train.count(),conteo_train))


# COMMAND ----------

# Almacenar los valores de la inercia en una lista
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_train)
    inertia.append(kmeans.inertia_)

# Plotear los valores de la inercia para cada número de clusters
plt.plot(range(1, 11), inertia)
plt.title('Técnica del codo')
plt.xlabel('Número de clusters')
plt.ylabel('Inercia')
plt.show()

# COMMAND ----------

    kmeans_modelo = KMeans(n_clusters=6,algorithm="auto",init="k-means++")
    kmeans_modelo.fit(X_train)
    centroides = kmeans_modelo.cluster_centers_
    etiquetas = kmeans_modelo.labels_

# COMMAND ----------

plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=etiquetas)
plt.scatter(centroides[:, 0], centroides[:, 1], marker='*', s=300, c='r')
plt.show()

# COMMAND ----------

    kmeans_modelo2 = KMeans(n_clusters=6,algorithm="auto",init="k-means++")
    kmeans_modelo2.fit(X_train)
    centroides2 = kmeans_modelo2.cluster_centers_
    etiquetas2 = kmeans_modelo2.labels_

# COMMAND ----------

plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=etiquetas2)
plt.scatter(centroides2[:, 0], centroides2[:, 1], marker='*', s=300, c='r')
plt.show()

# COMMAND ----------

print(len(etiquetas))
print(len(y_train))

tabla_contingencia = pd.crosstab(y_train, etiquetas, normalize="columns")

# Imprimir la tabla de contingencia
print(tabla_contingencia)

# COMMAND ----------

print(len(etiquetas2))
print(len(y_train))

tabla_contingencia = pd.crosstab(y_train, etiquetas2, normalize="columns")

# Imprimir la tabla de contingencia
print(tabla_contingencia)

# COMMAND ----------

print(y_train.value_counts())

# COMMAND ----------

    kmeans_modelo3 = KMeans(n_clusters=6,algorithm="auto",init="random")
    kmeans_modelo3.fit(X_train)
    centroides3 = kmeans_modelo3.cluster_centers_
    etiquetas3 = kmeans_modelo3.labels_

# COMMAND ----------

print(len(etiquetas3))
print(len(y_train))

tabla_contingencia = pd.crosstab(y_train, etiquetas3, normalize="columns")

# Imprimir la tabla de contingencia
print(tabla_contingencia)

# COMMAND ----------

#El Supervised Clustering Editing (SCE)
#https://www.researchgate.net/figure/Editing-a-dataset-using-supervised-clustering_fig3_220924845
#https://www.researchgate.net/profile/Christoph-Eick/publication/220924845_Using_Supervised_Clustering_to_Enhance_Classifiers/links/55156b050cf2f7d80a32e40c/Using-Supervised-Clustering-to-Enhance-Classifiers.pdf

# COMMAND ----------

#Version spark

# COMMAND ----------

df = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/rlascocz@gmail.com/diabetic_data.csv")

# COMMAND ----------

columnas_numericas = df_num.columns

# Convertir las columnas a tipo float utilizando un loop
for columna in columnas_numericas:
    df = df.withColumn(columna, col(columna).cast("float"))


# COMMAND ----------

valores=df.select("readmitted").distinct().collect()
print(valores)


# COMMAND ----------

from pyspark.sql.functions import when
# columnas eliminadas porque estan fuera de contexto del problema
drop_col=['encounter_id','patient_nbr',"examide",'examide', 'weight','citoglipton',"payer_code","medical_specialty","max_glu_serum","A1Cresult"]

for columna in drop_col:
    df = df.drop(columna)
columnas=['diag_1','diag_2','diag_3']
for column in columnas:
    df = df.withColumn(column, split(df[column], "\.").getItem(0))
    df = df.withColumn(column, regexp_replace(column, "[^0-9]", ""))
    
df = df.select([col(c) for c in columnas_importancia])
df_categ = df.select([col(c) for c in categorical_cols])
df_num = df.select([col(c) for c in numeric_cols])
#valores_permitidos = [1, 2, 3]
#nueva_etiqueta = 'otro'
#df_categ = df.withColumn('diag_1', when(df[''diag_1''].isin(valores_permitidos), df['col_a_reemplazar']).otherwise(nueva_etiqueta))
