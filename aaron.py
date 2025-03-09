# -*- coding: utf-8 -*-
"""
Laboratorio de datos - Verano 2025
Trabajo Práctico N° 1 

Integrantes:
- Sebastian Ceffalotti - sebastian.ceffalotti@gmail.com
- Aaron Cuellar - aaroncuellar2003@gmail.com
- Rodrigo Coppa - rodrigo.coppa98@gmail.com

Descripción:
En este script realizamos los graficos necesarios para el realizar
el analisis exploratorio de la fuente de datos MNIST-C (version fog)

Detalles técnicos:
- Lenguaje: Python
- Librerías utilizadas: numpy, matplotlib, seaborn, pandas y scikit-learn
"""

#%% 
################
# Importaciones 
################

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
import os

#%% 
################
# Carga de datos 
################

#%%

#rutas
_ruta_actual = os.getcwd()
_ruta_mnistc = os.path.join(_ruta_actual, 'mnist_c_fog_tp.csv')

# lectura mnistc
mnistc = pd.read_csv(_ruta_mnistc)
mnistc = mnistc.rename(columns={mnistc.columns[0]: 'indice'})

#%% 
################
# Funciones 
################
    
# funcion que toma n filas del mnistc y los grafica en una fig
def graficar_imagenes(df_valores, filas, cols, title):
    fig, ax = plt.subplots(filas, cols)
    fig.suptitle(title)
    
    for i in range(filas):
        for j in range(cols):
            num_fila = cols*i + j
            ax[i, j].imshow(df_valores.iloc[num_fila].to_numpy().reshape(28, 28), cmap='gray')
            ax[i][j].axis('off')
            

#%% 
################
# Codigo 
################

#%% Analisis exploratorio
    #####################
#%%
# tomo solo los datos para las imagenes 
data = mnistc.iloc[:, 1:785]

# valores de que numeros representan
etiquetas = mnistc['labels']

# datos basicos del dataset
print('cantidad de imagenes:', len(data))
print('cantidad de atributos:', len(mnistc.columns))
print(f'tipos de los atributos:\n{mnistc.dtypes}')
print('cantidad de clases:', len(etiquetas.unique()))
print('cantidad de datos nulos:', mnistc.isnull().sum().sum())
print(f'datos descriptivos de las clases:\n{etiquetas.describe().round(2)}')
print('rango de datos img:', data.min().min(), 'a', data.max().max())

#%%
# visualizar distribucion
sns.histplot(etiquetas, bins=10, discrete=True, shrink=0.6)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.xlabel("clases")
plt.ylabel("Frecuencia")
plt.ylim(0, 9000)
plt.title("Distribución de las clases")
plt.show()
#%%
# desbalance de las clases
apariciones = etiquetas.value_counts()
desbalance = apariciones.max() - apariciones.min()
print('desbalance entre cantidad de clases:', desbalance)

#%%
# figuras con 8 graficos de ejemplo por numero
for i in range(10):
    arr = mnistc[mnistc['labels'] == i].iloc[0:8].iloc[:, 1:785] # agarro datos para img de primeros 4 valores con etiqueta num
    graficar_imagenes(arr, 2, 4, f'grafico de 8 numeros {i}')

# grafico 30 imagenes 0 para ver sus similitudes
arr_ceros = mnistc[mnistc['labels'] == 0].iloc[0:30].iloc[:, 1:785]
graficar_imagenes(arr_ceros, 5, 6, 'grafico de 30 ceros')

# grafico 30 imagenes 7 para ver sus similitudes
arr_sietes = mnistc[mnistc['labels'] == 7].iloc[0:30].iloc[:, 1:785]
graficar_imagenes(arr_sietes, 5, 6, 'grafico de 30 sietes')
    
#%%
# visualizacion de la media por pixel de cada clase
fig, ax = plt.subplots(2, 5, figsize=(10, 5)) 
fig.suptitle("Media de cada píxel por clase")
for numero in range(10):
    # imagenes de la clase actual
    data_clase = data[mnistc['labels'] == numero]
    
    # calcular la media por pixel
    media_por_pixel = np.mean(data_clase.to_numpy(), axis=0).reshape(28, 28)

    # graficar
    ax_actual = ax[numero // 5, numero % 5]  # ubicacion
    ax_actual.imshow(media_por_pixel, cmap='gray')
    ax_actual.set_title(f"numero {numero}")
    ax_actual.axis('off')
plt.show()

#%%
# desviacion. las partes mas oscuras son las que menos varian
fig, axes = plt.subplots(2, 5, figsize=(10, 5))  # 2 filas, 5 columnas
fig.suptitle("Desviación estándar de cada píxel por clase")

for numero in range(10):
    # Filtrar imágenes de la clase actual
    data_clase = data[mnistc['labels'] == numero]

    # Calcular la desviación estándar por píxel
    std_por_pixel = np.std(data_clase.to_numpy(), axis=0).reshape(28, 28)

    # Seleccionar el subplot correspondiente
    ax_actual = axes[numero // 5, numero % 5]
    ax_actual.imshow(std_por_pixel, cmap='gray')
    ax_actual.set_title(f"Número {numero}")
    ax_actual.axis('off')
plt.show()


<<<<<<< HEAD

#%% Ejercicio 1
    ###########
#%%

# subconjunto de digitos 0 o 1
ceros_y_unos = mnistc[(mnistc['labels'] == 0) | (mnistc['labels'] == 1)]
ceros = data[mnistc['labels'] == 0]
unos = data[mnistc['labels'] == 1]

print(f'cantidad de muestras con valores 0 o 1 = {len(ceros_y_unos)}')
print(f'cantidad de muestras con valor 0 = {len(ceros)}')
print(f'cantidad de muestras con valor 1 = {len(unos)}')
print(f'diferencia entre clase 0 y clase 1 = {len(unos) - len(ceros)}')


#%%

# train y test split

X = ceros_y_unos.drop(columns=['labels', 'indice'])  # data imagenes ceros_y_unos
y = ceros_y_unos['labels'] # etiquetas ceros_y_unos

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50, shuffle=True)

#%% Con X_train, y_train entreno el modelo, con X_test, y_test lo testeo y evaluo metricas

# prueba solo con 3 atributos

# uso las primeras 3 columnas
X_train_first_3 = X_train.iloc[:, :3]
X_test_first_3 = X_test.iloc[:, :3]

# Crear y ajustar el modelo KNN con k=3
k=3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_first_3, y_train) # entreno con los primeros 3 atributos

# Predecir sobre el conjunto de test
y_pred_3 = knn.predict(X_test_first_3)

# Calcular la exactitud (accuracy)
accuracy_3 = round(accuracy_score(y_test, y_pred_3), 2) # medimos exactitud comparando valores predecidos y reales
print(f"Exactitud con 3 atributos: {accuracy_3}")


#%% Eligo los 3 atributos con mayor desviacion 

# agarrar los 3 puntos con mayor desviacion de cada clase y uso esos atributos para entrenar en cada clase
n = 3

# Calcular la desviación estándar por pixel (columna)
desviacion_0 = np.std(ceros.to_numpy(), axis=0)
desviacion_1 = np.std(unos.to_numpy(), axis=0)

# Agregar la desviacion estandar como fila en el DataFrame
ceros.loc['desviacion'] = desviacion_0
unos.loc['desviacion'] = desviacion_1

# agarrar n cols con mayor desviacion
cols_mayor_desv_0 = np.argsort(desviacion_0)[-n:]  # indices de las n columnas con mayor desviacion en ceros
cols_mayor_desv_1 = np.argsort(desviacion_1)[-n:]  # indices de las n columnas con mayor desviacion en unos

# unir los indices unicos de ambas clases
cols_seleccionadas = np.unique(np.concatenate([cols_mayor_desv_0, cols_mayor_desv_1]))

# seleccionar las columnas correspondientes en X_train y X_test
X_train_n_desviacion = X_train.iloc[:, cols_seleccionadas]
X_test_n_desviacion = X_test.iloc[:, cols_seleccionadas]

# crear y ajustar el modelo knn con k=3
k=3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_n_desviacion, y_train) 

# predecir sobre el conjunto de test
y_pred_n = knn.predict(X_test_n_desviacion)

# calcular metricas
accuracy_n = round(accuracy_score(y_test, y_pred_n), 2)
recall_n = round(recall_score(y_test, y_pred_n, average="binary"), 2)
f1_n = round(f1_score(y_test, y_pred_n, average="binary"), 2)

print(f"Exactitud con {len(cols_seleccionadas)} atributos: {accuracy_n}")
print(f"Recall: {recall_n}")
print(f"F1-score: {f1_n}")


#%% pruebo con n atributos segun su desviacion

# agarrar los 3 puntos con mayor desviacion de cada clase y uso esos atributos para entrenar en cada clase
# numero de puntos que quieres seleccionar por clase (mayor desviación estandar)
n = 40

# calcular la desviación estandar por pixel (columna)
desviacion_0 = np.std(ceros.to_numpy(), axis=0)
desviacion_1 = np.std(unos.to_numpy(), axis=0)

# agregar la desviacion estandar como fila en el DataFrame
ceros.loc['desviacion'] = desviacion_0
unos.loc['desviacion'] = desviacion_1

# agarrar n cols con mayor desviacion
cols_mayor_desv_0 = np.argsort(desviacion_0)[-n:]
cols_mayor_desv_1 = np.argsort(desviacion_1)[-n:]  

# uno los indices de ambas clases, pero si hay repes que busque otro/s de forma que haya n en total
cols_seleccionadas = list(cols_mayor_desv_0)

# seleccionar las columnas correspondientes en X_train y X_test
X_train_n_desviacion = X_train.iloc[:, cols_seleccionadas]
X_test_n_desviacion = X_test.iloc[:, cols_seleccionadas]

# crear y ajustar el modelo knn con k=3
k=3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_n_desviacion, y_train)  

# predecir sobre el conjunto de test
y_pred_n = knn.predict(X_test_n_desviacion)

# calcular metricas
accuracy_n = round(accuracy_score(y_test, y_pred_n), 2)
recall_n = round(recall_score(y_test, y_pred_n, average="binary"), 2)  
f1_n = round(f1_score(y_test, y_pred_n, average="binary"), 2)

print('--------------------')
print(f"Exactitud con {len(cols_seleccionadas)} atributos: {accuracy_n}")
print(f"Recall: {recall_n}")
print(f"F1-score: {f1_n}")

#%%


















=======
#%%
>>>>>>> 077775e9b9a5aedfa4997d9c41ffe94b87163f31











