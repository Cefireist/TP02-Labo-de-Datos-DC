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
#from sklearn import 
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
    
# funcion que toma 8 filas del mnistc y los grafica en una fig
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

# visualizar distribucion
sns.histplot(etiquetas, bins=10, discrete=True, shrink=0.6)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.xlabel("clases")
plt.ylabel("Frecuencia")
plt.ylim(0, 9000)
plt.title("Distribución de las clases")
plt.show()

# desbalance de las clases
apariciones = etiquetas.value_counts()
desbalance = apariciones.max() - apariciones.min()
print('desbalance entre cantidad de clases:', desbalance)


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














