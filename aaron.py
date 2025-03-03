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

# funcion que toma una fila del mnist (img) y la grafica
def graficar_imagen(valores):
    arr_img = valores.to_numpy().reshape(28, 28)
    plt.imshow(arr_img, cmap='gray')
    plt.axis('off')
    plt.show()

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


# grafico algunos digitos
graficar_imagen(data.iloc[1])
graficar_imagen(data.iloc[2025])
graficar_imagen(data.iloc[912])


