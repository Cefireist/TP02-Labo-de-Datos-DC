#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laboratorio de datos - Verano 2025
Trabajo Práctico N° 2

Integrantes:
- Sebastian Ceffalotti - sebastian.ceffalotti@gmail.com
- Aaron Cuellar - aaroncuellar2003@gmail.com
- Rodrigo Coppa - rodrigo.coppa98@gmail.com

Descripción:


Detalles técnicos:


"""

# %% IMPORTACION DE LIBRERIAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# %% LECTURA DE ARCHIVOS


#rutas
_ruta_actual = os.getcwd()
_ruta_mnistc = os.path.join(_ruta_actual, 'mnist_c_fog_tp.csv')

# lectura mnistc, con el index_col podes decirle que columna usar de indice :)
mnistc = pd.read_csv(_ruta_mnistc, index_col = 0)
labels = mnistc["labels"]
# Guardo los pixeles en X 
X = mnistc.drop(columns = ["labels"]) 

#%% EJEMPLO PARA GRAFICAR UNA IMAGEN

img = np.array(X.iloc[0]).reshape((28,28))
plt.imshow(img, cmap='gray') 
plt.title(f'Dígito: {labels.iloc[0]}')
plt.show()

#%% FUNCION PARA GRAFICAR 10 imagenes de un sigito, semilla es para que sea al azar
def graficarDigitos(digito, semilla):
    # selecciono las imágenes del dígito
    digitos = X[labels == digito]
    
    # elijo 10 imágenes aleatorias
    muestras = digitos.sample(10, random_state=semilla)
    imagenes = muestras.values.reshape(10, 28, 28)
    
    # Grafico
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5))
    
    indice = 0
    num_filas = axes.shape[0]
    num_columnas = axes.shape[1]
    for i in range(num_filas):
        for j in range(num_columnas):
            axes[i, j].imshow(imagenes[indice], cmap='gray')
            axes[i, j].axis('off')
            indice += 1
    plt.suptitle(f"Ejemplos de imagenes del digito {digito}")
    plt.show()
#%%
for digito in range(0,10):
    graficarDigitos(digito,1)

#%% ACA VA EL EJERCICIO 2

# Leo los datos para usar, saco los de 0 y 1  solamente
datos = mnistc[mnistc["labels"].isin([0, 1])]
labels_bin = datos["labels"]

# Cuento y veo el balance de clases
contador = labels_bin.value_counts()
print(f"Hay {contador[0]} ceros")
print(f"hay {contador[1]} unos")
print("No esta balanceada la cantidad de clases, por eso las balanceo")

# separo los datos en TRAIN y TEST, hago 80 % train y el resto para test,
# manteniendo el balance de clase

X_train, X_test, y_train, y_test = train_test_split(datos, labels_bin,
test_size = 0.2, stratify = labels_bin, random_state = 160)

datos_ceros = datos[datos["labels"] == 0].drop(columns = "labels")
datos_unos = datos[datos["labels"] == 1].drop(columns = "labels")

#%% GRAFICO LAS IMAGENES PROMEDIO DE CADA DIGITO Y LA RESTA

imagen_promedio_ceros = np.sum(datos_ceros, axis = 0)/len(datos_ceros)
imagen_promedio_unos = np.sum(datos_unos, axis = 0)/len(datos_unos)

img = np.array(imagen_promedio_ceros).reshape((28,28))
plt.imshow(img, cmap='gray') 
plt.title("Imagen promedio del 0")
plt.show()

img = np.array(imagen_promedio_unos).reshape((28,28))
plt.imshow(img, cmap='gray') 
plt.title("Imagen promedio del 1")
plt.show()

resta = np.abs(imagen_promedio_unos-imagen_promedio_ceros)
img = np.array(resta).reshape((28,28))
plt.imshow(img, cmap='gray') 
plt.title("Resta imagenes promedio")
plt.show()

print("""Viendo las imagenes elijo 3 pixeles de manera arbitraria, 
      elijo el del centro, uno a la izquierda y otro a la derecha""")
      
#%% SELECCIONO LOS PIXELES Y EXTRAIGO ESOS DATOS PARA ENTRENAR EL KNN
def obtenerPosColumna(posicion):
    fila, columna = posicion[0], posicion[1]
    return 28*(fila-1) + columna - 1 # resto porque arranca en 0 (?

def entrenar_modelo(X_train_seleccionado, X_test_seleccionado, y_train, y_test, nro_pixeles):
    rango_k = np.arange(1,25,1)
    
    accuracy = []
    precision = []
    recall = []
    
    # pruebo diferentes valores de k
    for k in rango_k:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_seleccionado, y_train)
        # veo las predicciones
        y_pred = knn.predict(X_test_seleccionado)
        # Calculo las metricas
        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"K = {k}")
        print(cm)
        print("")
    # Grafico la precisión en función de k
    plt.figure(figsize=(10, 6))
    plt.plot(rango_k, accuracy, marker='o', linestyle='--', color='r', label = "accuracy")
    plt.plot(rango_k, precision, marker='o', linestyle='--', color='g', label = "precision")
    plt.plot(rango_k, recall, marker='o', linestyle='--', color='b', label = "recall")
    
    plt.title(f'Metricas KNN en función de k con {nro_pixeles} pixeles')
    plt.xlabel('Número de vecinos (k)')
    plt.ylabel('valor de metrica')
    plt.legend()
    plt.xticks(rango_k)  
    plt.grid(True)
    plt.show()
    
#%% ENTRENO EL MODELO eligiendo 1 pixel

pixeles_seleccionados = [[14, 14]]
columnas_pixeles = []
for pixel in pixeles_seleccionados:
    columnas_pixeles.append(obtenerPosColumna(pixel))

X_train_seleccionado = X_train.iloc[:, columnas_pixeles].values
X_test_seleccionado = X_test.iloc[:, columnas_pixeles].values

entrenar_modelo(X_train_seleccionado, X_test_seleccionado, y_train, y_test, 1)
#%% ENTRENO EL MODELO eligiendo 2 pixeles

pixeles_seleccionados = [[8, 14], [14, 14]]
columnas_pixeles = []
for pixel in pixeles_seleccionados:
    columnas_pixeles.append(obtenerPosColumna(pixel))

X_train_seleccionado = X_train.iloc[:, columnas_pixeles].values
X_test_seleccionado = X_test.iloc[:, columnas_pixeles].values

entrenar_modelo(X_train_seleccionado, X_test_seleccionado, y_train, y_test, 2)
#%% ENTRENO EL MODELO eligiendo 3 pixeles

pixeles_seleccionados = [[8, 14], [14, 14], [22, 14]]
columnas_pixeles = []
for pixel in pixeles_seleccionados:
    columnas_pixeles.append(obtenerPosColumna(pixel))

X_train_seleccionado = X_train.iloc[:, columnas_pixeles].values
X_test_seleccionado = X_test.iloc[:, columnas_pixeles].values

entrenar_modelo(X_train_seleccionado, X_test_seleccionado, y_train, y_test, 3)

#%% ENTRENO EL MODELO eligiendo 4 pixeles

pixeles_seleccionados = [[8, 14], [11,14], [14, 14], [22, 14]]
columnas_pixeles = []
for pixel in pixeles_seleccionados:
    columnas_pixeles.append(obtenerPosColumna(pixel))

X_train_seleccionado = X_train.iloc[:, columnas_pixeles].values
X_test_seleccionado = X_test.iloc[:, columnas_pixeles].values

entrenar_modelo(X_train_seleccionado, X_test_seleccionado, y_train, y_test, 4)


#%% ENTRENO un modelo eligiendo otra cantidad de pixeles, uso 5

pixeles_seleccionados = [[8, 14], [11,14], [14, 14], [18,14], [22, 14]]
columnas_pixeles = []
for pixel in pixeles_seleccionados:
    columnas_pixeles.append(obtenerPosColumna(pixel))

X_train_seleccionado = X_train.iloc[:, columnas_pixeles].values
X_test_seleccionado = X_test.iloc[:, columnas_pixeles].values

entrenar_modelo(X_train_seleccionado, X_test_seleccionado, y_train, y_test, 5)

#%%