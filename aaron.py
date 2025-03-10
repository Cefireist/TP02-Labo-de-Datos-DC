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
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
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
            
# Funcion que dado valores de la imagen, un modelo, clasifica
def clasificador_numeros(img_data, clasificador):
    num_pred = clasificador.predict(img_data)
    print(f"valor predicho por el modelo: {num_pred[0]}")
            

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


#%% Ejercicio 2
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
y = ceros_y_unos['labels']  # etiquetas ceros_y_unos

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50, shuffle=True)

#%% Con X_train, y_train entreno el modelo, con X_test, y_test lo testeo y evalúo métricas

# prueba solo con 3 atributos
X_train_first_3 = X_train.iloc[:, :3]
X_test_first_3 = X_test.iloc[:, :3]

# Crear y ajustar el modelo KNN con k=3
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_first_3, y_train)  # entreno con los primeros 3 atributos

# Predecir sobre el conjunto de test
y_pred_3 = knn.predict(X_test_first_3)

# Calcular la exactitud (accuracy)
accuracy_3 = round(accuracy_score(y_test, y_pred_3), 2)  # medimos exactitud comparando valores predecidos y reales
print(f"Exactitud con 3 atributos: {accuracy_3}")


#%% Elijo los 3 atributos con mayor desviación 

n = 3

# Calcular la desviacion estandar por pixel (columna)
desviacion_0 = np.std(ceros.to_numpy(), axis=0)
desviacion_1 = np.std(unos.to_numpy(), axis=0)

# Agregar la desviacion estándar como fila en el DataFrame
ceros.loc['desviacion'] = desviacion_0
unos.loc['desviacion'] = desviacion_1

# Agarrar n columnas con mayor desviación
cols_mayor_desv_0 = np.argsort(desviacion_0)[-n:]  # indices de las n columnas con mayor desviacion en ceros
cols_mayor_desv_1 = np.argsort(desviacion_1)[-n:]  # indices de las n columnas con mayor desviacion en unos

# Unir los indices unicos de ambas clases
cols_seleccionadas = np.unique(np.concatenate([cols_mayor_desv_0, cols_mayor_desv_1]))

# Seleccionar las columnas correspondientes en X_train y X_test
X_train_n_desviacion = X_train.iloc[:, cols_seleccionadas]
X_test_n_desviacion = X_test.iloc[:, cols_seleccionadas]

# Crear y ajustar el modelo knn con k=3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_n_desviacion, y_train) 

# Predecir sobre el conjunto de test
y_pred_n = knn.predict(X_test_n_desviacion)

# Calcular metricas
accuracy_n = round(accuracy_score(y_test, y_pred_n), 2)
recall_n = round(recall_score(y_test, y_pred_n, average="binary"), 2)
f1_n = round(f1_score(y_test, y_pred_n, average="binary"), 2)

print(f"Exactitud con {len(cols_seleccionadas)} atributos (con mayor desviacion): {accuracy_n}")
print(f"Recall: {recall_n}")
print(f"F1-score: {f1_n}")


#%% distintos valores de k

n = 3

# calcular la desviacion estandar por pixel (columna)
desviacion_0 = np.std(ceros.to_numpy(), axis=0)
desviacion_1 = np.std(unos.to_numpy(), axis=0)

# agregar la desviacion estandar como fila en el df
ceros.loc['desviacion'] = desviacion_0
unos.loc['desviacion'] = desviacion_1

# agarrar n columnas con mayor desviacion
cols_mayor_desv_0 = np.argsort(desviacion_0)[-n:]
cols_mayor_desv_1 = np.argsort(desviacion_1)[-n:]  

# uno los indices de ambas clases
cols_seleccionadas = list(cols_mayor_desv_0)

# seleccionar las columnas correspondientes en X_train y X_test
X_train_n_desviacion = X_train.iloc[:, cols_seleccionadas]
X_test_n_desviacion = X_test.iloc[:, cols_seleccionadas]

# crear y ajustar el modelo knn con k=1
k = 1
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_n_desviacion, y_train)  

# predecir sobre el conjunto de test
y_pred_n = knn.predict(X_test_n_desviacion)

# calcular metricas
accuracy_n = round(accuracy_score(y_test, y_pred_n), 2)
recall_n = round(recall_score(y_test, y_pred_n, average="binary"), 2)  
f1_n = round(f1_score(y_test, y_pred_n, average="binary"), 2)

print('--------------------')
print(f"Exactitud con k = {k}: {accuracy_n}")
print(f"Recall: {recall_n}")
print(f"F1-score: {f1_n}")

#%%
# distintos valores de k
k = 10
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_n_desviacion, y_train)  

# predecir sobre el conjunto de test
y_pred_n = knn.predict(X_test_n_desviacion)

# calcular metricas
accuracy_n = round(accuracy_score(y_test, y_pred_n), 2)
recall_n = round(recall_score(y_test, y_pred_n, average="binary"), 2)  
f1_n = round(f1_score(y_test, y_pred_n, average="binary"), 2)

print('--------------------')
print(f"Exactitud con k = {k}: {accuracy_n}")
print(f"Recall: {recall_n}")
print(f"F1-score: {f1_n}")

#%%
# distintos valores de k
k = 150
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_n_desviacion, y_train)  

# predecir sobre el conjunto de test
y_pred_n = knn.predict(X_test_n_desviacion)

# calcular metricas
accuracy_n = round(accuracy_score(y_test, y_pred_n), 2)
recall_n = round(recall_score(y_test, y_pred_n, average="binary"), 2)  
f1_n = round(f1_score(y_test, y_pred_n, average="binary"), 2)

print('--------------------')
print(f"Exactitud con k = {k}: {accuracy_n}")
print(f"Recall: {recall_n}")
print(f"F1-score: {f1_n}")


#%% pruebo con n atributos segun su desviacion
n = 40

# calcular la desviacion estandar por pixel (columna)
desviacion_0 = np.std(ceros.to_numpy(), axis=0)
desviacion_1 = np.std(unos.to_numpy(), axis=0)

# agregar la desviacion estandar como fila en el dataframe
ceros.loc['desviacion'] = desviacion_0
unos.loc['desviacion'] = desviacion_1

# agarrar n columnas con mayor desviacion
cols_mayor_desv_0 = np.argsort(desviacion_0)[-n:]
cols_mayor_desv_1 = np.argsort(desviacion_1)[-n:]  

# uno los indices de ambas clases, pero si hay repes que busque otro/s de forma que haya n en total
cols_seleccionadas = list(cols_mayor_desv_0)

# seleccionar las columnas correspondientes en x_train y x_test
X_train_n_desviacion = X_train.iloc[:, cols_seleccionadas]
X_test_n_desviacion = X_test.iloc[:, cols_seleccionadas]

# crear y ajustar el modelo knn con k=3
k = 3
KNN = KNeighborsClassifier(n_neighbors=k)
KNN.fit(X_train_n_desviacion, y_train)  

# predecir sobre el conjunto de test
y_pred_n = KNN.predict(X_test_n_desviacion)

# calcular metricas
accuracy_n = round(accuracy_score(y_test, y_pred_n), 2)
recall_n = round(recall_score(y_test, y_pred_n, average="binary"), 2)  
f1_n = round(f1_score(y_test, y_pred_n, average="binary"), 2)

print('--------------------')
print(f"exactitud con {len(cols_seleccionadas)} atributos con mayor desviacion: {accuracy_n}")
print(f"recall: {recall_n}")
print(f"f1-score: {f1_n}")


#%% Ejercicio 3
    ###########
    
#%%

# dividir en entrenamiento y validacion 
X_train, X_val, y_train, y_val = train_test_split(mnistc.drop(columns=['indice', 'labels']), 
                                                  mnistc['labels'], 
                                                  test_size=0.2, 
                                                  random_state=10)

# ajustar el modelo de arbol de decision para distintas profundidades
profundidades = range(1, 11)
accuracy_scores = []

for profundidad in profundidades:
    # crear el clasificador con la profundidad actual
    arbol = DecisionTreeClassifier(max_depth=profundidad, random_state=10)
    arbol.fit(X_train.drop(columns=['indice', 'labels']), X_train['labels'])
    
    # predecir en el conjunto de validacion
    y_pred = arbol.predict(X_val.drop(columns=['indice', 'labels']))
    
    # calcular la exactitud
    accuracy = accuracy_score(X_val['labels'], y_pred)
    accuracy_scores.append(accuracy)

# graficar los resultados
plt.plot(profundidades, accuracy_scores, marker='o')
plt.xlabel('Profundidad del arbol')
plt.ylabel('Exactitud')
plt.title('Exactitud del arbol de decision vs profundidad')
plt.show()


#%%
# configurar los rangos de los hiperparámetros
depths = range(1, 11)
criterions = ['gini', 'entropy']

# inicializar una lista para almacenar los resultados
results = []

# iterar sobre diferentes combinaciones de hiperparámetros
for depth in depths:
    for criterion in criterions:
        # crear el modelo de árbol de decisión con los hiperparámetros
        arbol = DecisionTreeClassifier(max_depth=depth, criterion=criterion, max_features=None, random_state=10)
        
        # realizar validación cruzada con 5-folds y calcular la exactitud promedio
        cv_scores = cross_val_score(arbol, X_train, y_train, cv=5, scoring='accuracy')
        mean_score = np.mean(cv_scores)
        
        # guardar los resultados para cada configuración
        results.append((depth, criterion, mean_score))

# ordenar los resultados por la exactitud en orden descendente
best_result = sorted(results, key=lambda x: x[2], reverse=True)[0]

# mostrar el mejor modelo
print("mejor configuración de hiperparámetros:")
print(f"profundidad: {best_result[0]}")
print(f"criterio: {best_result[1]}")
print(f"exactitud promedio: {round(best_result[2], 2)}")

#%%
# entrenar el modelo final con los mejores hiperparámetros (profundidad 10, criterio 'entropy')
arbol_model = DecisionTreeClassifier(max_depth=best_result[0], criterion=best_result[1], max_features=None, random_state=10)

# ajustar el modelo con los datos de entrenamiento (todo el conjunto de desarrollo)
arbol_model.fit(X_train, y_train)

# predecir las clases en el conjunto held-out (en este caso, X_val)
y_pred = arbol_model.predict(X_val)

# evaluar el rendimiento del modelo en el conjunto held-out
accuracy_held_out = accuracy_score(y_val, y_pred)
recall_held_out = recall_score(y_val, y_pred, average='macro')  # 'macro' para balance entre clases
f1_held_out = f1_score(y_val, y_pred, average='macro')  # 'macro' para balance entre clases

# mostrar el rendimiento
print(f"precisión en el conjunto held-out: {round(accuracy_held_out, 2)}")
print(f"recall en el conjunto held-out: {round(recall_held_out, 2)}")
print(f"f1-score en el conjunto held-out: {round(f1_held_out, 2)}")

# matriz de confusión
conf_matrix = confusion_matrix(y_val, y_pred)

# visualizar la matriz de confusión
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=True, yticklabels=True)
plt.title("matriz de confusión")
plt.xlabel("predicciones")
plt.ylabel("verdaderos valores")
plt.show()

# reporte de clasificación para todas las clases
print("\nreporte de clasificación:")
print(classification_report(y_val, y_pred))
#%%








