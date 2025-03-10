#%% DATOS CARATULA
"""
Laboratorio de datos - Verano 2025
Trabajo Práctico N° 2 

Integrantes:
- Sebastian Ceffalotti - sebastian.ceffalotti@gmail.com
- Aaron Cuellar - aaroncuellar2003@gmail.com
- Rodrigo Coppa - rodrigo.coppa98@gmail.com

Descripción:
<<<<<<< HEAD

=======
En este script trabajamos con el conjunto de datos MNIST-C (version fog), lo analizamos y entrenamos modelos de clasificación para predecir clases

Detalles técnicos:
- Lenguaje: Python
- Librerías utilizadas: numpy, matplotlib, seaborn, os, pandas y scikit-learn
>>>>>>> 6c10665a287a3388b4b0cec2eff028fd6c5f15aa
"""
# %% IMPORTACION DE LIBRERIAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn import tree

#%% LECTURA DE DATOS

# rutas
_ruta_actual = os.getcwd()
_ruta_mnistc = os.path.join(_ruta_actual, 'mnist_c_fog_tp.csv')

# lectura mnistc, con el index_col podes decirle que columna usar de indice
mnistc = pd.read_csv(_ruta_mnistc, index_col = 0)
labels = mnistc["labels"]
# Guardo los pixeles en X 
X = mnistc.drop(columns = ["labels"]) 

#%% DECLARACION DE FUNCIONES
#%% FUNCION QUE CALCULA LA INTENSIDAD PROMEDIO DE UN DIGITO
def img_promedio_digito(datos, digito):
    datos_digito = datos[datos["labels"] == digito].drop(columns = "labels")
    img_promedio = np.sum(datos_digito, axis = 0)/len(datos_digito)
    return img_promedio
#%% FUNCION PARA GRAFICAR 15 IMAGENES ALEATORIAS DE UN DIGITO
def graficarMuestraDigitos(digito, semilla):
    # selecciono las imágenes del dígito
    digitos = X[labels == digito]
    
    # elijo 15 imágenes aleatorias
    muestras = digitos.sample(15, random_state=semilla)
    imagenes = muestras.values.reshape(15, 28, 28)
    
    # Grafico
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(10, 5))
    
    indice = 0
    num_filas = axes.shape[0]
    num_columnas = axes.shape[1]
    for i in range(num_filas):
        for j in range(num_columnas):
            axes[i, j].imshow(imagenes[indice], cmap='gray')
            axes[i, j].axis('off')
            indice += 1
    plt.suptitle(f"Ejemplos de imagenes del digito {digito}", fontsize = 18)
    plt.show()
#%% FUNCION QUE CALCULA LA POSICION DE LA COLUMNA DE UN PIXEL DE ACUERDO A SUS COORDENADAS
def obtenerPosColumna(posicion):
    fila, columna = posicion[0], posicion[1]
    return 28*(fila-1) + columna - 1 # resto porque arranca en 0
#%% FUNCION QUE ENTRENA UN KNN CON LOS PIXELES SELECCIONADOS
def entrenar_modelo(X_train_seleccionado, X_test_seleccionado, y_train, y_test, titulo):
    rango_k = np.arange(1, 25, 1)
    
    accuracy_train = []
    precision_train = []
    recall_train = []
    accuracy_test = []
    precision_test = []
    recall_test = []
    
    for k in rango_k:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_seleccionado, y_train)
        
        # Predicciones para TRAIN
        y_train_pred = knn.predict(X_train_seleccionado)
        accuracy_train.append(accuracy_score(y_train, y_train_pred))
        precision_train.append(precision_score(y_train, y_train_pred, average='macro'))
        recall_train.append(recall_score(y_train, y_train_pred, average='macro'))
        
        # Predicciones para TEST
        y_test_pred = knn.predict(X_test_seleccionado)
        accuracy_test.append(accuracy_score(y_test, y_test_pred))
        precision_test.append(precision_score(y_test, y_test_pred, average='macro'))
        recall_test.append(recall_score(y_test, y_test_pred, average='macro'))
    
    return accuracy_train, precision_train, recall_train, accuracy_test, precision_test, recall_test
#%% FUNCION QUE ENTRENA UN ARBOL SEGUN HIPERPARAMETROS ELEGIDOS, CRITERIO Y MAXIMA PROFUNDIDAD
def EntrenarArbol(alturas, kf, criterio, X_dev, y_dev):
    # matrices donde almacenar los resultados
    resultados_accuracy_train = np.zeros((nsplits, len(alturas)))
    resultados_precision_train = np.zeros((nsplits, len(alturas)))
    resultados_recall_train = np.zeros((nsplits, len(alturas)))
    
    resultados_accuracy_test = np.zeros((nsplits, len(alturas)))
    resultados_precision_test = np.zeros((nsplits, len(alturas)))
    resultados_recall_test = np.zeros((nsplits, len(alturas)))
    
    # en cada fold, entreno cada modelo y guardo todo en las matrices de resultados
    for i, (train_index, test_index) in enumerate(kf.split(X_dev, y_dev)):
        kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
        kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]
        
        for j, hmax in enumerate(alturas):
            # entreno al arbol
            arbol = tree.DecisionTreeClassifier(max_depth=hmax, criterion=criterio)
            arbol.fit(kf_X_train, kf_y_train)
            
            # obtengo la prediccion y las metricas para el conjunto de train
            pred_train = arbol.predict(kf_X_train)
            accuracy_train = accuracy_score(kf_y_train, pred_train)
            precision_train = precision_score(kf_y_train, pred_train, average='macro')
            recall_train = recall_score(kf_y_train, pred_train, average='macro')
            
            # obtengo la prediccion y las metricas para el conjunto de test
            pred_test = arbol.predict(kf_X_test)
            accuracy_test = accuracy_score(kf_y_test, pred_test)
            precision_test = precision_score(kf_y_test, pred_test, average='macro')
            recall_test = recall_score(kf_y_test, pred_test, average='macro')
            
            # guardo los resultados en las matrices
            resultados_accuracy_train[i, j] = accuracy_train
            resultados_precision_train[i, j] = precision_train
            resultados_recall_train[i, j] = recall_train
            
            resultados_accuracy_test[i, j] = accuracy_test
            resultados_precision_test[i, j] = precision_test
            resultados_recall_test[i, j] = recall_test
    
    return resultados_accuracy_train, resultados_accuracy_test, resultados_precision_train, resultados_precision_test, resultados_recall_train, resultados_recall_test
#%% FUNCION QUE RECIBE LAS METRICAS DE TRAIN Y TEST Y LAS GRAFICA
def GraficarMetricasArbol(alturas,scores_accuracy_train,scores_accuracy_test,
                          scores_precision_test, scores_precision_train,scores_recall_test,scores_recall_train, criterio):
    
    plt.figure(figsize=(12, 6))
    plt.plot(alturas, scores_accuracy_train, marker='o', linestyle='--', color='r', label="Train Accuracy")
    plt.plot(alturas, scores_accuracy_test, marker='o', linestyle='--', color='g', label="Test Accuracy")
    plt.xlabel("Profundidad maxima", fontsize = 18)
    plt.ylabel("Accuracy", fontsize = 18)
    plt.title(f"Curvas de Accuracy usando {criterio}", fontsize = 18)
    plt.legend(fontsize = 18)
    plt.tick_params(axis='both', which='major', labelsize=18) 
    plt.xticks(alturas)
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(alturas, scores_precision_train, marker='o', linestyle='--', color='r', label="Train Precision")
    plt.plot(alturas, scores_precision_test, marker='o', linestyle='--', color='g', label="Test Precision")
    plt.xlabel("Profundidad maxima", fontsize = 18)
    plt.ylabel("Precision", fontsize = 18)
    plt.title(f"Curvas de Precision usando {criterio}", fontsize = 18)
    plt.legend(fontsize = 18)
    plt.tick_params(axis='both', which='major', labelsize=18) 
    plt.xticks(alturas)
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(alturas, scores_recall_train, marker='o', linestyle='--', color='r', label="Train Recall")
    plt.plot(alturas, scores_recall_test, marker='o', linestyle='--', color='g', label="Test Recall")
    plt.xlabel("Profundidad maxima", fontsize = 18)
    plt.ylabel("Recall", fontsize = 18)
    plt.title(f"Curvas de Recall usando {criterio}", fontsize = 18)
    plt.legend(fontsize = 18)
    plt.tick_params(axis='both', which='major', labelsize=18) 
    plt.xticks(alturas)
    plt.grid(True)
    plt.show()

#%% EJERCICIO 1.a
#%% GRAFICO LA IMAGEN PROMEDIO DE TODOS LOS DIGITOS

fig, axes = plt.subplots(3, 5, figsize=(12, 8))
axes = axes.flatten()
suma_todos_digitos = np.zeros((28, 28))

for digito in range(0,10):
    img_prom = img_promedio_digito(mnistc, digito)
    img = np.array(img_prom).reshape((28,28))
    suma_todos_digitos += img
    
    im = axes[digito].imshow(img, cmap='inferno')
    axes[digito].set_title(f"Digito {digito}")

# promedio de todos los digitos
suma_todos_digitos = suma_todos_digitos/10

# oculto ejes vacios y muestro el promedio entre todos los digitos
for i in [10, 11, 13, 14]:
    axes[i].axis('off')
axes[12].imshow(suma_todos_digitos, cmap='inferno')
axes[12].set_title("Suma de todos los dígitos")

# Al usar im estoy usando la ultima como referencia de intensidad
fig.suptitle("Promedio de intensidad por dígito", fontsize=18)
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  
cbar = fig.colorbar(im, cax=cax)
cbar.set_label("Intensidad promedio", fontsize=14)

#%% GRAFICO LA REGION DE LOS PIXELES MENOS RELEVANTES

umbral = 100 # lo elijo arbitrariamente viendo la imagen anterior
# creo una mascara binaria para graficar
mascara = suma_todos_digitos >= umbral
pixeles_menores_100 = np.sum(suma_todos_digitos < umbral)

print(f"Porcentaje de píxeles con intensidad menor a 100: {100*pixeles_menores_100/784}")
plt.imshow(mascara, cmap='gray')
plt.title("Píxeles con intensidad menor a 100 (negros)")
plt.show()
#%% EJERCICIO 1.b
#%% CALCULO LAS DISTANCIAS ENTRE LAS IMAGENES PROMEDIO DE CADA DIGITO
# pienso cada imagen como un vector en R^784, entonces la distancia 
# es la norma de la diferencia entre dos vectores

imgs_promedio = {}
for digito in range(0,10):
    imagenes_digito = mnistc[mnistc["labels"] == digito].drop(columns="labels")
    imgs_promedio[digito] = np.mean(imagenes_digito, axis=0)

# creo una matriz donde guardar las distancias promedio entre cada digito, y las guardo
distancias = np.zeros((10, 10))  
for i in range(10):
    for j in range(10):
        distancias[i, j] = np.linalg.norm(imgs_promedio[i] - imgs_promedio[j])

# redondeo a enteros
distancias = distancias.astype(int) 
# grafico la matriz
sns.heatmap(distancias, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Digito')
plt.ylabel('Digito')
plt.title('Matriz de distancias entre digitos')
plt.show()

#%% EJERCICIO 1.c
#%% GRAFICO 15 MUESTRAS ALEATORIAS DE CADA DIGITO, EN TODOS LOS DIGITOS HAY MUCHA VARIACION
plt.figure(figsize=(10, 10))
for digito in range(0,10):
    graficarMuestraDigitos(digito,7)
    
#%% EJERCICIO 2
#%% EXTRAIGO LOS DATOS DE LOS 0 Y 1, VEO EL BALANCE Y SEPARO EN TRAIN Y TEST
datos = mnistc[mnistc["labels"].isin([0, 1])]
labels_bin = datos["labels"]

# Cuento y veo el balance de clases
contador = labels_bin.value_counts()
print(f"Hay {contador[0]} ceros")
print(f"hay {contador[1]} unos")

"""
separo los datos en TRAIN y TEST, hago 80 % train y el resto para test,
manteniendo el balance de clase, uso como metrica accuracy ya que mas o menos estan
balanceadas las clases. De todos modos calculo tambien recall y precision para ver.
"""

X_train, X_test, y_train, y_test = train_test_split(datos, labels_bin,
test_size = 0.2, stratify = labels_bin, random_state = 160)

X_train = X_train.drop(columns="labels")
X_test = X_test.drop(columns = "labels")
#%% GRAFICO LOS PROMEDIOS DEL 0, EL 1 Y SU RESTA. POR INSPECCION DECIDO QUE PIXELES USAR

fig, ax = plt.subplots(1, 3, figsize=(12, 6))

img_prom_0 = img_promedio_digito(datos, 0)
ax[0].imshow(np.array(img_prom_0).reshape((28, 28)), cmap='gray')
ax[0].set_title("Promedio del 0", fontsize = 18)

img_prom_1 = img_promedio_digito(datos, 1)
ax[1].imshow(np.array(img_prom_1).reshape((28, 28)), cmap='gray')
ax[1].set_title("Promedio del 1", fontsize = 18)

resta = np.abs(img_prom_1-img_prom_0)
# uso como referencia esta imagen para la barra
im = ax[2].imshow(np.array(resta).reshape((28, 28)), cmap='gray')
ax[2].set_title("Resta", fontsize = 18)

fig.suptitle("Promedio de intensidad por dígito", fontsize=18)
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  
cbar = fig.colorbar(im, cax=cax)  # Usar la imagen 'im' como referencia
cbar.set_label("Intensidad promedio", fontsize=16)
plt.show()

"""
Viendo las imagenes elijo pixeles de manera arbitraria, 
elijo el del centro, uno a la izquierda y otro a la derecha por ejemplo
"""   
#%% ENTRENO EL MODELO KNN ELIGIENDO DIVERSOS PIXELES ARBITRARIAMENTE

# SE ENTRENA ELIGIENDO 1 PIXEL, EL CENTRAL
pixeles_seleccionados_1 = [[14, 14]]
columnas_pixeles_1 = [obtenerPosColumna(pixel) for pixel in pixeles_seleccionados_1]

X_train_seleccionado_1 = X_train.iloc[:, columnas_pixeles_1].values
X_test_seleccionado_1 = X_test.iloc[:, columnas_pixeles_1].values

accuracy_train_1, precision_train_1, recall_train_1, accuracy_test_1, precision_test_1, recall_test_1 = entrenar_modelo(X_train_seleccionado_1, X_test_seleccionado_1, y_train, y_test, "Usando 1 pixel")

# SE ENTRENA ELIGIENDO 3 PIXELES, EL CENTRAL Y DOS A LOS COSTADOS, DONDE HAY MAXIMOS DEL 0
pixeles_seleccionados_3 = [[8, 14], [14, 14], [22, 14]]
columnas_pixeles_3 = [obtenerPosColumna(pixel) for pixel in pixeles_seleccionados_3]

X_train_seleccionado_3 = X_train.iloc[:, columnas_pixeles_3].values
X_test_seleccionado_3 = X_test.iloc[:, columnas_pixeles_3].values

accuracy_train_3, precision_train_3, recall_train_3, accuracy_test_3, precision_test_3, recall_test_3 = entrenar_modelo(X_train_seleccionado_3, X_test_seleccionado_3, y_train, y_test, "Usando 3 pixeles")

# SE ENTRENA ELIGIENDO 14 PIXELES, LOS DE LA FILA CENTRAL
pixeles_seleccionados_14 = [[14, col] for col in range(15)]
columnas_pixeles_14 = [obtenerPosColumna(pixel) for pixel in pixeles_seleccionados_14]

X_train_seleccionado_14 = X_train.iloc[:, columnas_pixeles_14].values
X_test_seleccionado_14 = X_test.iloc[:, columnas_pixeles_14].values

accuracy_train_14, precision_train_14, recall_train_14, accuracy_test_14, precision_test_14, recall_test_14 = entrenar_modelo(X_train_seleccionado_14, X_test_seleccionado_14, y_train, y_test, "Usando 14 pixeles")

#%% GRAFICOS DE ACCURACY
fig, ax = plt.subplots(3, 1, figsize=(12, 18))

k = np.arange(1,25,1)
ax[0].plot(k, accuracy_train_1, marker='o', linestyle='--', color='r', label="Train 1 pixel")
ax[0].plot(k, accuracy_test_1, marker='o', linestyle='-', color='g', label="Test 1 pixel")
ax[0].set_title("Accuracy con 1 pixel", fontsize=18)

ax[1].plot(k, accuracy_train_3, marker='o', linestyle='--', color='r', label="Train 3 pixeles")
ax[1].plot(k, accuracy_test_3, marker='o', linestyle='-', color='g', label="Test 3 pixeles")
ax[1].set_title("Accuracy con 3 pixeles", fontsize=18)

ax[2].plot(k, accuracy_train_14, marker='o', linestyle='--', color='r', label="Train 14 pixeles")
ax[2].plot(k, accuracy_test_14, marker='o', linestyle='-', color='g', label="Test 14 pixeles")
ax[2].set_title("Accuracy con 14 pixeles", fontsize=18)

for a in ax:
    a.legend(fontsize=14)
    a.set_xlabel('Numero de vecinos (k)', fontsize=18)
    a.set_ylabel('Accuracy', fontsize=18)
    a.set_xticks(k)
    a.tick_params(axis='both', which='major', labelsize=14)
    a.grid(True)

plt.tight_layout()
plt.show()
#%% GRAFICOS DE PRECISION
fig, ax = plt.subplots(3, 1, figsize=(12, 18))

ax[0].plot(k, precision_train_1, marker='o', linestyle='--', color='r', label="Train 1 pixel")
ax[0].plot(k, precision_test_1, marker='o', linestyle='-', color='g', label="Test 1 pixel")
ax[0].set_title("Precision con 1 pixel", fontsize=18)

ax[1].plot(k, precision_train_3, marker='o', linestyle='--', color='r', label="Train 3 pixeles")
ax[1].plot(k, precision_test_3, marker='o', linestyle='-', color='g', label="Test 3 pixeles")
ax[1].set_title("Precision con 3 pixeles", fontsize=18)

ax[2].plot(k, precision_train_14, marker='o', linestyle='--', color='r', label="Train 14 pixeles")
ax[2].plot(k, precision_test_14, marker='o', linestyle='-', color='g', label="Test 14 pixeles")
ax[2].set_title("Precision con 14 pixeles", fontsize=18)

for a in ax:
    a.legend(fontsize=14)
    a.set_xlabel('Numero de vecinos (k)', fontsize=18)
    a.set_ylabel('Precision', fontsize=18)
    a.set_xticks(k)
    a.tick_params(axis='both', which='major', labelsize=14)
    a.grid(True)

plt.tight_layout()
plt.show()

#%% GRAFICOS DE RECALL
fig, ax = plt.subplots(3, 1, figsize=(12, 18))

ax[0].plot(k, recall_train_1, marker='o', linestyle='--', color='r', label="Train 1 pixel")
ax[0].plot(k, recall_test_1, marker='o', linestyle='-', color='g', label="Test 1 pixel")
ax[0].set_title("Recall con 1 pixel", fontsize=18)

ax[1].plot(k, recall_train_3, marker='o', linestyle='--', color='r', label="Train 3 pixeles")
ax[1].plot(k, recall_test_3, marker='o', linestyle='-', color='g', label="Test 3 pixeles")
ax[1].set_title("Recall con 3 pixeles", fontsize=18)

ax[2].plot(k, recall_train_14, marker='o', linestyle='--', color='r', label="Train 14 pixeles")
ax[2].plot(k, recall_test_14, marker='o', linestyle='-', color='g', label="Test 14 pixeles")
ax[2].set_title("Recall con 14 pixeles", fontsize=18)

for a in ax:
    a.legend(fontsize=14)
    a.set_xlabel('Numero de vecinos (k)', fontsize=18)
    a.set_ylabel('Recall', fontsize=18)
    a.set_xticks(k)
    a.tick_params(axis='both', which='major', labelsize=14)
    a.grid(True)

plt.tight_layout()
plt.show()
#%% PREPARO LOS DATOS PARA ENTRENAR UN MODELO BASANDOSE EN DISTANCIAS

# obtengo la imagen promedio del 0 y el 1
imgs_promedio = {}
for digito in range(0,2):
    imagenes_digito = mnistc[mnistc["labels"] == digito].drop(columns="labels")
    imgs_promedio[digito] = np.mean(imagenes_digito, axis=0)
    
# calculo las distancias en train y test respecto al promedio del 0 y el 1
distancias_al_0_train = np.linalg.norm(X_train - imgs_promedio[0], axis=1)
distancias_al_1_train = np.linalg.norm(X_train - imgs_promedio[1], axis=1)
distancias_al_0_test = np.linalg.norm(X_test - imgs_promedio[0], axis=1)
distancias_al_1_test = np.linalg.norm(X_test - imgs_promedio[1], axis=1)
# agrego las distancias a los dataframes
X_train_dist = X_train.copy()
X_test_dist = X_test.copy()
X_train_dist["distancia_al_0"] = distancias_al_0_train
X_train_dist["distancia_al_1"] = distancias_al_1_train
X_test_dist["distancia_al_0"] = distancias_al_0_test
X_test_dist["distancia_al_1"] = distancias_al_1_test
#%% ENTRENO EL MODELO KNN, MIRO LA DISTANCIA DE CADA IMAGEN A LA PROMEDIO DE 0 Y 1
X_train_seleccionado = X_train_dist[["distancia_al_0", "distancia_al_1"]]
X_test_seleccionado = X_test_dist[["distancia_al_0", "distancia_al_1"]]
accuracy_train, precision_train, recall_train, accuracy_test, precision_test, recall_test = entrenar_modelo(X_train_seleccionado, X_test_seleccionado, y_train, y_test, "Metricas en funcion de k basandose en distancias")

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

k = np.arange(1,25,1)
ax.plot(k, accuracy_train, marker='o', linestyle='--', color='r', label="Train")
ax.plot(k, accuracy_test, marker='o', linestyle='-', color='g', label="Test")
ax.set_title("Accuracy en funcion de los k basandose en distancias", fontsize=18)
ax.legend(fontsize=14)
ax.set_xlabel('Numero de vecinos (k)', fontsize=18)
ax.set_ylabel('Accuracy', fontsize=18)
ax.set_xticks(k)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.grid(True)

"""
MEJORA MUCHO LA ACCURACY, AUNQUE REQUIERE UN PREPROCESAMIENTO DE LOS DATOS.
USAMOS UN ARBOL PARA QUE ENCUENTRE LOS PIXELES MAS RELEVANTES, ASI VEMOS COMO QUEDA UN KNN CON LO 3 MEJORES PIXELES.
"""
#%% ENTRENO UN ARBOL PARA VER CUALES CONSIDERA COMO LOS PIXELES MAS IMPORTANTES

arbol = tree.DecisionTreeClassifier(max_depth = 3, random_state = 160)
arbol.fit(X_train, y_train)
# veo la prediccion
y_pred = arbol.predict(X_test)

# calculo las metricas para ver el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# se puede extraer la importancia de cada atributo en el arbol, lo hago y armo una matriz
importancia_pixeles = arbol.feature_importances_
importancia_matriz = importancia_pixeles.reshape(28, 28)
pixeles = np.argsort(importancia_pixeles)[::-1]

print(f"Los 10 pixeles mas relevantes son: {pixeles[:10]}" )

# grafico la importancia de cada pixel en una imagen de 28x28
plt.figure(figsize=(12, 10))
plt.imshow(importancia_matriz, cmap='gray_r', interpolation='nearest', extent=[0, 28, 28, 0])  # extent ajusta la imagen

cbar = plt.colorbar()
cbar.set_label("Importancia", fontsize=18)
cbar.ax.tick_params(labelsize=14)

plt.title("Importancia de los píxeles según ubicación", fontsize=18)
plt.xlabel("Número de columna", fontsize=18)
plt.ylabel("Número de fila", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=13)

plt.xticks(np.arange(0, 29, 1))  
plt.yticks(np.arange(0, 29, 1))
plt.grid(visible=True, color='black', linewidth=0.5, linestyle='--')
plt.show()

#%% USO LOS MEJORES 3 PIXELES SEGUN EL ARBOL

X_train_seleccionado = X_train_dist[["406","400","318"]]
X_test_seleccionado = X_test_dist[["406","400","318"]]
accuracy_train, precision_train, recall_train, accuracy_test, precision_test, recall_test = entrenar_modelo(X_train_seleccionado, X_test_seleccionado, y_train, y_test, "Metricas en funcion de los pixeles mas relevantes")

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

k = np.arange(1,25,1)
ax.plot(k, accuracy_train, marker='o', linestyle='--', color='r', label="Train")
ax.plot(k, accuracy_test, marker='o', linestyle='-', color='g', label="Test")
ax.set_title("Accuracy en funcion de k basandose en los pixeles mas relevantes", fontsize=18)
ax.legend(fontsize=14)
ax.set_xlabel('Numero de vecinos (k)', fontsize=18)
ax.set_ylabel('Accuracy', fontsize=18)
ax.set_xticks(k)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.grid(True)
#%% ACA COMIENZA EL EJERCICIO 3
#%% DIVIDO LOS DATOS EN DEV Y HELD OUT, DEFINO PARAMETROS DE ENTRENAMIENTO

# Cuento y veo el balance de clases
contador = labels.value_counts()
print(contador)
# Las clases estan bastante balanceadas, asi que evaluamos usando accuracy, de todas formas miro tambien recall y precision
X_dev, X_heldout, y_dev, y_heldout = train_test_split(X, labels, test_size=0.2, random_state=160, stratify=labels)

alturas = [1,2,3,4,5,6,7,8,9,10]  # alturas del arbol
nsplits = 3  # numero de folds
# generamos los folds, stratifiedkfold permite dividir de forma balanceada los folds
kf = StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=7)

#%% ENTRENO ARBOLES USANDO GINI

resultados_accuracy_train, resultados_accuracy_test, resultados_precision_train, resultados_precision_test, resultados_recall_train, resultados_recall_test = EntrenarArbol(alturas, kf, "gini", X_dev, y_dev)

# calculo el promedio de accuracy, precision y recall para todos los folds
scores_accuracy_train = resultados_accuracy_train.mean(axis=0)
scores_accuracy_test = resultados_accuracy_test.mean(axis=0)

scores_precision_train = resultados_precision_train.mean(axis=0)
scores_precision_test = resultados_precision_test.mean(axis=0)

scores_recall_train = resultados_recall_train.mean(axis=0)
scores_recall_test = resultados_recall_test.mean(axis=0)

# GRAFICO LOS RESULTADOS DE LAS METRICAS PARA EL CRITERIO GINI
GraficarMetricasArbol(alturas, scores_accuracy_train, scores_accuracy_test, scores_precision_test, scores_precision_train, scores_recall_test, scores_recall_train, "Gini")

#%% ENTRENO ARBOLES USANDO ENTROPY

resultados_accuracy_train, resultados_accuracy_test, resultados_precision_train, resultados_precision_test, resultados_recall_train, resultados_recall_test = EntrenarArbol(alturas, kf, "entropy", X_dev, y_dev)

# calculo el promedio de accuracy, precision y recall para todos los folds
scores_accuracy_train = resultados_accuracy_train.mean(axis=0)
scores_accuracy_test = resultados_accuracy_test.mean(axis=0)

scores_precision_train = resultados_precision_train.mean(axis=0)
scores_precision_test = resultados_precision_test.mean(axis=0)

scores_recall_train = resultados_recall_train.mean(axis=0)
scores_recall_test = resultados_recall_test.mean(axis=0)

GraficarMetricasArbol(alturas, scores_accuracy_train, scores_accuracy_test, scores_precision_test, scores_precision_train, scores_recall_test, scores_recall_train, "entropia")

#%% entreno el modelo elegido en el conjunto dev entero para la mejor profundidad
mejor_profundidad = 6
mejor_criterio = "entropy"
arbol_elegido = tree.DecisionTreeClassifier(max_depth=mejor_profundidad, criterion=mejor_criterio)
arbol_elegido.fit(X_dev, y_dev)

# Predecir en el conjunto de validación (held-out)
y_pred = arbol_elegido.predict(X_heldout)

# Calcular las métricas finales
score_accuracy = accuracy_score(y_heldout, y_pred)
score_precision = precision_score(y_heldout, y_pred, average='macro')
score_recall = recall_score(y_heldout, y_pred, average='macro')

print(f"Accuracy del arbol con depth {mejor_profundidad} en HELD OUT: {score_accuracy}")
print(f"Precision del arbol con depth {mejor_profundidad} en HELD OUT: {score_precision}")
print(f"Recall del arbol con depth {mejor_profundidad} en HELD OUT: {score_recall}")

# Matriz de confusión
matriz_confusion = confusion_matrix(y_heldout, y_pred)
sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()
#%% GRAFICO EL ARBOL 
plt.figure(figsize=(50, 20))  
tree.plot_tree(arbol_elegido, filled=True, 
    feature_names=[f"pixel_{i}" for i in range(X_dev.shape[1])],
    class_names=[str(i) for i in range(10)], fontsize=10, max_depth=3)
plt.title("Primeros 3 niveles del arbol de decision final")
plt.show()
#%% VEO CUALES FUERON LOS PIXELES MAS RELEVANTES
importancia_pixeles = arbol_elegido.feature_importances_
importancia_matriz = importancia_pixeles.reshape(28, 28)
pixeles = np.argsort(importancia_pixeles)[::-1]

print(f"Los 10 pixeles mas relevantes son: {pixeles[:10]}" )
# grafico la importancia de cada pixel en una imagen de 28x28
plt.figure(figsize=(12, 10))
plt.imshow(importancia_matriz, cmap='gray_r', interpolation='nearest', extent=[0, 28, 28, 0])  # extent ajusta la imagen

cbar = plt.colorbar()
cbar.set_label("Importancia", fontsize=18)
cbar.ax.tick_params(labelsize=14)

plt.title("Importancia de los píxeles según ubicación", fontsize=18)
plt.xlabel("Número de columna", fontsize=18)
plt.ylabel("Número de fila", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=13)

plt.xticks(np.arange(0, 29, 1))  
plt.yticks(np.arange(0, 29, 1))
plt.grid(visible=True, color='black', linewidth=0.5, linestyle='--')
plt.show()

#%% 
