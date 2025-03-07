#%% DATOS CARATULA
"""
Laboratorio de datos - Verano 2025
Trabajo Práctico N° 2

Integrantes:
- Sebastian Ceffalotti - sebastian.ceffalotti@gmail.com
- Aaron Cuellar - aaroncuellar2003@gmail.com
- Rodrigo Coppa - rodrigo.coppa98@gmail.com

Descripción:


Detalles técnicos:
abcesllueve1
"""
# %% IMPORTACION DE LIBRERIAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

#%% FUNCION PARA GRAFICAR 10 imagenes de un digito, semilla es para que sea al azar
def graficarMuestraDigitos(digito, semilla):
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
    
#%% CACLULA LA IMAGEN PROMEDIO DE UN DIGITO
def img_promedio_digito(datos, digito):
    datos_digito = datos[datos["labels"] == digito].drop(columns = "labels")
    img_promedio = np.sum(datos_digito, axis = 0)/len(datos_digito)
    return img_promedio

#%% OBTENGO LA POSICION DE LA COLUMNA DE UN PIXEL DE ACUERDO A SUS COORDENADAS
def obtenerPosColumna(posicion):
    fila, columna = posicion[0], posicion[1]
    return 28*(fila-1) + columna - 1 # resto porque arranca en 0 (? chequear esto
#%% ENTRENO UN KNN CON LOS PIXELES SELECCIONADOS
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
# %% LECTURA DE ARCHIVOS

#rutas
_ruta_actual = os.getcwd()
_ruta_mnistc = os.path.join(_ruta_actual, 'mnist_c_fog_tp.csv')

# lectura mnistc, con el index_col podes decirle que columna usar de indice :)
mnistc = pd.read_csv(_ruta_mnistc, index_col = 0)
labels = mnistc["labels"]
# Guardo los pixeles en X 
X = mnistc.drop(columns = ["labels"]) 

#%% ACA COMIENZA EL EJERCICIO 1
#%% EJEMPLO PARA GRAFICAR UNA IMAGEN

img = np.array(X.iloc[0]).reshape((28,28))
plt.imshow(img, cmap='gray') 
plt.title(f'Dígito: {labels.iloc[0]}')
plt.show()

#%% GRAFICO 10 MUESTRAS ALEATORIAS DE CADA DIGITO
plt.figure(figsize=(10, 10))
for digito in range(0,10):
    graficarMuestraDigitos(digito,1)
    
#%% GRAFICO LA IMAGEN PROMEDIO DE TODOS LOS DIGITOS
plt.figure(figsize=(12, 6))
for digito in range(0,10):
    img_prom = img_promedio_digito(mnistc, digito)
    img = np.array(img_prom).reshape((28,28))
    
    plt.subplot(2, 5, digito + 1)
    plt.imshow(img, cmap='inferno')
    plt.title(f"Promedio del {digito}")

#%% ACA COMIENZA EL EJERCICIO 2
#%% DE los datos extraigo los de 0 y 1, veo el balance y separo en train y test
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

#%% GRAFICO LOS PROMEDIOS DEL 0 y el 1 y la resta, por inspeccion decido que pixeles usar

plt.figure(figsize=(12, 6))
img_prom_0 = img_promedio_digito(datos, 0)
plt.subplot(1, 3, 1)
plt.imshow(np.array(img_prom_0).reshape((28, 28)), cmap='gray')
plt.title("Promedio del 0")

img_prom_1 = img_promedio_digito(datos, 1)
plt.subplot(1, 3, 2)  
plt.imshow(np.array(img_prom_1).reshape((28, 28)), cmap='gray')
plt.title("Promedio del 1")

plt.subplot(1, 3, 3)  
resta = np.abs(img_prom_1-img_prom_0)
plt.imshow(np.array(resta).reshape((28, 28)), cmap='gray')
plt.title("Resta")

plt.tight_layout()
plt.show()

"""
Viendo las imagenes elijo pixeles de manera arbitraria, 
elijo el del centro, uno a la izquierda y otro a la derecha por ejemplo
"""      
    
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

#%% Ejercicio 3

# Definimos variables predictoras y la etiqueta
y = mnistc["labels"]

# Separamos en desarrollo (80%) y held-out (20%)
X_dev, X_held, y_dev, y_held = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# %% Entrenamos el arbol de decision
depth_range = range(1, 11)  # Profundidades de 1 a 10
results = {}

for depth in depth_range:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_dev, y_dev)  # Entrenar en conjunto de desarrollo
    y_pred = clf.predict(X_dev)  # Predecir en el mismo conjunto
    acc = accuracy_score(y_dev, y_pred)
    results[depth] = acc
    print(f"Profundidad {depth} - Exactitud: {acc:.4f}")

# Graficamos los resultados
plt.figure(figsize=(8, 5))
plt.plot(depth_range, list(results.values()), marker='o', linestyle='-', color='b')
plt.xlabel("Profundidad del Árbol")
plt.ylabel("Exactitud")
plt.title("Exactitud en función de la profundidad del árbol")
plt.grid(True)
plt.show()

# %% Validacion cruzada con k = 5

best_score = 0
best_depth = None

for depth in depth_range:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(clf, X_dev, y_dev, cv=5, scoring='accuracy')  # K-Folding con k=5
    mean_score = scores.mean()
    
    print(f"Profundidad {depth} - Media de Exactitud: {mean_score:.4f}")
    
    if mean_score > best_score:
        best_score = mean_score
        best_depth = depth

print(f"\nMejor profundidad: {best_depth} con exactitud promedio de {best_score:.4f}")

# %% Matriz de Confusion

# Entrenamos el mejor modelo en el conjunto completo de desarrollo
best_clf = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
best_clf.fit(X_dev, y_dev)

# Evaluamos en el conjunto held-out
y_held_pred = best_clf.predict(X_held)
held_accuracy = accuracy_score(y_held, y_held_pred)

print(f"\nExactitud en conjunto held-out: {held_accuracy:.4f}")

# Matriz de confusión
cm = confusion_matrix(y_held, y_held_pred)

# Visualizamos la matriz de confusión
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicción")
plt.ylabel("Realidad")
plt.title("Matriz de Confusión en Held-Out")
plt.show()
