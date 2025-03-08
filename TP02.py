"""
Laboratorio de datos - Verano 2025
Trabajo Práctico N° 2 

Integrantes:
- Sebastian Ceffalotti - sebastian.ceffalotti@gmail.com
- Aaron Cuellar - aaroncuellar2003@gmail.com
- Rodrigo Coppa - rodrigo.coppa98@gmail.com

Descripción:
En este script realizamos los graficos necesarios para el realizar
el analisis exploratorio de la fuente de datos MNIST-C

Detalles técnicos:
- Lenguaje: Python
- Librerías utilizadas: numpy, matplotlib, duckdb, pandas, seaborn y scikit-learn
"""

#%% IMPORTO LAS LIBRERIAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% LECTURA DE ARCHIVOS
mnist = pd.read_csv("mnist_c_fog_tp.csv", index_col = 0)

mnist.head()
#%%
y = mnist["labels"]  # Etiqueta (dígito)
X = mnist.drop(columns = ["labels"])  # Datos de imagen (píxeles)

#Plot imagen 
img = np.array(X.iloc[12]).reshape((28,28))
plt.imshow(img, cmap='gray') 
plt.title(f'Dígito: {y.iloc[12]}')
plt.show() 

# %% Media y varianza de los pixeles
pixel_media = X.mean(axis=0).values.reshape((28,28))

# %% Mapa de calor de la media de los pixeles
plt.figure(figsize=(8, 6))
sns.heatmap(pixel_media, cmap="inferno", xticklabels=False, yticklabels=False)
plt.title("Mapa de calor de la media de los píxeles")
plt.show()

# %% Imagen promedio de cada digito
digito_media = X.groupby(y).mean()

# %% Digitos para comparar (1, 3 y 8)
digito_1_media = digito_media.loc[1].values.reshape((28,28))
digito_3_media = digito_media.loc[3].values.reshape((28, 28))
digito_8_media = digito_media.loc[8].values.reshape((28, 28))

#%% Mapa de calor de la imagen promedio del dígito 1
plt.figure(figsize=(8, 6))
sns.heatmap(digito_1_media, cmap="inferno", xticklabels=False, yticklabels=False)
plt.title("Imagen promedio del dígito 1")
plt.show()

#%% Mapa de calor de la imagen promedio del dígito 3
plt.figure(figsize=(8, 6))
sns.heatmap(digito_3_media, cmap="inferno", xticklabels=False, yticklabels=False)
plt.title("Imagen promedio del dígito 3")
plt.show()

# %% Mapa de calor de la imagen promedio del dígito 8
plt.figure(figsize=(8, 6))
sns.heatmap(digito_8_media, cmap="inferno", xticklabels=False, yticklabels=False)
plt.title("Imagen promedio del dígito 8")
plt.show()

# %% Calculamos la diferencia entre la imagen promedio del dígito 1 y la del dígito 3
dif_1_3 = digito_1_media - digito_3_media

#%% Mapa de calor de la diferencia entre los dos dígitos 1 y 3
plt.figure(figsize=(8, 6))
sns.heatmap(dif_1_3, cmap="coolwarm", center=0, xticklabels=False, yticklabels=False)
plt.title("Diferencia entre el dígito 1 y el 3")
plt.show()

# %% Diferencia entre la imagen promedio del dígito 3 y la del dígito 8
dif_3_8 = digito_3_media - digito_8_media
print(dif_3_8)

#%% Mapa de calor de la diferencia entre los dos dígitos 3 y 8
plt.figure(figsize=(8, 6))
sns.heatmap(dif_3_8, cmap="coolwarm", center=0, xticklabels=False, yticklabels=False)
plt.title("Diferencia entre el dígito 3 y el 8")
plt.show()

#%% Filtrar solo las imágenes del dígito 0
digito_0 = X[y == 0]

# %% Imagen promedio del dígito 0
digito_0_media = digito_0.mean(axis=0).values.reshape((28, 28))

# %% Mapa de calor de la imagen promedio del dígito 0
plt.figure(figsize=(8, 6))
sns.heatmap(digito_0_media, cmap="inferno", xticklabels=False, yticklabels=False)
plt.title("Imagen promedio del dígito 0")
plt.show()

# %% Mostramos ejemplos aleatorios del dígito 0
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
sample_images = digito_0.sample(10, random_state=42).values.reshape(10, 28, 28)

for i, ax in enumerate(axes.flat):
    ax.imshow(sample_images[i], cmap='gray')
    ax.axis('off')

plt.suptitle("Ejemplos de imágenes del dígito 0")
plt.show()
