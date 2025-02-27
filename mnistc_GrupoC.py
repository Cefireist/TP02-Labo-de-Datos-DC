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
import seaborn as sns

# %% LECTURA DE ARCHIVOS
mnist = pd.read_csv("mnist_c_fog_tp.csv", index_col = 0)

mnist.head()

#%%
y = mnist["labels"] #columna con los digitos
X = mnist.drop(columns = ["labels"]) 

#Plot imagen 
img = np.array(X.iloc[0]).reshape((28,28))
plt.imshow(img, cmap='gray') 
plt.title(f'Dígito: {y.iloc[0]}')
plt.show()

array = X.loc[0].values.reshape(28,28)
print(array)

#%%
contador = mnist["labels"].value_counts()

digito_0 = X[y == 0]
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
sample_images = digito_0.sample(10, random_state=42).values.reshape(10, 28, 28)

for i, ax in enumerate(axes.flat):
    ax.imshow(sample_images[i], cmap='gray')
    ax.axis('off')

plt.suptitle("Ejemplos de imágenes del dígito 5")
plt.show()

digito_1 = X[y == 1]
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
sample_images = digito_1.sample(10, random_state=42).values.reshape(10, 28, 28)

for i, ax in enumerate(axes.flat):
    ax.imshow(sample_images[i], cmap='gray')
    ax.axis('off')

plt.suptitle("Ejemplos de imágenes del dígito 5")
plt.show()


# %%
image_data = X.iloc[2]
print(image_data)

image_matrix = image_data.values.reshape(28,28)

fourier_transform = np.fft.fft2(image_matrix)

fourier_shifted = np.fft.fftshift(fourier_transform)

magnitude = np.abs(fourier_shifted)

fig, ax = plt.subplots(1, 2, figsize=(12,6))

ax[0].imshow(image_matrix, cmap ="gray")
ax[0].set_title("Imagen Original")

ax[1].imshow(np.log(magnitude + 1), cmap ='gray')
ax[1].set_title("Magnitud de la Transformada de Fourier")

#%%

# Crear una máscara para mantener solo las bajas frecuencias
# Definir el tamaño de la ventana de las frecuencias bajas (por ejemplo, un 30% del tamaño total)
rows, cols = image_matrix.shape
center_row, center_col = rows // 2, cols // 2
radius = int(min(rows, cols) * 0.2) # Mantener solo las frecuencias dentro de un radio del centro

# Crear un filtro de baja frecuencia (mascara)
mask = np.zeros((rows, cols), dtype=np.uint8)

# Poner en 1 los valores dentro del radio
for i in range(rows):
    for j in range(cols):
        if np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2) < radius:
            mask[i, j] = 1

# Aplicar la máscara a las frecuencias
fourier_filtered = fourier_shifted * mask

# Deshacer el desplazamiento para obtener las frecuencias originales
fourier_filtered_shifted_back = np.fft.ifftshift(fourier_filtered)

# Reconstruir la imagen con las frecuencias bajas
image_reconstructed = np.fft.ifft2(fourier_filtered_shifted_back).real

# Mostrar la imagen original y la reconstruida con bajas frecuencias
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Imagen original
ax[0].imshow(image_matrix, cmap='gray')
ax[0].set_title("Imagen Original")

# Imagen reconstruida con frecuencias bajas
ax[1].imshow(image_reconstructed, cmap='gray')
ax[1].set_title("Reconstrucción con Frecuencias Bajas")

plt.show()


#%% CARGA DE DATOS


#%% FUNCIONES A UTILIZAR


#%% CODIGO PRINCIPAL
