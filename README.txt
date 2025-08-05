En este informe se analiza la clasificación de imágenes del dataset MNIST-C en su versión “Fog”, un conjunto de datos basado en dígitos escritos a mano con ruido agregado.
Se realizaron experimentos para clasificación binaria (dígitos 0 y 1) y multiclase (dígitos del 0 al 9) utilizando modelos de KNN y árboles de decisión. 
Se llevó a cabo un análisis exploratorio para evaluar las características del dataset, seleccionando atributos relevantes y visualizando datos mediante gráficos. 
Para la clasificación binaria, se evaluó el rendimiento de KNN con distintas configuraciones de atributos y valores de k. En la clasificación
multiclase, se entrenaron modelos de árboles de decisión con diferentes profundidades y se aplicó validación cruzada para optimizar el desempeño. 
Finalmente, se compararon los resultados obtenidos y se presentaron conclusiones sobre el rendimiento de los modelos implementados.

REQUERIMIENTOS
Instalar las siguientes bibliotecas:
 - pandas, numpy, matplotlib, seaborn, sklearn


INSTRUCCIONES
 - descargar el conjunto de datos mnistc versión fog en formato csv y colocarlo en la raiz del proyecto, con el siguiente nombre: 'mnist_c_fog_tp.csv'
 - instalar bibliotecas requeridas con el siguiente comando: pip install pandas numpy matplotlib seaborn scikit-learn
 - ejecutar por celdas de código, para ir visualizando las impresiones por consola y los graficos
