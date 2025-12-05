# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os


def cargar_dataset_local(ruta_x='X.npy', ruta_y='y.npy', tamano=(28,28)):
    if not os.path.exists(ruta_x) or not os.path.exists(ruta_y):
        raise FileNotFoundError('No se encontraron los archivos del dataset. Ejecuta `prepare_dataset.py`.')
    imagenes_crudas = np.load(ruta_x)
    imagenes_crudas = imagenes_crudas / 255.0
    etiquetas = np.load(ruta_y)
    imagenes_redimensionadas = []
    for imagen in imagenes_crudas:
        imagenes_redimensionadas.append(resize(imagen, tamano))
    imagenes = np.array(imagenes_redimensionadas)
    if imagenes.ndim == 3:
        imagenes = imagenes[..., None]
    return imagenes, etiquetas


def evaluar_modelo(modelo, imagenes_prueba, etiquetas_prueba, mapa_etiquetas=None):
    probabilidades = modelo.predict(imagenes_prueba)
    predicciones = np.argmax(probabilidades, axis=1)
    exactitud = accuracy_score(etiquetas_prueba, predicciones)
    matriz_confusion = confusion_matrix(etiquetas_prueba, predicciones)
    reporte = classification_report(etiquetas_prueba, predicciones)
    return {'exactitud': exactitud, 'matriz_confusion': matriz_confusion, 'reporte': reporte}


if __name__ == '__main__':
    print('Archivo de apoyo para ejecución local. Usa `prepare_dataset.py` y `train_models.py`.')