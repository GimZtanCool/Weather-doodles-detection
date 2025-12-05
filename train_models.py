"""Entrena dos redes sobre los bocetos meteorológicos de QuickDraw.

Red A: se preentrena en MNIST y luego se ajusta al dataset propio (transfer learning).
Red B: se entrena desde cero únicamente con los bocetos disponibles.

Salidas: `redA_finetuned.h5`, `redB_fromscratch.h5` y métricas impresas en consola.

Uso:
    python train_models.py
"""
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from demo_clasificador import cargar_dataset_local


def crear_modelo(num_clases, forma_entrada=(28,28,1)):
    modelo = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=forma_entrada),
        MaxPool2D(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPool2D(),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPool2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_clases, activation='softmax')
    ])
    return modelo


def preprocesar_quickdraw(imagenes, etiquetas, proporcion_prueba=0.2):
    imagenes = imagenes.astype('float32')
    if imagenes.max() > 1.0:
        imagenes = imagenes / 255.0
    if imagenes.ndim == 3:
        imagenes = imagenes[..., None]
    imgs_entrenamiento, imgs_prueba, etq_entrenamiento, etq_prueba = train_test_split(
        imagenes,
        etiquetas,
        test_size=proporcion_prueba,
        random_state=42,
        stratify=etiquetas
    )
    return imgs_entrenamiento, imgs_prueba, etq_entrenamiento, etq_prueba


def main():
    if not (os.path.exists('X.npy') and os.path.exists('y.npy')):
        raise FileNotFoundError('Ejecuta primero prepare_dataset.py para crear X.npy y y.npy')

    imagenes, etiquetas = cargar_dataset_local('X.npy', 'y.npy')
    numero_clases = len(np.unique(etiquetas))
    print('Dataset QuickDraw cargado:', imagenes.shape, 'etiquetas', etiquetas.shape, 'clases', numero_clases)
    imgs_entrenamiento, imgs_prueba, etq_entrenamiento, etq_prueba = preprocesar_quickdraw(imagenes, etiquetas)

    # --- Red A: preentrenamiento en MNIST + ajuste fino ---
    (mn_x, mn_y), _ = mnist.load_data()
    mn_x = mn_x.astype('float32') / 255.0
    mn_x = mn_x[..., None]
    mn_y = mn_y.astype('int')

    modelo_base = crear_modelo(num_clases=10, forma_entrada=(28,28,1))
    modelo_base.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print('\nPreentrenando modelo base en MNIST...')
    modelo_base.fit(mn_x, mn_y, batch_size=128, epochs=3, validation_split=0.1)
    modelo_base.save('mnist_pretrained.h5')

    red_a = crear_modelo(num_clases=numero_clases, forma_entrada=(28,28,1))
    for indice, capa in enumerate(red_a.layers):
        try:
            capa.set_weights(modelo_base.layers[indice].get_weights())
        except Exception:
            pass

    for capa in red_a.layers:
        capa.trainable = isinstance(capa, Dense)

    red_a.compile(optimizer=Adam(5e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print('\nAjuste inicial de capas densas (Red A)...')
    red_a.fit(
        imgs_entrenamiento,
        etq_entrenamiento,
        batch_size=64,
        epochs=5,
        validation_data=(imgs_prueba, etq_prueba),
        callbacks=[EarlyStopping(patience=2, restore_best_weights=True)]
    )

    for capa in red_a.layers:
        capa.trainable = True

    red_a.compile(optimizer=Adam(3e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print('\nAjuste fino global Red A (todas las capas entrenables)...')
    red_a.fit(
        imgs_entrenamiento,
        etq_entrenamiento,
        batch_size=64,
        epochs=25,
        validation_data=(imgs_prueba, etq_prueba),
        callbacks=[
            ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
            EarlyStopping(patience=6, restore_best_weights=True)
        ]
    )
    red_a.save('redA_finetuned.h5')

    # --- Red B: entrenamiento desde cero ---
    red_b = crear_modelo(num_clases=numero_clases, forma_entrada=(28,28,1))
    red_b.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print('\nEntrenando Red B desde cero...')
    red_b.fit(
        imgs_entrenamiento,
        etq_entrenamiento,
        batch_size=64,
        epochs=15,
        validation_data=(imgs_prueba, etq_prueba),
        callbacks=[
            ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
            EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )
    red_b.save('redB_fromscratch.h5')

    # --- Evaluación ---
    print('\nEvaluando Red A...')
    predicciones_a = np.argmax(red_a.predict(imgs_prueba), axis=1)
    print('Exactitud Red A:', np.mean(predicciones_a == etq_prueba))
    print('Reporte de clasificación Red A:\n', classification_report(etq_prueba, predicciones_a))
    print('Matriz de confusión Red A:\n', confusion_matrix(etq_prueba, predicciones_a))

    print('\nEvaluando Red B...')
    predicciones_b = np.argmax(red_b.predict(imgs_prueba), axis=1)
    print('Exactitud Red B:', np.mean(predicciones_b == etq_prueba))
    print('Reporte de clasificación Red B:\n', classification_report(etq_prueba, predicciones_b))
    print('Matriz de confusión Red B:\n', confusion_matrix(etq_prueba, predicciones_b))


if __name__ == '__main__':
    main()
