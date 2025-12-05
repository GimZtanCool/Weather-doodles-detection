# Trabajo Final - Clasificación de Dibujos (QuickDraw Weather)

Resumen rápido:

- Tema elegido: Doodles meteorológicos (cloud, rainbow, umbrella) del dataset QuickDraw — dibujos reales creados por miles de usuarios, recortados a 500 muestras total (simula el escenario de 500 dibujos hechos en clase).
- Red A: preentrenada en MNIST, luego fine-tune en QuickDraw (transfer learning).
- Red B: entrenada desde cero en QuickDraw.

Archivos añadidos:

- `prepare_dataset.py` - descarga categorías QuickDraw (cloud, rainbow, umbrella), extrae 500 muestras (≈167 por clase) y construye `X.npy` / `y.npy`.
- `weather_doodles_training.ipynb` - pipeline completo (exploración, entrenamiento de Red A/B, métricas y visualizaciones).
- `requirements.txt` - paquetes necesarios.
- `Informe_Trabajo_Final.docx` - informe escrito final (ya no se genera dentro del notebook).

Instrucciones rápidas:

1. Crear y activar entorno (usar Python 3.10, ya instalado en `C:\Users\usuario\AppData\Local\Programs\Python\Python310`):

```pwsh
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Preparar dataset (solo la primera vez):

```pwsh
python prepare_dataset.py
```

3. Abrir `weather_doodles_training.ipynb` (VS Code o Jupyter) y ejecutar las celdas para entrenar/reevaluar las redes. Todas las funciones auxiliares viven dentro del cuaderno.

4. El informe formal se encuentra en `Informe_Trabajo_Final.docx` y no se genera automáticamente desde el notebook.

> Nota: TensorFlow 2.15.1 solo brinda ruedas para Python 3.10/3.11. Si creas el entorno con Python 3.14 verás el error “No matching distribution found for tensorflow”.

Detalles del dataset QuickDraw usado:

- Categorías seleccionadas: `cloud`, `rainbow`, `umbrella` (tema “fenómenos meteorológicos / íconos del clima”).
- Cada archivo trae hasta 75k dibujos 28x28 en formato numpy bitmap; el script toma solo 500 muestras en total (166–167 por clase) para respetar el requisito del curso.
- El resultado (`X.npy`, `y.npy`) tiene 24k imágenes listas para entrenar las dos redes.

Siguientes pasos recomendados:

- Ajustar número de épocas y batch size para mejor rendimiento.
- Exportar un cuaderno `.ipynb` con visualizaciones y el informe final.
- Generar el informe con la estructura: Problemática, Justificación, Desarrollo, Conclusiones.
