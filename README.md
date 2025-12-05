# Weather Doodles Detection

Este repositorio contiene únicamente los artefactos necesarios para reproducir el trabajo final sin descargar nuevamente el dataset de QuickDraw.

## Contenido

- `X.npy` / `y.npy`: 500 bocetos (nube, arcoíris, paraguas) ya balanceados y normalizados.
- `weather_doodles_training.ipynb`: cuaderno con exploración, entrenamiento y generación del informe.
- `demo_clasificador.py`: utilidades para cargar el dataset local.
- `train_models.py`: entrena Red A (transfer learning) y Red B (desde cero).
- `redA_finetuned.h5` / `redB_fromscratch.h5`: pesos resultantes.
- `Informe_Trabajo_Final.md`: reporte automatizado.
- `requirements.txt`: dependencias.

## Uso rápido

```pwsh
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_models.py
```

También puedes abrir `weather_doodles_training.ipynb` en VS Code o Jupyter para ejecutar todo el pipeline y regenerar el informe.
