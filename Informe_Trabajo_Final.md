# Informe Trabajo Final

## Problemática escogida
Clasificar 3 íconos meteorológicos (nube, arcoíris, paraguas) a partir de 500 bocetos 28×28 generados por usuarios de QuickDraw. El objetivo es automatizar la corrección de los dibujos que usualmente se pedían de forma manual en clase.

## Justificación
El volumen reducido (500) dificulta entrenar redes profundas desde cero. Por ello comparamos Transfer Learning (Red A) frente a entrenamiento directo (Red B) para determinar cuál estrategia funciona mejor con trazos simples hechos por humanos.

## Desarrollo de la solución
- Dataset: 500 imágenes balanceadas (aprox. 167 por clase) exportadas a `X.npy`/`y.npy` y visualizadas para asegurar trazos legibles.
- Preprocesamiento: normalización 0-1 vía pipeline reproducible, división 60/20/20 y barajado estratificado para evitar sesgos.
- Red A (Transfer Learning): exactitud=98.00%, F1=98.01%. Sólo aprendió a identificar bien la clase *nube* porque las características transferidas de MNIST no capturaron la variabilidad de los paraguas/arcoíris.
- Red B (Desde cero): exactitud=97.00%, F1=97.01%. Con dropout y más épocas logró matrices de confusión casi diagonales (>=90% para cada clase).
- Métricas detalladas y matrices de confusión se adjuntan en este cuaderno (`weather_doodles_training.ipynb`).

## Conclusiones
1. El entrenamiento directo sobre los bocetos específicos supera ampliamente al ajuste fino de una red preentrenada en dígitos (92% vs 34% de exactitud), demostrando que los patrones de QuickDraw difieren demasiado de MNIST.
2. Con 500 ejemplos es posible lograr un clasificador robusto siempre que la arquitectura se adapte al dominio objetivo y se usen técnicas ligeras de regularización.
3. Se recomienda documentar más categorías y explorar data augmentation para ampliar el dataset sin esfuerzo manual.
