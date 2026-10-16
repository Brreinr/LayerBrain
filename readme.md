# LayerBrain

LayerBrain es una librería **experimental y didáctica** para construir redes
neuronales de impulsos (SNN) con Python y NumPy. Permite probar capas,
conexiones dispersas, rutas y señales neuromoduladoras inspiradas en el cerebro.
No es una simulación biológica validada ni un modelo de lenguaje de propósito
general.

## Empieza aquí

Necesitas Python y NumPy. Desde la carpeta del proyecto instala la dependencia
principal y ejecuta el ejemplo:

```powershell
python -m pip install numpy
python main.py
```

El programa entrena una red pequeña con cinco transiciones de texto
(`<INICIO> → H → O → L → A → <FIN>`), muestra algunas métricas e intenta
generar la secuencia aprendida. También crea `network_activity.html`, una
visualización interactiva de la red y su actividad. Para no abrir el navegador
automáticamente:

```powershell
$env:LAYERBRAIN_OPEN_VISUALIZATION="0"
python main.py
```

El resultado exacto puede variar al cambiar la arquitectura, los parámetros o
el código. La demo memoriza una secuencia corta: **no demuestra que la red pueda
generalizar a texto nuevo**.

## Tu primer modelo

El ejemplo de abajo muestra el flujo básico: preparar pares de entrenamiento,
crear el clasificador, entrenarlo y predecir entradas sin etiqueta.

```python
from layerbrain import BrainTrainer, SpikingClassifier, TextDataset

dataset = TextDataset(
    samples=[
        ("<INICIO>", "H"),
        ("H", "O"),
        ("O", "L"),
        ("L", "A"),
        ("A", "<FIN>"),
    ],
    prediction_samples=("<INICIO>", "H", "O", "L", "A"),
    classes=("H", "O", "L", "A", "<FIN>"),
)

model = SpikingClassifier(
    dataset=dataset,
    hidden_sizes=(16,),
    steps_per_sample=30,
)

trainer = BrainTrainer(model.network)
history = trainer.fit_dataset(
    dataset,
    epochs=60,
    steps_per_sample=model.steps_per_sample,
    lineal=False,
)

for prediction in model.predict_dataset():
    print(prediction.label, prediction.spike_counts)

print("Exactitud de la última época:", history["accuracy"][-1])
```

Cada elemento de `samples` es una entrada junto con su etiqueta esperada.
`prediction_samples` contiene entradas para las que quieres obtener una
predicción. `classes` enumera las etiquetas posibles. `TextDataset` codifica
el texto para la red; no tienes que convertirlo manualmente en impulsos.

También puedes ejecutar [`main.py`](./main.py), que es el ejemplo completo del
proyecto e incluye capas especializadas, neuromodulación, rutas adaptativas,
compresión de pesos y generación del HTML.

## Cómo explorar la demo

En [`main.py`](./main.py) puedes probar cambios concretos:

- Cambia los pares de `samples` para enseñar otras transiciones.
- Cambia `prediction_samples` para pedir predicciones de otras entradas.
- Ajusta `epochs`, `steps_per_sample` y `learning_rate` para experimentar con
  el entrenamiento.
- Cambia los tamaños de las capas en `ThalamicConfig`,
  `DentateGranuleConfig` y `PyramidalConfig`.
- Ajusta `synapse_density` para cambiar cuántas conexiones dispersas se crean.
- Revisa la exactitud y la pérdida impresas al final: que el programa termine
  sin errores no significa que haya aprendido bien.

La demo usa `lineal=False`. Con `lineal=True` las conexiones van entre capas
adyacentes en orden; con `lineal=False` se permiten conexiones entre distintos
pares de capas y rutas adaptativas. El modo libre puede usar más memoria. El
modo lineal incluye una decodificación de voltaje subumbral cuando no hay
impulsos de salida: en ese caso puede producir una etiqueta aunque
`spike_counts` sea cero. Esa etiqueta no significa que haya ocurrido un
impulso, y el modo lineal todavía es experimental.

## Qué contiene la librería

- **Capas neuronales:** LIF y variantes inspiradas en sistemas sensoriales,
  talámicos, dentados, piramidales y estriatales.
- **Sinapsis:** conexiones químicas y reservorios dispersos, con dinámica
  temporal y plasticidad experimental.
- **Clasificación y entrenamiento:** `SpikingClassifier` prepara una red y
  `BrainTrainer` la entrena con datasets.
- **Datasets:** texto, audio, visión, sabor, olor, tacto y vectores de impulsos.
- **Visualización:** `visualization_template.html` muestra la topología, las
  conexiones y los datos registrados durante el ejemplo.

Los nombres biológicos son inspiración para experimentar: las ecuaciones y
reglas son simplificaciones de software y no describen por completo el
funcionamiento de un cerebro.

## Datos de otras modalidades

Los datasets sensoriales aceptan pares `(dato, etiqueta)` y entradas opcionales
para predecir. Por ejemplo, para vectores de audio ya convertidos a espectros:

```python
from layerbrain import AudioDataset, SpikingClassifier

audio = AudioDataset(
    samples=[
        (espectro_voz, "voz"),
        (espectro_ruido, "ruido"),
    ],
    prediction_samples=[espectro_nuevo],
    classes=("voz", "ruido"),
)
model = SpikingClassifier(dataset=audio, hidden_sizes=(16,))
```

`VisionDataset`, `TasteDataset`, `SmellDataset` y `TouchDataset` siguen un
patrón similar. Audio espera espectros, a menos que le pases archivos de audio;
visión también puede leer archivos de imagen. Algunas conversiones requieren
dependencias opcionales:

```powershell
python -m pip install Pillow opencv-python pypdf python-docx librosa
```

Instala solo los paquetes que necesites. JPG/JPEG/PNG usan Pillow, los vídeos
MP4 usan OpenCV, PDF usa `pypdf`, DOCX usa `python-docx` y MP3 usa `librosa`.
La fusión de varias modalidades en un único clasificador todavía no está
implementada.

## Capas, dendritas y memoria

Para capas básicas puedes usar `hidden_sizes=(16, 12)`. Para elegir capas
especializadas, importa sus configuraciones desde `layerbrain` y pásalas como
`hidden_layers`:

```python
from layerbrain import PyramidalConfig, SpikingClassifier

model = SpikingClassifier(
    dataset=dataset,
    hidden_layers=(
        PyramidalConfig(32, num_dendrites=6),
    ),
    output_parameters={"num_dendrites": 4},
)
```

`num_dendrites` limita el número total de conexiones que entran a cada neurona
de la capa, sumadas entre todas las capas de origen. Las sinapsis dispersas
guardan datos por conexión existente, no en matrices densas; por eso la
densidad influye en memoria y cómputo. Puedes consultar el consumo aproximado
de los arrays NumPy con `model.network.memory_usage_bytes`. Esto no garantiza
escalabilidad a millones de neuronas: mídela para la arquitectura y el equipo
que uses.

Para comprimir los pesos después del entrenamiento y antes de inferir:

```python
model.quantize_for_inference()
```

La cuantización usa pesos de punto fijo con cuatro decimales y puede cambiar
predicciones cuyos márgenes sean muy pequeños. `model.dequantize_weights()`
restaura los pesos a punto flotante para continuar trabajando con ellos.

## Guardar y cargar un modelo

Después de entrenar, guarda el clasificador con `save`. Más tarde, crea de
nuevo el clasificador con la misma arquitectura, orden de clases y parámetros,
y usa `load` para restaurar el modelo:

```python
model.save("mi_modelo.npz")

# En otra ejecución, vuelve a crear dataset y model con la misma configuración.
model = SpikingClassifier(
    dataset=dataset,
    hidden_sizes=(16,),
    steps_per_sample=30,
)
model.load("mi_modelo.npz")
prediction = model.predict(entrada_nueva)
```

`entrada_nueva` debe tener el formato de entrada de tu dataset (por ejemplo,
un token para `TextDataset` o un vector para `SpikeDataset`).

El archivo comprimido incluye los pesos, las conexiones dispersas exactas, las
puntuaciones de enrutamiento y metadatos de compatibilidad. Al cargar, LayerBrain
comprueba las clases, capas, dimensiones y topología; también restaura si el
entrenamiento guardado usó `lineal=True` o `lineal=False`. Si la configuración
no coincide o el archivo está dañado, se lanza `PersistenceError` en vez de
cargar pesos sobre una arquitectura incompatible.

`save` y `load` guardan el estado aprendido del modelo, no el dataset, el
historial de entrenamiento ni los estados temporales de las neuronas. Conserva
el código que crea la arquitectura y los datos necesarios para volver a usarla.
Los archivos antiguos creados con `network.save_brain(...)` se pueden cargar
cuando la red ya tiene la topología compatible; los nuevos modelos deberían
guardarse y cargarse mediante `SpikingClassifier.save(...)` y
`SpikingClassifier.load(...)`.

## Qué significa el resultado

La predicción incluye `label` y `spike_counts`. Si una salida se obtuvo por
voltaje subumbral en modo lineal, los conteos pueden ser cero. Examina también
las métricas del entrenamiento: accuracy, pérdida, pasos por muestra y
actividad. Una etiqueta devuelta por sí sola no es evidencia de aprendizaje.

GABA y otros neuromoduladores son mecanismos funcionales simplificados para
regular actividad, plasticidad y ruido. No garantizan evitar alucinaciones ni
sustituyen una evaluación rigurosa del modelo.

## Ejecutar las pruebas

Desde la raíz del proyecto:

```powershell
python -m unittest discover -s tests -v
```

Para explorar el coste de una sinapsis dispersa:

```powershell
python benchmarks/benchmark_scaling.py --neurons 10000 --density 0.01 --steps 100 --active-ratio 0.01 --quantized
```

El benchmark mide una sinapsis en el equipo donde se ejecuta; no certifica el
rendimiento de una red completa ni de un modelo grande.

## Estado y contribuciones

LayerBrain es un prototipo abierto a ideas, experimentos y mejoras. Algunas
partes son demostrativas y los resultados dependen de los parámetros; antes de
usarla en un proyecto real, valida el comportamiento con tus propios datos y
compara con métodos de referencia.

Al compartir una propuesta o un error, incluye los pasos para reproducirlo,
la configuración usada y el resultado esperado frente al obtenido. Las ideas
para mejorar aprendizaje, rendimiento, pruebas, ejemplos y documentación son
bienvenidas.
