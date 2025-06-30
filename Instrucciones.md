## Instrucciones de uso

### 1. Clonar este repositorio

```bash
git clone https://github.com/glaurung112/tinyobjectsdetection.git
cd tinyobjectsdetection/
```

---

### 2. Crear y activar un entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

En Windows, utilice:

```bash
venv\Scripts\activate
```

> Una vez activado el entorno, el prompt de la terminal se verá de esta forma:

```bash
(venv) user@maquina:~/ruta/proyecto/.../tinyobjectsdetection$
```

> Esto permite que todas las instalaciones y ejecuciones se realicen dentro del entorno virtual y no afecten el resto del sistema.

---

### 3. Clonar una versión modificada de YOLOv5 (HIC-YOLOv5)

```bash
git clone https://github.com/aash1999/yolov5-cbam.git
```

> Esta es una versión de YOLOv5 optimizada para la detección de objetos pequeños. Las mejoras están basadas en el artículo *HIC-YOLOv5: Improved YOLOv5 For Small Object Detection*.

---

### 4. Instalar dependencias

> Esto puede tardar unos minutos dependiendo de la conexión y el entorno.

```bash
python3 install_dependencies.py
```

> Si va a utilizar GPU y necesita instalar `torch` con CUDA, **no use `pip install torch`**. Vaya a:

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

> Y siga las instrucciones, por ejemplo:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Preparación del dataset

### 5. Obtener el dataset desde KaggleHub

```bash
python3 generate_dataset.py
```

> Esto debe generar una nueva carpeta `dataset/` con las imágenes de la base de datos a utilizar.

> Por defecto se utiliza el dataset de KaggleHub: `daenys2000/small-object-dataset` con imágenes de puntos blancos en un fondo negro, se descargará un máximo de 200 imágenes y se permite un mínimo de un 1 punto y un máximo de 60 puntos por imagen.

> Si desea ajustar estos parámetros puede utilizar, por ejemplo:

```bash
python3 generate_dataset.py --source <dataset_de_KaggleHub> --max_img 100 --min_points 5 --max_points 40
```

> Esto utilizará otro dataset de KaggleHub (<dataset_de_KaggleHub>), descargará un máximo de 100 imágenes y permitirá un mínimo de 5 puntos y un máximo de 40 puntos por imagen.

---

### 6. Generar labels en formato YOLO

```bash
python3 generate_yolo_labels.py
```

> Esto debe generar una nueva carpeta `dataset/labels` con archivos `.txt` con los mismos nombres de las imágenes.

---

### 7. Copiar imágenes y labels a una misma carpeta

```bash
python3 copy_to_cropped_base.py
```

 Esto debe generar una nueva carpeta `dataset/cropped_base` con archivos `.png` y `.txt` con los mismos nombres.

---

### 8. Verificar que todas las imágenes tengan su respectivo label

```bash
python3 check_crop.py
```

---

### 9. Generar recortes aleatorios con sus respectivos labels

> Esto genera recortes aleatorios sobre los datos para **aumentar la diversidad del conjunto de datos**, lo cual mejora la detección de objetos pequeños.


```bash
python3 crop_bulk.py --source dataset/cropped_base --output dataset/cropped_output --crops 10
```

> Ahora la carpeta `dataset/cropped_output` debería tener 10 veces más imágenes (con sus respectivos labels) que el dataset original. Puede ajustar el parámetro `--crops 10` si desea realizar menos recortes.

---

### 10. Visualizar muestras de los datos con sus labels

```bash
python3 control_yolo_matplotlib.py
```

> Se desplegará un pequeño menú de opciones de esta forma:

```bash
[Imagen <number>]: <name>
[n]ext, [b]ack, [r]andom, [q]uit:
```

> Esto le permitirá revisar las imágenes con sus respectivos labels. 

---

### 11. Dividir el dataset en entrenamiento y validación

```bash
python3 split_dataset.py
```

> Por defecto se divide en 80% entrenamiento y 20% validación (`ratio = 0.8`). Para dividir en otra proporción puede utilizar, por ejemplo:

```bash
python3 split_dataset.py --ratio 0.7
```

> Esto dividiría el dataset en 70% entrenamiento y 30% validación.

---

## Estructura esperada del dataset

> Hasta el momento, la carpeta `dataset/` debería verse de la siguiente forma:

```
dataset/
├── cropped_base/
├── cropped_output/
├── images/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── labels/
│   ├── image001.txt
│   ├── image002.txt
│   └── ...
├── train/
│   ├── image015.jpg
│   ├── image015.txt
│   └── ...
├── val/
│   ├── image042.jpg
│   ├── image042.txt
│   └── ...
```

---

### 12. Generar automáticamente el archivo `custom.yaml` con rutas absolutas

```bash
python3 generate_custom.py
```

> Este archivo es utilizado por YOLOv5 para ubicar los datos de entrenamiento y validación.

---

## Entrenamiento

### 13. Entrenar utilizando el modelo mejorado

```bash
python3 yolov5-cbam/train.py --img 320 --batch 16 --epochs 150 --data custom.yaml --cfg yolov5-cbam/models/yolo5m-cbam-involution.yaml --weights yolov5s.pt --name small_dots_yolo
```

> Esto puede tardar varias horas (o incluso días) dependiendo de la capacidad computacional y el entorno.

> Dependiendo de su capacidad computacional, puede ajustar los parámetros `--img`, `--batch`, `--epochs`, considerando lo siguiente:

#### `--img X`: dimensiones de las imágenes.

* Las imágenes se redimensionarán a **X×X píxeles** antes de entrar al modelo.
* Imágenes más grandes pueden capturar más detalle, pero consumen más memoria y tiempo.
* Valores comunes: `320`, `480`, `640`, `1280`.
* Usar `320` puede ser mejor para objetos pequeños.

---

#### `--batch X`: imágenes que se procesan al mismo tiempo durante cada paso de entrenamiento.

* El modelo procesa **X** imágenes antes de hacer una actualización de pesos.
* Valores mayores usan más memoria GPU, pero pueden estabilizar y acelerar el entrenamiento.
* Si cuenta con **GPU con poca RAM**, es recomendable usar `8` o `16`.
* Si cuenta con **12GB+**, se pueden usar `32` o más.

---

#### `--epochs X`: veces que se verá todo el conjunto de entrenamiento completo (ciclos).

* El modelo verá cada imagen del dataset unas **X** veces.
* Más epochs permiten mejor ajuste, pero pueden causar **overfitting** si el dataset es pequeño.
* Valores comunes: `50`–`300` (valores grandes pueden aumentar mucho el tiempo de entrenamiento).

> Aquí se recomienda utilizar:

* `--img 480`: porque se está utilizando **input partitioning** y esto evita pérdida de información fina.
* `--batch 16`: está bien para una GPU de gama media.
* `--epochs 150`: es un buen punto de partida (se puede ajustar si se ve que el modelo aún no converge).

---

## Evaluación del modelo

### Opción 1: Evaluación directa con el script de YOLOv5

```bash
python3 yolov5-cbam/detect.py --weights yolov5-cbam/runs/train/small_dots_yolo/weights/best.pt --source dataset/val --img 320 --conf-thres 0.1
```

### Opción 2: Evaluación personalizada

```bash
python3 detect_metrics.py --weights yolov5-cbam/runs/train/small_dots_yolo/weights/best.pt --source dataset/val --img 320 --conf-thres 0.1
```

> En este caso, se deben generar dos carpetas:

* `detecciones`: que contiene las imágenes con las detecciones realizadas por el modelo.
* `resultados`: que contiene las métricas obtenidas y la distribución de *False Negatives* y *True Positives*.

> Esta es una versión modificada de `detect.py` para ver las métricas obtenidas (Precision, Recall, F1 Score).

> En ambos casos, el parámetro `--conf` establece el **umbral de confianza mínima** que debe tener una predicción para ser considerada válida y mostrarse en la salida.

#### `--conf X`: Solo mostrar detecciones con una probabilidad mayor o igual a **X**.

* Valores bajos (`--conf 0.1`): detectas más objetos, pero con más falsos positivos.
* Valores altos (`--conf 0.7`): detectas solo los objetos más seguros, pero se pueden perder objetos pequeños.

> Aquí se recomienda usar `--conf 0.1` o incluso `--conf 0.05` ya que los puntos pequeños suelen tener **menos confianza** y se podrían perder muchos si el umbral es alto.

> En ambos casos debe revisar la carpeta a utilizar (si ha realizado varios entrenamientos): `small_dots_yolo`, `small_dots_yolo1`, `small_dots_yolo2`, ...

---

## Aspectos adicionales

* Se debe usar `Python ≥ 3.8` para mejor compatibilidad.
* El entorno virtual `venv` es opcional pero se recomienda mucho su uso.
* Es preferible utilizar archivos `.png` para mejor compatibilidad.
* Todos los scripts deben ejecutarse desde la raíz del repositorio.

---
