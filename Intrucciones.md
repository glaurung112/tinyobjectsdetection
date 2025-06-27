## Instrucciones de uso

### 1. Clonar este repositorio

```bash
git clone https://github.com/glaurung112/tinyobjectsdetection.git
cd tinyobjectsdetection/
```

---

### 2. Crear y activar el entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Clonar YOLOv5 (repositorio base)

```bash
git clone https://github.com/ultralytics/yolov5.git
```

---

### 4. Instalar dependencias

> Esto puede tardar unos minutos dependiendo de la conexión y el entorno.

```bash
python3 install_dependencies.py
```

---

## Preparación del dataset

### 5. Generar labels en formato YOLO

```bash
python3 generate_yolo_labels.py
```

---

### 6. Copiar imágenes y labels a una misma carpeta

```bash
python3 copy_to_cropped_base.py
```

---

### 7. Verificar que todas las imágenes tengan su respectivo label

```bash
python3 check_crop.py
```

---

### 8. Generar recortes aleatorios con sus respectivos labels

> Esto mejora la detección de objetos pequeños.

```bash
python3 crop_bulk.py --source dataset/cropped_base --output dataset/cropped_output --crops 10
```

---

### 9. Visualizar muestras de los datos con sus labels

```bash
python3 control_yolo_matplotlib.py
```

---

### 10. Dividir el dataset en entrenamiento y validación (80% entrenamiento y 20% validación)

```bash
python3 split_dataset.py
```

---

## Estructura esperada del dataset

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

### 11. Generar automáticamente el archivo `custom.yaml` con rutas absolutas

```bash
python3 generate_custom.py
```

> Este archivo es utilizado por YOLOv5 para ubicar los datos de entrenamiento y validación.

---

## Entrenamiento

### 12. Entrenar el modelo

```bash
python3 yolov5/train.py --img 640 --batch 16 --epochs 2 --data custom.yaml --weights yolov5s.pt --name small_dots_yolo
```

---

## Evaluación del modelo

### Opción 1: Evaluación por particiones personalizadas

```bash
python3 detect_partition.py --weights yolov5/runs/train/small_dots_yolo/weights/best.pt --source dataset/val --img 640 --conf 0.25
```

### Opción 2: Evaluación directa con el script de YOLOv5

```bash
python3 yolov5/detect.py --weights yolov5/runs/train/small_dots_yolo/weights/best.pt --source dataset/val --img 640 --conf 0.25
```

---

## Aspectos adicionales

* Se debe usar `Python ≥ 3.8` para mejor compatibilidad.
* El entorno virtual `venv` es opcional pero se recomienda mucho su uso.
* Todos los scripts deben ejecutarse desde la raíz del repositorio.

---
