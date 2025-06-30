# Small Particle Detection and Counting Project

## Overview

This repository is a fork of the *Small Particle Detection and Counting* project, led by Professor Marvin Coto Jiménez. The goal is to develop an automated system for detecting and counting small particles in images using machine learning techniques.

## Project Description

The objective of this project is to build a system capable of accurately detecting and counting small particles in images. It integrates machine learning algorithms with image processing techniques to improve detection performance.

Key project tasks include:

1. **Literature Review** — Investigate current methods for detecting small objects in images.
2. **Dataset Analysis** — Explore and prepare the dataset for model training.
3. **Model Implementation** — Train an object detection model (e.g., CNN, YOLO, Faster R-CNN).
4. **Model Evaluation** — Assess the model using metrics like precision, recall, and F1-score.
5. **Thresholding and Detection** — Tune detection thresholds based on particle size, count, and contrast.

## Project Approach

This project combines two approaches to enhance small object detection:


* Enhancements from the [Small-Object-Detection-with-YOLO](https://github.com/hamzagorgulu/Small-Object-Detection-with-YOLO) repository, which introduces improvements for small object detection using *input partitioning*, and provides useful scripts for dataset inspection and evaluation of results.

* Modifications of the official [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5/tree/c442a2e99321ebd72b242bc961824f82d46e4fd3) implementation from [HIC-YOLOv5](https://github.com/aash1999/yolov5-cbam), which incorporates attention mechanisms such as CBAM and involution layers to improve small object detection performance. These enhancements are based on the paper *HIC-YOLOv5: Improved YOLOv5 for Small Object Detection*.

---

## Scripts Description

This project includes a collection of scripts for managing the dataset, preparing labels, running experiments, and evaluating results.

* **`install_dependencies.py`**
  Installs all necessary dependencies for running YOLOv5 and the additional scripts used in this project.

* **`generate_dataset.py`**
  Downloads a dataset from Kaggle. You can specify the Kaggle dataset path, the maximum number of images to include, and constraints on the number of points per image (minimum and maximum).

* **`generate_yolo_labels.py`**
  *Modified rom [Small-Object-Detection-with-YOLO](https://github.com/hamzagorgulu/Small-Object-Detection-with-YOLO).*
  Generates YOLO-format labels from annotation files.

* **`copy_to_cropped_base.py`**
  Copies images and their corresponding labels into the same folder to create a consistent dataset structure.

* **`check_crop.py`**
  Verifies that every image has a corresponding YOLO label file.

* **`crop_bulk.py`**
  *Modified rom [Small-Object-Detection-with-YOLO](https://github.com/hamzagorgulu/Small-Object-Detection-with-YOLO).*
  Crops images into smaller patches to increase dataset diversity and enhance the model’s ability to detect small objects.

* **`control_yolo_matplotlib.py`**
  *Modified rom [Small-Object-Detection-with-YOLO](https://github.com/hamzagorgulu/Small-Object-Detection-with-YOLO).*
  Visualizes random samples from the dataset with their corresponding bounding boxes, using Matplotlib.

* **`split_dataset.py`**
  Splits the dataset into `train/` and `val/` folders. You can set the desired split ratio.

* **`generate_custom.py`**
  Generates a custom dataset configuration file (`.yaml`) pointing to the `train/` and `val/` directories, as required by YOLOv5.

* **`detect_metrics.py`**
  *Modified rom [Small-Object-Detection-with-YOLO](https://github.com/hamzagorgulu/Small-Object-Detection-with-YOLO).*
  Runs object detection on the test set and returns performance metrics such as Precision, Recall, and F1-Score. Also generates visualizations of false negatives and true positives.

---

## Setup Instructions

### 1. Clone this repository

```bash
git clone https://github.com/glaurung112/tinyobjectsdetection.git
cd tinyobjectsdetection/
```

---

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

> Once activated, your terminal should show something like:

```bash
(venv) user@machine:~/path/to/tinyobjectsdetection$
```

> This ensures all installations and executions stay within the virtual environment.

---

### 3. Clone the modified YOLOv5 version (HIC-YOLOv5)

```bash
git clone https://github.com/aash1999/yolov5-cbam.git
```

> This version includes architectural improvements specifically aimed at better small object detection.

---

### 4. Install dependencies

> This step may take a few minutes.

```bash
python3 install_dependencies.py
```

> If you want GPU support (CUDA), **do not run `pip install torch` directly**. Instead, follow instructions at:
> [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Dataset Preparation

### 5. Download the dataset from KaggleHub

```bash
python3 generate_dataset.py
```

> This creates a `dataset/` folder containing images.
> By default, it uses `daenys2000/small-object-dataset` with up to 200 black-background images containing white dots.

To customize:

```bash
python3 generate_dataset.py --source <kagglehub_dataset> --max_img 100 --min_points 5 --max_points 40
```

---

### 6. Generate YOLO-format labels

```bash
python3 generate_yolo_labels.py
```

> This creates a `dataset/labels/` folder with `.txt` annotation files matching each image.

---

### 7. Copy images and labels to a unified directory

```bash
python3 copy_to_cropped_base.py
```

> This produces a new folder `dataset/cropped_base/` with aligned `.png` and `.txt` files.

---

### 8. Verify all images have corresponding labels

```bash
python3 check_crop.py
```

---

### 9. Generate random image crops with labels

> This increases dataset diversity and improves the model's ability to detect small objects.

```bash
python3 crop_bulk.py --source dataset/cropped_base --output dataset/cropped_output --crops 10
```

> The `cropped_output` folder will contain 10x more images. You can adjust `--crops` to any number.

---

### 10. Visualize random samples with labels

```bash
python3 control_yolo_matplotlib.py
```

> A sample interface will appear:

```bash
[Image <number>]: <name>
[n]ext, [b]ack, [r]andom, [q]uit:
```

---

### 11. Split dataset into training and validation sets

```bash
python3 split_dataset.py
```

> Default: 80% training, 20% validation. Use `--ratio` to change this:

```bash
python3 split_dataset.py --ratio 0.7
```

---

## Expected Folder Structure

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

### 12. Automatically generate `custom.yaml`

```bash
python3 generate_custom.py
```

> This file is used by YOLOv5 to locate the training/validation sets.

---

## Training

### 13. Train the enhanced model

```bash
python3 yolov5-cbam/train.py --img 320 --batch 16 --epochs 150 --data custom.yaml --cfg yolov5-cbam/models/yolo5m-cbam-involution.yaml --weights yolov5s.pt --name small_dots_yolo
```

> Adjust `--img`, `--batch`, and `--epochs` based on your hardware:

* `--img 480`: better resolution for small object detail.
* `--batch 16`: suitable for mid-range GPUs.
* `--epochs 150`: a solid starting point for convergence.

---

## Model Evaluation

### Option 1: Use YOLOv5 built-in detection script

```bash
python3 yolov5-cbam/detect.py --weights yolov5-cbam/runs/train/small_dots_yolo/weights/best.pt --source dataset/val --img 320 --conf-thres 0.1
```

### Option 2: Custom evaluation script with metrics

```bash
python3 detect_metrics.py --weights yolov5-cbam/runs/train/small_dots_yolo/weights/best.pt --source dataset/val --img 320 --conf-thres 0.1
```

> This script creates two folders:

* `detections/`: images with predicted boxes
* `results/`: contains metrics like precision, recall, and F1, plus FN/TP distribution plots.

> Tip: use `--conf 0.05` or `--conf 0.1` to catch low-confidence small detections.

---

## Additional Notes

* Python >= 3.8 is recommended.
* Using a virtual environment (`venv`) is highly encouraged.
* All scripts must be run from the root directory.
* The folder `dataset_respaldo/` is unused and kept as an example.

---