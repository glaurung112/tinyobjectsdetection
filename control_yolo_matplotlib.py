import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import random

print("Desplegando muestras de los datos con sus labels...")
print()

DATASET_DIR = 'dataset/cropped_output'
CLASS_NAMES = ['dot']
LABEL_COLORS = [(1, 0, 0)]

def load_labels(txt_file, img_shape):
    boxes = []
    if not os.path.exists(txt_file):
        return boxes
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls, xc, yc, w, h = map(float, parts)
            cls = int(cls)
            x1 = (xc - w / 2) * img_shape[1]
            y1 = (yc - h / 2) * img_shape[0]
            x2 = (xc + w / 2) * img_shape[1]
            y2 = (yc + h / 2) * img_shape[0]
            boxes.append((cls, x1, y1, x2, y2))
    return boxes

def show_image_with_boxes(img_path, label_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error al leer imagen: {img_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = load_labels(label_path, img.shape)

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for cls, x1, y1, x2, y2 in boxes:
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=1, edgecolor=LABEL_COLORS[cls], facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, CLASS_NAMES[cls], color=LABEL_COLORS[cls], fontsize=8)

    ax.set_title(os.path.basename(img_path))
    plt.axis('off')
    plt.show()

# === MAIN LOOP ===
all_files = sorted(os.listdir(DATASET_DIR))
image_files = [f for f in all_files if f.endswith(('.png', '.jpg', '.jpeg'))]
index = 0

while True:
    img_path = os.path.join(DATASET_DIR, image_files[index])
    label_name = os.path.splitext(image_files[index])[0] + '.txt'
    label_path = os.path.join(DATASET_DIR, label_name)

    show_image_with_boxes(img_path, label_path)

    print(f"\n[Imagen {index+1}/{len(image_files)}] → {image_files[index]}")
    key = input("[n]ext, [b]ack, [r]andom, [q]uit: ").lower()

    if key == 'n':
        index = (index + 1) % len(image_files)
    elif key == 'b':
        index = (index - 1) % len(image_files)
    elif key == 'r':
        index = random.randint(0, len(image_files) - 1)
    elif key == 'q':
        break
    else:
        print("Opción no válida.")
