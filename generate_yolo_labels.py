import cv2
import os
import numpy as np

IMAGE_DIR = 'dataset/images'
LABEL_DIR = os.path.join('dataset/', 'labels')
os.makedirs(LABEL_DIR, exist_ok=True)

# Parámetros
THRESHOLD = 100 
SAVE_DEBUG = False

images_found = 0
images_with_labels = 0

print("Generando labels en formato YOLO...")
print()

for filename in os.listdir(IMAGE_DIR):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        images_found += 1
        image_path = os.path.join(IMAGE_DIR, filename)
        label_path = os.path.join(LABEL_DIR, filename.rsplit('.', 1)[0] + '.txt')

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"No se pudo leer la imagen: {filename}")
            continue

        h, w = img.shape

        _, binary = cv2.threshold(img, THRESHOLD, 255, cv2.THRESH_BINARY)
        if SAVE_DEBUG:
            cv2.imwrite(f'debug_{filename}', binary)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        with open(label_path, 'w') as f:
            count = 0
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw < 1 or bh < 1:
                    continue

                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                bw_norm = bw / w
                bh_norm = bh / h

                f.write(f"0 {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")
                count += 1

        if count > 0:
            images_with_labels += 1
            print(f"{filename}: {count} objetos")
        else:
            os.remove(label_path)
            print(f"{filename}: sin objetos detectados")

print(f"\nResumen:")
print(f"  - Imágenes encontradas: {images_found}")
print(f"  - Imágenes con labels: {images_with_labels}")
