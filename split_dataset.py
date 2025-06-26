import os
import random
import shutil

SOURCE_DIR = 'dataset/cropped_output'
TARGET_DIR = 'dataset'
SPLIT_RATIO = 0.8  # 80% entrenamiento, 20% validación

for split in ['train', 'val']:
    os.makedirs(os.path.join(TARGET_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, 'labels', split), exist_ok=True)

all_files = os.listdir(SOURCE_DIR)
images = [f for f in all_files if f.endswith(('.png', '.jpg', '.jpeg'))]

# Mezclar random
random.shuffle(images)

# Dividir
split_idx = int(len(images) * SPLIT_RATIO)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

def move_files(img_list, split):
    for img_file in img_list:
        label_file = os.path.splitext(img_file)[0] + '.txt'

        src_img = os.path.join(SOURCE_DIR, img_file)
        src_lbl = os.path.join(SOURCE_DIR, label_file)

        dst_img = os.path.join(TARGET_DIR, 'images', split, img_file)
        dst_lbl = os.path.join(TARGET_DIR, 'labels', split, label_file)

        # Copiar imagen
        shutil.copy(src_img, dst_img)

        # Copiar label
        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, dst_lbl)
        else:
            print(f"[WARN] No se encontró label para: {img_file}")

move_files(train_imgs, 'train')
move_files(val_imgs, 'val')

print(f"Dataset dividido: {len(train_imgs)} entrenamiento, {len(val_imgs)} validación.")
