import os
import random
import shutil
import argparse

print("Dividiendo el dataset en train/ y val/...")
print()

parser = argparse.ArgumentParser(description="Divide un dataset en entrenamiento y validación.")
parser.add_argument('--ratio', type=float, default=0.8, help='Proporción de imágenes para entrenamiento (por defecto 0.8)')
args = parser.parse_args()

SOURCE_DIR = 'dataset/cropped_output'
TARGET_DIR = 'dataset'
SPLIT_RATIO = args.ratio

for split in ['train', 'val']:
    os.makedirs(os.path.join(TARGET_DIR, split), exist_ok=True)

all_files = os.listdir(SOURCE_DIR)
images = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Mezclar random
random.shuffle(images)

split_idx = int(len(images) * SPLIT_RATIO)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

def move_files(img_list, split):
    for img_file in img_list:
        label_file = os.path.splitext(img_file)[0] + '.txt'

        src_img = os.path.join(SOURCE_DIR, img_file)
        src_lbl = os.path.join(SOURCE_DIR, label_file)

        dst_img = os.path.join(TARGET_DIR, split, img_file)
        dst_lbl = os.path.join(TARGET_DIR, split, label_file)

        shutil.copy(src_img, dst_img)

        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, dst_lbl)
        else:
            print(f"Advertencia: No se encontró label para {img_file}")

move_files(train_imgs, 'train')
move_files(val_imgs, 'val')

print(f"\nDataset dividido correctamente:")
print(f"   - {len(train_imgs)} imágenes en '{TARGET_DIR}/train'")
print(f"   - {len(val_imgs)} imágenes en '{TARGET_DIR}/val'")
