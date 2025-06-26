import os
import shutil

src_images = 'dataset/images'
src_labels = 'dataset/labels'

dst_base = 'dataset/cropped_base'
os.makedirs(dst_base, exist_ok=True)

# Copiar imágenes
for fname in os.listdir(src_images):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        shutil.copy2(os.path.join(src_images, fname), os.path.join(dst_base, fname))

# Copiar labels
for fname in os.listdir(src_labels):
    if fname.lower().endswith('.txt'):
        shutil.copy2(os.path.join(src_labels, fname), os.path.join(dst_base, fname))

print("Copia completa en dataset/cropped_base/")
