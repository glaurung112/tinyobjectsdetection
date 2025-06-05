import os
import cv2
import numpy as np
import shutil

paths = ['train', 'test']  # Buscar en train/ y test/
output_dir = 'filtered_dots'
max_img = 200
min_puntos = 0
max_puntos = 60

# Crear la carpeta de salida
os.makedirs(output_dir, exist_ok=True)

# Contador de imagenes seleccionadas
count = 0

# Recorrer train/ y test/
for base in paths:
    if not os.path.exists(base):
        continue

    # Recorrer subdirectorios
    for category in os.listdir(base):
        category_path = os.path.join(base, category)
        dots_path = os.path.join(category_path, 'gt-dots')

        if not os.path.isdir(dots_path):
            continue

        for file in sorted(os.listdir(dots_path)):
            if not file.endswith('.png'):
                continue

            dot_img_path = os.path.join(dots_path, file)

            # Leer la imagen de puntos
            dot_img = cv2.imread(dot_img_path, cv2.IMREAD_GRAYSCALE)
            if dot_img is None:
                continue

            # Contar puntos (pixeles blancos)
            num_points = np.count_nonzero(dot_img > 0)

            # Revisar si tiene el maximo de puntos deseados
            if min_puntos <= num_points <= max_puntos:
                # Generar nuevo nombre
                base_name = os.path.splitext(file)[0]
                new_name = f"{category}_{base_name}__{num_points}pts.png"
                out_path = os.path.join(output_dir, new_name)

                # Guardar imagen
                shutil.copyfile(dot_img_path, out_path)
                count += 1

            if count >= max_img:
                break
        if count >= max_img:
            break
    if count >= max_img:
        break

print(f"Se guardaron {count} imagenes en '{output_dir}'.")
