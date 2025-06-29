import os
import cv2
import numpy as np
import shutil
import kagglehub
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Filtra imágenes por cantidad de puntos blancos desde un dataset de KaggleHub.")
    parser.add_argument('--source', type=str, default="daenys2000/small-object-dataset", help='Nombre del dataset de KaggleHub')
    parser.add_argument('--max_img', type=int, default=200, help='Máximo de imágenes a guardar')
    parser.add_argument('--min_points', type=int, default=1, help='Cantidad mínima de puntos por imagen')
    parser.add_argument('--max_points', type=int, default=60, help='Cantidad máxima de puntos por imagen')
    return parser.parse_args()

def find_base_path_with_train_test(root):
    for dirpath, dirnames, _ in os.walk(root):
        if 'train' in dirnames and 'test' in dirnames:
            return dirpath
    return None

def main():
    opt = parse_args()

    print(f"Descargando dataset '{opt.source}' desde KaggleHub...")
    dataset_path = kagglehub.dataset_download(opt.source)
    print("Ruta del dataset:", dataset_path)

    base_dataset = find_base_path_with_train_test(dataset_path)
    if base_dataset is None:
        print("No se encontraron carpetas 'train' y 'test' dentro del dataset.")
        exit(1)

    paths = [os.path.join(base_dataset, 'train'), os.path.join(base_dataset, 'test')]
    output_dir = 'dataset/images'

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for base in paths:
        if not os.path.exists(base):
            print(f"Directorio no encontrado: {base}")
            continue

        for category in os.listdir(base):
            category_path = os.path.join(base, category)
            dots_path = os.path.join(category_path, 'gt-dots')

            if not os.path.isdir(dots_path):
                continue

            for file in sorted(os.listdir(dots_path)):
                if not file.endswith('.png'):
                    continue

                dot_img_path = os.path.join(dots_path, file)
                dot_img = cv2.imread(dot_img_path, cv2.IMREAD_GRAYSCALE)
                if dot_img is None:
                    continue

                num_points = np.count_nonzero(dot_img > 0)

                if opt.min_points <= num_points <= opt.max_points:
                    base_name = os.path.splitext(file)[0]
                    new_name = f"{category}_{base_name}__{num_points}pts.png"
                    out_path = os.path.join(output_dir, new_name)

                    shutil.copyfile(dot_img_path, out_path)
                    count += 1

                if count >= opt.max_img:
                    break
            if count >= opt.max_img:
                break
        if count >= opt.max_img:
            break

    print(f"\nSe guardaron {count} imágenes en '{output_dir}'.")

if __name__ == "__main__":
    main()
