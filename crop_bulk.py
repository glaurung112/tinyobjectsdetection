import os
import cv2
import numpy as np
import re
from tqdm import tqdm
import time

import argparse

print("Generando recortes aleatorios...")
print()

image_ext = ['.png', '.jpg', '.bmp']
CROP_SIZE = 224

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, required=True, help='Carpeta que contiene imágenes y etiquetas YOLO')
parser.add_argument('--output', type=str, default='cropped_output', help='Carpeta de salida')
parser.add_argument('--crops', type=int, default=10, help='Número de crops aleatorios por imagen')
args = parser.parse_args()

NUMBER_OF_CROPS = args.crops
PATH = args.source
SAVE_DIR = args.output

def natural_sort(images):

    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(images, key=alphanum_key)


def get_files(path):

    images = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                images.append(apath)
    return natural_sort(images)


def read_gt(label):

    with open(label) as f:
        labels = []
        Lines = f.readlines()
        for line in Lines:
            label = int(line.strip().split(' ')[0])
            x = float(line.strip().split(' ')[1])
            y = float(line.strip().split(' ')[2])
            w_r = float(line.strip().split(' ')[3])
            h_r = float(line.strip().split(' ')[4])
            labels.append([label, x, y, w_r, h_r])
    f.close()
    return labels

def random_crop(t, image, crop_size, num_crops = NUMBER_OF_CROPS):

    num_labels = len(t)
    crop_img_list = []
    t_new_list = []

    if image.shape[0] < crop_size or image.shape[1] < crop_size:
        return [t], [image]


    for _ in range(num_crops):

        crop_x = np.random.randint(low=0, high = max(0,image.shape[1]-crop_size), size=1)[0]
        crop_y = np.random.randint(low=0, high= max(0,image.shape[0]-crop_size), size=1)[0]

        labels_in_crop = []
        for label in t:
            x = label[1]*image.shape[1]
            y = label[2]*image.shape[0]
            w = label[3]*image.shape[1]
            h = label[4]*image.shape[0]

            xmin = int(round(x-w/2))
            ymin = int(round(y-h/2))
            xmax = int(round(x+w/2))
            ymax = int(round(y+h/2))

            if xmin >= crop_x and xmax <= crop_x+crop_size and ymin >= crop_y and ymax <= crop_y+crop_size:
                labels_in_crop.append(label)

        if len(labels_in_crop) == 0:
            continue

        crop_img = image[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size, :]

        t_new = []
        for label in labels_in_crop:
            obj = label[0]
            x = label[1]*image.shape[1]
            y = label[2]*image.shape[0]
            w = label[3]*image.shape[1]
            h = label[4]*image.shape[0]

            new_x = (x-crop_x)/crop_img.shape[1]
            new_y = (y-crop_y)/crop_img.shape[0]
            new_w = w / crop_img.shape[1]
            new_h = h / crop_img.shape[0]

            t_new.append([obj, new_x, new_y, new_w, new_h])

        crop_img_list.append(crop_img)
        t_new_list.append(t_new)

    return t_new_list, crop_img_list



def random_crop_column(t, image, crop_size):

    x = t[1]*image.shape[1]
    w = t[3]*image.shape[1]

    x_min = int(round(x-w/2))
    x_max = int(round(x+w/2))

    crop_x = np.random.randint(low=max(
        0, x_max+1-crop_size), high=min(x_min-1, image.shape[1]-crop_size), size=1)[0]

    crop_img = image[:, crop_x:crop_x+crop_size]

    new_x = (x-crop_x)/crop_img.shape[1]

    new_w = w/crop_img.shape[1]

    t_new = [t[0], new_x, t[2], new_w, t[4]]

    return t_new, crop_img


def random_crop_column_n(t, image, crop_size):

    crop_x = np.random.randint(
        low=0, high=image.shape[1]-crop_size-1, size=1)[0]
    
    crop_img = image[:, crop_x:crop_x+crop_size]

    t_new = t

    return t_new, crop_img

def imread_8bit(img_path):
    img_anydept = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    img8_8bit = (img_anydept/256).astype('uint8')
    return img8_8bit


def imread_normalized(img_path):
    img_anydept = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    img8_8bit_normalized = cv2.normalize(
        img_anydept, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return img8_8bit_normalized


def main():
    start = time.time()
    data_path = PATH
    save_dir = SAVE_DIR

    i = 0

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)



    image_files = get_files(data_path)
    for img_path in tqdm(image_files):

        label_path = img_path[:img_path.rfind('.')] + '.txt'
        if not os.path.exists(label_path):
            print(img_path)
            continue
        current_shot = img_path.split('/')[-2]


        current_image = cv2.imread(img_path)

        labels = read_gt(label_path)
        
        t_new_list, crop_img_list = random_crop(labels, current_image, crop_size=CROP_SIZE)

        i = 0
        for t_new in t_new_list: 
            tag = "_crop_" + str(i)
            new_txt = open(os.path.join(save_dir, label_path.split('/')[-1].split(".")[0] + tag + ".txt"), 'w')
            i += 1
            for ele in t_new:
                for value in ele:
                    new_txt.write(str(value)+' ')
                new_txt.write('\n')
            new_txt.close()
        i = 0
        for crop_img in crop_img_list:
            tag = "_crop_" + str(i)
            i += 1
            cv2.imwrite(os.path.join(save_dir, img_path.split('/')[-1].split(".")[0] + tag + ".png"), crop_img)

    end = time.time()
    print("Tiempo total: ", end-start)


if __name__ == '__main__':
    main()