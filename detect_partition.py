import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5-cbam'))

from utils.torch_utils import select_device, smart_inference_mode
from utils.plots import Annotator, colors, save_one_box
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from models.common import DetectMultiBackend
import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

np.random.seed(0)

class_name_map = {
    0: "pedestrian",
    1: 'people',
    2: 'bicycle',
    3: "car",
    4: 'van',
    5: "truck",
    6: "tricycle",
    7: "awning - tricycle",
    8: "bus",
    9: "motor"}


def color_generator():
    label_colors = []
    for i in range(80):
        c = (int(np.random.randint(60, 255, 3)[0]),
             int(np.random.randint(60, 255, 3)[1]),
             int(np.random.randint(60, 255, 3)[2]))
        label_colors.append(c)
    return label_colors


def infer(im, model, dt):
    detections = []
    height, width = im.shape[:2]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None] 

    with dt[1]:
        pred = model(im, augment=False, visualize=False)

    with dt[2]:

        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

    for det in pred:
        if len(det):
            boxes = det[:, :4]
            boxes[:, 0].clamp_(0, width) 
            boxes[:, 1].clamp_(0, height)
            boxes[:, 2].clamp_(0, width)
            boxes[:, 3].clamp_(0, height)
            det[:, :4] = boxes
            for *xyxy, conf, cls in reversed(det):
                x_min = xyxy[0].item()
                y_min = xyxy[1].item()
                x_max = xyxy[2].item()
                y_max = xyxy[3].item()
                w = x_max - x_min
                h = y_max - y_min
                detections.append([cls.item(), conf.item(), [x_min, y_min, w, h]])

    return detections


def get_detections_from_partitions(model, img, partition_size, dt, ROI_flag=False, ROI_SIZE=0.8, CONFIDENCE_TH=0.25, NMS_TH=0.45, SELECT_HC_FLAG=False, NUMBER_OF_HC=5):
    detections = []
    h, w = img.shape[:2]

    if ROI_flag:
        roi = int(min(img.shape[:2]) * ROI_SIZE)
        y_max_roi = int(h - 0.1 * w)
        y_min_roi = int(y_max_roi - roi)
        x_min_roi = int(w / 2) - int(roi / 2)
        x_max_roi = int(w / 2) + int(roi / 2)
        cropped_img = img[y_min_roi:y_max_roi, x_min_roi:x_max_roi]
    else:
        y_max_roi = h
        y_min_roi = 0
        x_min_roi = 0
        x_max_roi = w
        cropped_img = img[y_min_roi:y_max_roi, x_min_roi:x_max_roi]

    img_h, img_w = cropped_img.shape[:2]
    num_of_partition_w = int(img_w / partition_size) + 1
    num_of_partition_h = int(img_h / partition_size) + 1

    partition_shift_w = 0 if num_of_partition_w == 1 else (partition_size * num_of_partition_w - img_w) / (num_of_partition_w - 1)
    partition_shift_h = 0 if num_of_partition_h == 1 else (partition_size * num_of_partition_h - img_h) / (num_of_partition_h - 1)

    x_pad = x_min_roi
    y_pad = y_min_roi

    img_boxes = []
    img_confs = []
    img_labels = []
    partition_rectangles = []

    for c in range(num_of_partition_w):
        for r in range(num_of_partition_h):
            bxmin = int(c * (partition_size - partition_shift_w))
            bxmax = bxmin + partition_size
            bymin = int(r * (partition_size - partition_shift_h))
            bymax = bymin + partition_size

            tile = cropped_img[bymin:bymax, bxmin:bxmax]
            tile_h, tile_w = tile.shape[:2]

            if tile_h != partition_size or tile_w != partition_size:
                padded_tile = np.zeros((partition_size, partition_size, 3), dtype=np.uint8)
                padded_tile[:tile_h, :tile_w] = tile
                current_input_img = padded_tile
            else:
                current_input_img = tile

            partition_rectangles.append([(bxmin + x_min_roi, bymin + y_min_roi),
                                         (bxmax + x_min_roi, bymax + y_min_roi)])

            detections = infer(current_input_img, model, dt)

            x_pad_in = x_pad + bxmin
            y_pad_in = y_pad + bymin
            for label, confidence, box in detections:
                box = list(map(int, np.rint(np.array(box))))
                box[0] += int(x_pad_in)
                box[1] += int(y_pad_in)
                img_boxes.append(box)
                img_confs.append(confidence)
                img_labels.append(int(label))

    indexes = cv2.dnn.NMSBoxes(img_boxes, img_confs, CONFIDENCE_TH, NMS_TH)
    
    img_boxes = np.array(img_boxes)[indexes]
    img_confs = np.array(img_confs)[indexes]
    img_labels = np.array(img_labels)[indexes]

    if SELECT_HC_FLAG:
        if len(img_boxes) > NUMBER_OF_HC:
            img_boxes = img_boxes[:NUMBER_OF_HC - 1]
            img_confs = img_confs[:NUMBER_OF_HC - 1]
            img_labels = img_labels[:NUMBER_OF_HC - 1]

    return img_boxes, img_confs, img_labels, partition_rectangles


class TestSmallObject:
    def __init__(self, iou_thres, save_dir):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.result_dict = {}
        self.iou_thres = iou_thres
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.fn_area_list = []
        self.tp_area_list = []
        self.fp_area_list = []
        self.iou_list = []
        self.non_labeled_frames = []
        self.gt_labels = None

    def read_gt(self, label, img_height, img_width):
        self.gt_labels = []
        with open(label) as f:
            Lines = f.readlines()
            for line in Lines:
                label = int(line.strip().split(' ')[0])
                x = float(line.strip().split(' ')[1]) * img_width
                y = float(line.strip().split(' ')[2]) * img_height
                w_r = float(line.strip().split(' ')[3]) * img_width
                h_r = float(line.strip().split(' ')[4]) * img_height
                x_min = x - w_r / 2
                y_min = y - h_r / 2
                x_max = x + w_r / 2
                y_max = y + h_r / 2
                detect_flag = 0
                self.gt_labels.append([label, x_min, y_min, x_max, y_max, detect_flag])

        f.close()

    def bb_intersection_over_union(self, boxA, boxB):

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def compare_detections(self, pred):
        cls = pred[0]
        x_min = pred[1]
        y_min = pred[2]
        x_max = pred[3]
        y_max = pred[4]
        detection_flag = False
        if cls not in self.result_dict:
            self.result_dict[cls] = {'tp': 0, 'fp': 0, 'fn': 0}
        for i in range(0, len(self.gt_labels)):
            current_label = self.gt_labels[i]
            gt_cls = current_label[0]
            x_min_gt = current_label[1]
            y_min_gt = current_label[2]
            x_max_gt = current_label[3]
            y_max_gt = current_label[4]

            iou = self.bb_intersection_over_union([x_min, y_min, x_max, y_max], [
                                                  x_min_gt, y_min_gt, x_max_gt, y_max_gt])
            if iou > self.iou_thres:
                if cls == gt_cls:
                    self.tp = self.tp + 1
                    self.tp_area_list.append((x_max_gt - x_min_gt) + (y_max_gt - y_min_gt) / 2)
                    detection_flag = True
                    self.gt_labels[i][-1] = 1
                    self.result_dict[cls]['tp'] += 1
                    self.iou_list.append(iou)
        if not detection_flag:
            self.fp = self.fp + 1
            self.fp_area_list.append((x_max - x_min) + (y_max - y_min) / 2)
            self.result_dict[cls]['fp'] += 1

    def false_negative_counter(self):
        for current_label in self.gt_labels:
            if current_label[5] == 0:
                cls = current_label[0]
                x_min_gt = current_label[1]
                y_min_gt = current_label[2]
                x_max_gt = current_label[3]
                y_max_gt = current_label[4]
                self.fn = self.fn + 1
                if cls not in self.result_dict:
                    self.result_dict[cls] = {'tp': 0, 'fp': 0, 'fn': 0}
                self.result_dict[cls]['fn'] += 1
                self.fn_area_list.append((x_max_gt - x_min_gt) + (y_max_gt - y_min_gt) / 2)

    def plot_width(self):
        if not (self.fn_area_list or self.tp_area_list or self.fp_area_list):
            print("No data to plot in `plot_width`. Skipping figure generation.")
            return

        _, bin_ratio_fn = np.histogram(self.fn_area_list)
        mean_area_fn = sum(self.fn_area_list) / len(self.fn_area_list) if self.fn_area_list else 0

        _, bin_ratio_fp = np.histogram(self.fp_area_list)
        mean_area_fp = sum(self.fp_area_list) / len(self.fp_area_list) if self.fp_area_list else 0

        _, bin_ratio_tp = np.histogram(self.tp_area_list)
        mean_area_tp = sum(self.tp_area_list) / len(self.tp_area_list) if self.tp_area_list else 0

        fig, axs = plt.subplots(2, 1)
        fig.set_figheight(16)
        fig.set_figwidth(20)

        all_vals = self.fn_area_list + self.tp_area_list + self.fp_area_list
        min_point = min(all_vals) - 5
        max_point = max(all_vals) + 5

        if self.fn_area_list:
            sns.distplot(self.fn_area_list, axlabel=False, rug=True, bins=bin_ratio_fn, ax=axs[0])
            axs[0].axvline(mean_area_fn, color='r', linestyle='--',
                        label='Mean = ' + str(round(mean_area_fn, 2)))
            axs[0].title.set_text('(W+H)/2 Distribution of False Negatives; mean: %0.2f' % mean_area_fn)
            axs[0].set_xlim([min_point, max_point])
            axs[0].legend()
            axs[0].grid(True)
        else:
            axs[0].set_visible(False)

        if self.tp_area_list:
            sns.distplot(self.tp_area_list, axlabel=False, rug=True, bins=bin_ratio_tp, ax=axs[1])
            axs[1].axvline(mean_area_tp, color='r', linestyle='--',
                        label='Mean = ' + str(round(mean_area_tp, 2)))
            axs[1].title.set_text('(W+H)/2 Distribution of True Positives; mean: %0.2f' % mean_area_tp)
            axs[1].set_xlim([min_point, max_point])
            axs[1].legend()
            axs[1].grid(True)
        else:
            axs[1].set_visible(False)

        fig.savefig(f'{self.save_dir}/Radius_Distributions.png')


    def save(self):
        self.plot_width()
        total_samples = self.tp + self.fn
        total_predictions = self.tp + self.fp
        total_total = self.tp + self.fp + self.fn

        mean_iou = round(sum(self.iou_list) / len(self.iou_list), 4) if self.iou_list else 0.0

        accuracy = self.tp / total_total if total_total > 0 else 0.0
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        f1_score = self.tp / (self.tp + 0.5 * (self.fp + self.fn)) if (self.tp + 0.5 * (self.fp + self.fn)) > 0 else 0.0

        with open(f"{self.save_dir}/results.txt", 'w') as f:
            f.write(f'Resultados Generales\n')
            f.write(f'Total de imágenes procesadas: {total_samples}\n\n')
            f.write(f'True Positives (TP):  {self.tp}\n')
            f.write(f'False Positives (FP): {self.fp}\n')
            f.write(f'False Negatives (FN): {self.fn}\n\n')

            f.write(f'Métricas Generales:\n')
            f.write(f'Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n')
            f.write(f'Precision: {precision:.4f} ({precision*100:.2f}%)\n')
            f.write(f'Recall:    {recall:.4f} ({recall*100:.2f}%)\n')
            f.write(f'F1 Score:  {f1_score:.4f} ({f1_score*100:.2f}%)\n')
            f.write(f'Mean IoU:  {mean_iou:.4f}\n\n')

            f.write(f'Métricas por Clase:\n')
            for cls in self.result_dict:
                tp = self.result_dict[cls]['tp']
                fp = self.result_dict[cls]['fp']
                fn = self.result_dict[cls]['fn']
                total_cls = tp + fp + fn

                cls_accuracy = tp / total_cls if total_cls > 0 else 0.0
                cls_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                cls_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                cls_f1 = tp / (tp + 0.5 * (fp + fn)) if (tp + 0.5 * (fp + fn)) > 0 else 0.0

                class_name = class_name_map.get(cls, f"clase {cls}")
                f.write(f'Clase: {class_name}\n')
                f.write(f'  TP: {tp}, FP: {fp}, FN: {fn}\n')
                f.write(f'  Accuracy:  {cls_accuracy:.4f} ({cls_accuracy*100:.2f}%)\n')
                f.write(f'  Precision: {cls_precision:.4f} ({cls_precision*100:.2f}%)\n')
                f.write(f'  Recall:    {cls_recall:.4f} ({cls_recall*100:.2f}%)\n')
                f.write(f'  F1 Score:  {cls_f1:.4f} ({cls_f1*100:.2f}%)\n\n')


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',
        source=ROOT / 'data/images',
        data=ROOT / 'data/coco128.yaml',
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        view_img=False,
        save_txt=False,
        save_conf=False, 
        save_crop=False,
        nosave=False,
        classes=None,
        agnostic_nms=False, 
        augment=False, 
        visualize=False, 
        update=False,  
        project=ROOT / 'runs/detect',  
        name='exp',
        exist_ok=False, 
        line_thickness=3, 
        hide_labels=False, 
        hide_conf=False, 
        half=False,
        dnn=False,
        vid_stride=1,
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt') 
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    results = TestSmallObject(0.2, 'parti480')
    for path, im, im0s, vid_cap, s in tqdm(dataset):
        gt_path = path[:path.rfind('.')] + '.txt'
        img_height, img_width = im0s.shape[:2]
        results.read_gt(gt_path, img_height, img_width)
        label_colors = color_generator()
        img_boxes, img_confs, img_labels, partition_rectangles = get_detections_from_partitions(
            model, im0s, 480, dt,
            CONFIDENCE_TH=conf_thres,
            NMS_TH=iou_thres)

        for i, rect in enumerate(partition_rectangles):
            cv2.rectangle(im0s, rect[0], rect[1], (0, 0, 0), 1)

        for i, box in enumerate(img_boxes):
            x_min = box[0]
            y_min = box[1]
            w = box[2]
            h = box[3]
            x_max = x_min + int(w)
            y_max = y_min + int(h)
            results.compare_detections([img_labels[i], x_min, y_min, x_max, y_max])
            cv2.rectangle(im0s, (x_min, y_min), (x_max, y_max), label_colors[img_labels[i]], 2)
            cv2.putText(im0s, class_name_map[img_labels[i]], (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 1)

        results.false_negative_counter()
        cv2.imwrite(os.path.join('parti480', path.split('/')[-1]), im0s)
    results.save()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)