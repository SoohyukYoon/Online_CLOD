from pathlib import Path
import json
import os
import random
import numpy as np
import glob


def make_stream(dataset_path, target_dataset, repeats, sigmas, seeds):

    clses = list(range(60, 80))

    cls_list = {
        # 40: "wine glass",
        # 41: "cup",
        # 42: "fork",
        # 43: "knife",
        # 44: "spoon",
        # 45: "bowl",
        # 46: "banana",
        # 47: "apple",
        # 48: "sandwich",
        # 49: "orange",
        # 50: "broccoli",
        # 51: "carrot",
        # 52: "hot dog",
        # 53: "pizza",
        # 54: "donut",
        # 55: "cake",
        # 56: "chair",
        # 57: "couch",
        # 58: "potted plant",
        # 59: "bed",
        60: "dining table",
        61: "toilet",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        68: "microwave",
        69: "oven",
        70: "toaster",
        71: "sink",
        72: "refrigerator",
        73: "book",
        74: "clock",
        75: "vase",
        76: "scissors",
        77: "teddy bear",
        78: "hair drier",
        79: "toothbrush"
    }
    
    target_labels = {
        # 1: "person",
        # 2: "bicycle",
        # 3: "car",
        # 4: "motorcycle",
        # 5: "airplane",
        # 6: "bus",
        # 7: "train",
        # 8: "truck",
        # 9: "boat",
        # 10: "traffic light",
        # 11: "fire hydrant",
        # 13: "stop sign",
        # 14: "parking meter",
        # 15: "bench",
        # 16: "bird",
        # 17: "cat",
        # 18: "dog",
        # 19: "horse",
        # 20: "sheep",
        # 21: "cow",
        # 22: "elephant",
        # 23: "bear",
        # 24: "zebra",
        # 25: "giraffe",
        # 27: "backpack",
        # 28: "umbrella",
        # 31: "handbag",
        # 32: "tie",
        # 33: "suitcase",
        # 34: "frisbee",
        # 35: "skis",
        # 36: "snowboard",
        # 37: "sports ball",
        # 38: "kite",
        # 39: "baseball bat",
        # 40: "baseball glove",
        # 41: "skateboard",
        # 42: "surfboard",
        # 43: "tennis racket",
        # 44: "bottle",
        # 46: "wine glass",
        # 47: "cup",
        # 48: "fork",
        # 49: "knife",
        # 50: "spoon",
        # 51: "bowl",
        # 52: "banana",
        # 53: "apple",
        # 54: "sandwich",
        # 55: "orange",
        # 56: "broccoli",
        # 57: "carrot",
        # 58: "hot dog",
        # 59: "pizza",
        # 60: "donut",
        # 61: "cake",
        # 62: "chair",
        # 63: "couch",
        # 64: "potted plant",
        # 65: "bed",
        67: "dining table",
        70: "toilet",
        72: "tv",
        73: "laptop",
        74: "mouse",
        75: "remote",
        76: "keyboard",
        77: "cell phone",
        78: "microwave",
        79: "oven",
        80: "toaster",
        81: "sink",
        82: "refrigerator",
        84: "book",
        85: "clock",
        86: "vase",
        87: "scissors",
        88: "teddy bear",
        89: "hair drier",
        90: "toothbrush"
    }
    
    # target_labels = [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89]
    target_labels = [67,70,72,73,74,75,76,77,78,79,
                     80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    index_dict = {target_label: 60+i for i, target_label in enumerate(target_labels)}
    
    cls_datalist = {}
    for id in cls_list.keys():
        cls_datalist[id] = []

    with open(dataset_path / 'annotations' / f"instances_{target_dataset}.json", "r") as file:
        labels_data = json.load(file)

    for i in range(len(labels_data['annotations'])):
        label = labels_data['annotations'][i]['category_id']
        image_id = labels_data['annotations'][i]['image_id']
        
        if label in target_labels:
            
            image_id = str(image_id).zfill(12)
            cls_datalist[index_dict[label]].append(target_dataset + '/' + image_id)
            
    for cls in clses:
        cls_datalist[cls] = list(set(cls_datalist[cls]))

    for repeat in repeats:
        for sigma in sigmas:
            for seed in seeds:
                random.seed(seed)
                np.random.seed(seed)
                n_classes = len(clses)
                cls_increment_time = np.zeros(n_classes)
                samples_list = []
                for cls in clses:
                    datalist = cls_datalist[cls]
                    random.shuffle(datalist)
                    samples_list.append(datalist)
                stream = []
                for i in range(n_classes):
                    times = np.random.normal(i/n_classes, sigma, size=len(samples_list[i]))
                    choice = np.random.choice(repeat, size=len(samples_list[i]))
                    times += choice
                    for ii, sample in enumerate(samples_list[i]):
                        if choice[ii] >= cls_increment_time[i]:
                            stream.append({'file_name': samples_list[i][ii], 'klass': cls_list[clses[i]], 'label':clses[i], 'time':times[ii]})
                random.shuffle(stream)
                stream = sorted(stream, key=lambda d: d['time'])
                data = {'cls_dict':cls_list, 'stream':stream, 'cls_addition':list(cls_increment_time)}

                with open(f'collections/{target_dataset}_sigma{int(sigma*100)}_repeat{repeat}_seed{seed}.json', 'w') as fp:
                    json.dump(data, fp)

if __name__ == "__main__":

    dataset_path = Path("./data/coco")
    target_dataset = 'train2017'

    repeats = [1]
    sigmas = [0.1]
    seeds = [1, 2, 3]
    init_cls = 1

    make_stream(dataset_path, target_dataset, repeats, sigmas, seeds)

# def make_stream(dataset_path, target_dataset, repeats, sigmas, seeds):

#     clses = list(range(10, 20))
#     cls_list = {
#         10: 'diningtable',
#         11: 'dog',
#         12: 'horse',
#         13: 'motorbike',
#         14: 'person',
#         15: 'pottedplant',
#         16: 'sheep',
#         17: 'sofa',
#         18: 'train',
#         19: 'tvmonitor',
#     }

#     cls_datalist = {}
#     for id in cls_list.keys():
#         cls_datalist[id] = []

#     with open(dataset_path / 'annotations' / f"instances_{target_dataset}.json", "r") as file:
#         labels_data = json.load(file)

#     for i in range(len(labels_data['annotations'])):
#         label = labels_data['annotations'][i]['category_id'] - 1 # index starts from 1, not 0
#         image_id = labels_data['annotations'][i]['image_id']
#         if label in clses:

#             if '2007' in target_dataset:
#                 image_id = str(image_id).zfill(6)
#             elif '2012' in target_dataset:
#                 image_id = str(image_id)[:4] + '_' + str(image_id)[4:]

#             cls_datalist[label].append(target_dataset + '/' + image_id)

#     for cls in clses:
#         cls_datalist[cls] = list(set(cls_datalist[cls]))

#     for repeat in repeats:
#         for sigma in sigmas:
#             for seed in seeds:
#                 random.seed(seed)
#                 np.random.seed(seed)
#                 n_classes = len(clses)
#                 cls_increment_time = np.zeros(n_classes)
#                 samples_list = []
#                 for cls in clses:
#                     datalist = cls_datalist[cls]
#                     random.shuffle(datalist)
#                     samples_list.append(datalist)
#                 stream = []
#                 for i in range(n_classes):
#                     times = np.random.normal(i/n_classes, sigma, size=len(samples_list[i]))
#                     choice = np.random.choice(repeat, size=len(samples_list[i]))
#                     times += choice
#                     for ii, sample in enumerate(samples_list[i]):
#                         if choice[ii] >= cls_increment_time[i]:
#                             stream.append({'file_name': samples_list[i][ii], 'class': cls_list[clses[i]], 'label':clses[i], 'time':times[ii]})
#                 random.shuffle(stream)
#                 stream = sorted(stream, key=lambda d: d['time'])
#                 data = {'cls_dict':cls_list, 'stream':stream, 'cls_addition':list(cls_increment_time)}

#                 with open(f'collections/{target_dataset}_sigma{int(sigma*100)}_repeat{repeat}_seed{seed}.json', 'w') as fp:
#                     json.dump(data, fp)

# if __name__ == "__main__":

#     dataset_path = Path("./data/voc")
#     target_dataset = 'train2012'

#     repeats = [1]
#     sigmas = [0.1]
#     seeds = [1, 2, 3]
#     init_cls = 1

#     make_stream(dataset_path, target_dataset, repeats, sigmas, seeds)