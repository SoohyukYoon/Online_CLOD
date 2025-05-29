from pathlib import Path
import json
import os
import random
import numpy as np
import glob

def make_stream(dataset_path, target_dataset, repeats, sigmas, seeds):

    clses = list(range(10, 20))
    cls_list = {
        10: 'diningtable',
        11: 'dog',
        12: 'horse',
        13: 'motorbike',
        14: 'person',
        15: 'pottedplant',
        16: 'sheep',
        17: 'sofa',
        18: 'train',
        19: 'tvmonitor',
    }

    cls_datalist = {}
    for id in cls_list.keys():
        cls_datalist[id] = []

    with open(dataset_path / 'annotations' / f"instances_{target_dataset}.json", "r") as file:
        labels_data = json.load(file)

    for i in range(len(labels_data['annotations'])):
        label = labels_data['annotations'][i]['category_id'] - 1 # index starts from 1, not 0
        image_id = labels_data['annotations'][i]['image_id']
        if label in clses:

            if '2007' in target_dataset:
                image_id = str(image_id).zfill(6)
            elif '2012' in target_dataset:
                image_id = str(image_id)[:4] + '_' + str(image_id)[4:]

            cls_datalist[label].append(target_dataset + '/' + image_id)

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
                            stream.append({'file_name': samples_list[i][ii], 'class': cls_list[clses[i]], 'label':clses[i], 'time':times[ii]})
                random.shuffle(stream)
                stream = sorted(stream, key=lambda d: d['time'])
                data = {'cls_dict':cls_list, 'stream':stream, 'cls_addition':list(cls_increment_time)}

                with open(f'collections/{target_dataset}_sigma{int(sigma*100)}_repeat{repeat}_seed{seed}.json', 'w') as fp:
                    json.dump(data, fp)

if __name__ == "__main__":

    dataset_path = Path("./data/voc")
    target_dataset = 'train2012'

    repeats = [1]
    sigmas = [0.1]
    seeds = [1, 2, 3]
    init_cls = 1

    make_stream(dataset_path, target_dataset, repeats, sigmas, seeds)