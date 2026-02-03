import logging.config
import random
import os
import json
from typing import List
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch

import glob

logger = logging.getLogger()


from statistics import mean
from typing import Generator, List, Tuple, Union
from damo.dataset.datasets.mosaic_wrapper import MosaicWrapper
from damo.dataset.datasets.coco import COCODataset
from damo.dataset.collate_batch import BatchCollator, BatchCollator2, HarmoniousBatchCollator
from damo.dataset.transforms import build_transforms, build_transforms_memorydataset
from damo.structures.bounding_box import BoxList

import os
from pathlib import Path
from PIL import Image
import re

def get_statistics(dataset: str):
    """
    Returns statistics of the dataset given a string of dataset name. To add new dataset, please add required statistics here
    """
    if dataset == 'VOC_10_10':
        return 20, 'data/voc/images', 'data/voc/annotations' ## 경로 수정
    elif dataset == 'VOC_15_5':
        return 20, 'data/voc/images', 'data/voc/annotations' ## 경로 수정
    elif dataset == 'BDD_domain' or dataset == 'BDD_domain_small':
        return 13, 'data/bdd100k/images', 'data/bdd100k/annotations'
    elif dataset == 'SHIFT_domain' or dataset == 'SHIFT_domain_small' or dataset == 'SHIFT_domain_small2' or 'hanhwa' in dataset:
        return 6, 'data/shift/images', 'data/shift/annotations'
        # return 6, '/disk1/jhpark/clod/data/shift/images', 'data/shift/annotations'
    elif 'MILITARY_SYNTHETIC_domain' in dataset:
        return 9, 'data/military_synthetic/images', 'data/military_synthetic/annotations'
    elif dataset == 'VisDrone_3_4':
        return 7, 'data/VisDrone2019-VID/images', 'data/VisDrone2019-VID/annotations'
    elif dataset == 'COCO_70_10' or dataset == 'COCO_60_20':
        return 80, 'data/coco/images', 'data/coco/annotations'
    elif dataset == 'HS_TOD_class' or dataset == 'HS_TOD_class_new':
        return 8, 'data/HS_TOD_winter/images', 'data/HS_TOD_winter/annotations'
    elif dataset == 'HS_TOD_domain':
        return 1, 'data/HS_TOD_winter/images', 'data/HS_TOD_winter/annotations'
    else:
        raise ValueError("Wrong dataset name")

def get_pretrained_statistics(dataset: str):
    if dataset == 'VOC_10_10':
        return 10, 'data/voc_10/images', 'data/voc_10/annotations' ## 경로 수정
    elif dataset == 'VOC_15_5':
        return 10, 'data/voc_15/images', 'data/voc_15/annotations' ## 경로 수정
    elif dataset == 'BDD_domain' or dataset == 'BDD_domain_small':
        return 13, 'data/bdd100k_source/images', 'data/bdd100k_source/annotations'
    elif dataset == 'SHIFT_domain' or dataset == 'SHIFT_domain_small' or dataset == 'SHIFT_domain_small2' or 'hanhwa' in dataset:
        return 6, 'data/shift_source/images', 'data/shift_source/annotations'
        # return 6, '/disk1/jhpark/clod/data/shift/images', 'data/shift_source/annotations'
    elif 'MILITARY_SYNTHETIC_domain' in dataset:
        return 9, 'data/military_synthetic_domain_source/images', 'data/military_synthetic_domain_source/annotations'
    elif dataset == 'VisDrone_3_4':
        return 3, 'data/VisDrone2019-VID-3/images', 'data/VisDrone2019-VID-3/annotations'
    elif dataset == 'COCO_70_10':
        return 70, 'data/coco_70/images', 'data/coco_70/annotations'
    elif dataset == 'COCO_60_20':
        return 60, 'data/coco_60/images', 'data/coco_60/annotations'
    elif dataset == 'HS_TOD_class' or dataset == 'HS_TOD_domain' or dataset == 'HS_TOD_class_new':
        return 1, '', ''
    else:
        raise ValueError("Wrong dataset name")
    
def get_exposed_classes(dataset: str):
    if dataset == 'VOC_10_10':
        return ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow']
    elif dataset == 'VOC_15_5':
        return ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person']
    elif dataset == 'BDD_domain' or dataset == 'BDD_domain_small':
        return ['pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'traffic light', 'traffic sign', 'train', 'trailer', 'other person', 'other vehicle']
    elif dataset == 'SHIFT_domain' or dataset == 'SHIFT_domain_small' or dataset == 'SHIFT_domain_small2' or 'hanhwa' in dataset:
        return ['pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']
    elif 'MILITARY_SYNTHETIC_domain' in dataset:
        return ['fishing vessel', 'warship', 'merchant vessel', 'fixed-wing aircraft', 'rotary-wing aircraft', 'Unmanned Aerial Vehicle', 'bird', 'leaflet', 'waste bomb']
    elif dataset == 'VisDrone_3_4':
        return ['people', 'bicycle', 'car']
    elif dataset == 'COCO_70_10':
        return ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven']#, 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    elif dataset == 'COCO_60_20':
        return ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed']#, 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    elif dataset == 'HS_TOD_class' or dataset == 'HS_TOD_domain' or dataset == 'HS_TOD_class_new':
        return ['person']
    
def get_train_datalist(dataset, sigma, repeat, rnd_seed):
    with open(f"collections/{dataset}/{dataset}_sigma{sigma}_repeat{repeat}_seed{rnd_seed}.json") as fp:
        train_list = json.load(fp)
    return train_list['stream'] #, train_list['cls_dict'], train_list['cls_addition']
        

def get_test_datalist(dataset) -> List:
    try:
        print("test name", f"collections/{dataset}/{dataset}_val.json")
        return pd.read_json(f"collections/{dataset}/{dataset}_val.json").to_dict(orient="records")
    except:
        print("test name", f"collections/{dataset}/{dataset}_val2.json")
        return pd.read_json(f"collections/{dataset}/{dataset}_val2.json").to_dict(orient="records")        

class MemoryDataset(COCODataset):
    def __init__(self, ann_file, root, transforms=None, class_names=None,
                 dataset=None, cls_list=None, device=None, data_dir=None, memory_size=None, 
                 init_buffer_size=None, image_size=(640, 640), aug=None
                 ):
        super().__init__(ann_file, root, transforms, class_names)
        # self.args = args
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_sizes = [int(image_size[0]), int(image_size[1])]
        self.memory_size = memory_size

        self.buffer = []
        self.stream_data = []

        self.dataset = dataset
        self.device = device
        self.data_dir = data_dir
        
        self.class_usage_cnt = []
        self.cls_list = cls_list if cls_list else []
        
        
        self.cls_count = []
        self.cls_idx = []
        self.cls_train_cnt = np.array([0,]*len(cls_list)) if cls_list else np.array([])
        
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
            # l
            self.name2id = {re.sub(r'\.(jpg|jpeg|png)$', '', ann['file_name'], flags=re.I): ann['id'] for ann in annotations['images']}
        
        if 'HS_TOD' not in self.dataset:
            self.build_initial_buffer(init_buffer_size)

        n_classes, image_dir, label_path = get_statistics(dataset=self.dataset)
        self.image_dir = image_dir
        self.label_path = label_path
        
        # set transform
        transforms = aug.transform
        transforms = build_transforms_memorydataset(**transforms)
        self._transforms = transforms
        
        print("mosaic_prob", aug.mosaic_mixup.mosaic_prob)
        print("mixup_prob", aug.mosaic_mixup.mixup_prob)
        print("degrees", aug.mosaic_mixup.degrees)
        print("translate", aug.mosaic_mixup.translate)
        print("mosaic_scale", aug.mosaic_mixup.mosaic_scale)
        
        
        self.mosaic_wrapper = MosaicWrapper(
            dataset=self,                      # <-- this class will provide pull_item/load_anno
            img_size=image_size,
            mosaic_prob=aug.mosaic_mixup.mosaic_prob,                   # always “allowed”; actual apply is still random inside wrapper
            mixup_prob=aug.mosaic_mixup.mixup_prob,
            transforms=self._transforms,                   # we'll run your own transforms after mosaic
            degrees=aug.mosaic_mixup.degrees,
            translate=aug.mosaic_mixup.translate,
            mosaic_scale=aug.mosaic_mixup.mosaic_scale,
            keep_ratio=True,
        )
        self.use_mosaic_mixup=True
        
        self.batch_collator = BatchCollator(size_divisible=32)
    
    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, inp):
        if type(inp) is tuple:
            idx = inp[1]
        else:
            idx = inp

        # Feel like this is the source of all misery 
        img, anno = super(COCODataset, self).__getitem__(idx)
        # PIL to numpy array
        return img, anno, idx

    def load_data(self, img_path, is_stream, data_class=None):
        # debug
        img_path = re.sub(r'\.(jpg|jpeg|png)$', '', img_path, flags=re.I)
        #print("[DEBUG] after splitext:", img_path)

        try:
            if not is_stream and self.dataset == 'VisDrone_3_4':
                img_path = 'sequences/'+img_path
            image_id = self.name2id[img_path]
        except:
            image_id = self.name2id[img_path.split('/')[-1]]
        idx = self.ids.index(image_id)
        img, label, img_id = self.__getitem__(idx)
    
        label = self.load_valid_labels(label, is_stream, data_class=data_class)
        
        return img, label, img_id
    
    def load_valid_labels(self, label, is_stream, data_class):
        indices_to_keep = []
        
        for i in range(len(label)):
            if is_stream:
                if self.dataset == 'VOC_10_10' or self.dataset == 'VOC_15_5' or self.dataset == 'VisDrone_3_4' or 'COCO' in self.dataset or 'HS_TOD_class' in self.dataset:
                    if self.contiguous_class2id[self.ori_id2class[label[i]['category_id']]] <= data_class: # == data_class:
                        indices_to_keep.append(i)
                else:
                    indices_to_keep.append(i)
            else:
                if self.dataset == 'VOC_10_10':
                    if self.contiguous_class2id[self.ori_id2class[label[i]['category_id']]] < 10:
                        indices_to_keep.append(i)
                elif self.dataset == 'VOC_15_5':
                    if self.contiguous_class2id[self.ori_id2class[label[i]['category_id']]] < 15:
                        indices_to_keep.append(i)
                elif self.dataset == 'VisDrone_3_4':
                    if self.contiguous_class2id[self.ori_id2class[label[i]['category_id']]] < 3:
                        indices_to_keep.append(i)
                elif self.dataset == 'COCO_70_10':
                    if self.contiguous_class2id[self.ori_id2class[label[i]['category_id']]] < 70:
                        indices_to_keep.append(i)
                elif self.dataset == 'COCO_60_20':
                    if self.contiguous_class2id[self.ori_id2class[label[i]['category_id']]] < 60:
                        indices_to_keep.append(i)
                else:
                    indices_to_keep.append(i)
        label = [label[i] for i in indices_to_keep]
        
        return label
    
    def build_initial_buffer(self, buffer_size=None):
        n_classes, images_dir, label_path = get_pretrained_statistics(self.dataset)
        self.image_dir = images_dir
        self.label_path = label_path
        
        if self.dataset == 'VOC_10_10' or self.dataset == 'VOC_15_5':
            image_files = glob.glob(os.path.join(images_dir, "train","*/*.jpg"))
        elif self.dataset == 'VisDrone_3_4':
            image_files = glob.glob(os.path.join(images_dir, "trainval","*/*/*.jpg"))
        elif 'COCO' in self.dataset:
            image_files = glob.glob(os.path.join(images_dir, "train2017","*.jpg"))
        else:
            image_files = glob.glob(os.path.join(images_dir, "train","*.jpg"))
            
        if 'SHIFT' in self.dataset:
            cleaned_files = []
            for i in image_files:
                key = re.sub(r'\.(jpg|jpeg|png)$', '', i, flags=re.IGNORECASE)
                try:
                    temp = self.name2id[key]
                except KeyError:
                    try:
                        temp = self.name2id[key.split('/')[-1]]
                    except KeyError:
                        continue  # skip this file
                cleaned_files.append(i)
            image_files = cleaned_files
                    

        indices = np.random.choice(range(len(image_files)), size=buffer_size or self.memory_size, replace=False)
        for idx in indices:
            image_path = image_files[idx]
            split_name = image_path.split('/')[-2]
            base_name = image_path.split('/')[-1]
            self.register_sample_for_initial_buffer({'file_name': split_name + '/' + base_name, 'label': None})    
    
    def replace_sample(self, sample, info=0, idx=None):
        data_class = sample.get('label', None)
        img, labels, img_id = self.load_data(sample['file_name'], is_stream=True,data_class=data_class)
        data = (img, labels, img_id)

        if idx is None:
            self.buffer.append(data)
        else:
            self.buffer[idx] = data
            
    def register_sample_for_initial_buffer(self, sample, idx=None):
        img, labels, img_id = self.load_data(sample['file_name'], is_stream=False)
        data = (img, labels, img_id)

        if idx is None:
            self.buffer.append(data)
        else:
            self.buffer[idx] = data
            
    def register_stream(self, datalist):
        self.stream_data = []
        for data in datalist:
            data_class = data.get('label', None)
            img_path = data.get('file_name', data.get('filepath'))
            
            img, labels, img_id = self.load_data(img_path, is_stream=True, data_class=data_class)
            
            self.stream_data.append((img, labels, img_id))

    @torch.no_grad()
    def get_batch(self, batch_size, stream_batch_size=0, weight_method=None):
        assert batch_size >= stream_batch_size
        stream_batch_size = min(stream_batch_size, len(self.stream_data))
        batch_size = min(batch_size, stream_batch_size + len(self.buffer))
        memory_batch_size = batch_size - stream_batch_size

        data = []
        
        # append stream data to buffer for batch creation
        buffer_size = len(self.buffer)
        self.buffer.extend(self.stream_data)
 
        if stream_batch_size > 0:
            stream_indices = np.random.choice(range(len(self.stream_data)), size=stream_batch_size, replace=False)
            for i in stream_indices:
                
                
                if self.use_mosaic_mixup:
                    img, label, img_id = self.mosaic_wrapper.__getitem__((True, i + buffer_size))
                else:
                    img, anno, img_id = self.buffer[i + buffer_size]
                    anno = [obj for obj in anno if obj['iscrowd'] == 0]

                    boxes = [obj['bbox'] for obj in anno]
                    boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                    target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

                    classes = [obj['category_id'] for obj in anno]
                    classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                            for c in classes]

                    classes = torch.tensor(classes)
                    target.add_field('labels', classes)


                    target = target.clip_to_image(remove_empty=True)
                    
                    img = np.asarray(img)
                    img, label = self._transforms(img, target)
                
                data.append((img, label, img_id))

        if memory_batch_size > 0:
            indices = np.random.choice(range(buffer_size), size=memory_batch_size, replace=False)
            for i in indices:
                if self.use_mosaic_mixup:
                    img, label, img_id = self.mosaic_wrapper.__getitem__((True, i))
                else:
                    img, label, img_id = self.buffer[i]
                    anno = [obj for obj in anno if obj['iscrowd'] == 0]

                    boxes = [obj['bbox'] for obj in anno]
                    boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                    target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

                    classes = [obj['category_id'] for obj in anno]
                    classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                            for c in classes]

                    classes = torch.tensor(classes)
                    target.add_field('labels', classes)


                    target = target.clip_to_image(remove_empty=True)
                    
                    img = np.asarray(img)
                    img, label = self._transforms(img, target)
                data.append((img, label, img_id))

        # remove stream data from buffer
        self.buffer = self.buffer[:buffer_size]
        
        return self.batch_collator(data)

    def add_new_class(self, cls_list):
        self.cls_list = cls_list
        self.cls_count.append(0)
        self.cls_idx.append([])
        self.class_usage_cnt.append(0)
        self.cls_train_cnt = np.append(self.cls_train_cnt, 0)

    def pull_item(self, idx):
        img, anno, img_id = self.buffer[idx]
        
        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj['bbox'] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')
        
        target = target.clip_to_image(remove_empty=True)

        classes = [obj['category_id'] for obj in anno]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]

        obj_masks = []
        for obj in anno:
            obj_mask = []
            if 'segmentation' in obj:
                for mask in obj['segmentation']:
                    obj_mask += mask
                if len(obj_mask) > 0:
                    obj_masks.append(obj_mask)
        seg_masks = [
            np.array(obj_mask, dtype=np.float32).reshape(-1, 2)
            for obj_mask in obj_masks
        ]

        res = np.zeros((len(target.bbox), 5))
        for idx in range(len(target.bbox)):
            res[idx, 0:4] = target.bbox[idx]
            res[idx, 4] = classes[idx]

        img = np.asarray(img)  # rgb

        return img, res, seg_masks, img_id
    
    def load_anno(self, idx):
        _, anno, _ = self.buffer[idx]
        anno = [obj for obj in anno if obj['iscrowd'] == 0]
        classes = [obj['category_id'] for obj in anno]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]
        return classes

###############################################################################################
# Frequency-based Dataset
class FreqDataset(MemoryDataset):
    def __init__(self, ann_file, root, transforms=None, class_names=None,
                 dataset=None, cls_list=None, device=None, data_dir=None, memory_size=None, 
                 init_buffer_size=None, image_size=(640, 640), aug=None
                 ):
        super().__init__(ann_file, root, transforms, class_names, 
                         dataset, cls_list, device, data_dir, memory_size,
                         init_buffer_size, image_size, aug)
        self.alpha = 1.0                  # smoothing constant in 1/(usage+α)
        self.beta = 1.0
        self.usage_decay = 0.995
        self.use_mosaic_mixup=False

    def replace_sample(self, sample, idx=None):
        data_class = sample.get('label', None)
        img, labels, img_id = self.load_data(sample['file_name'], is_stream=True,data_class=data_class)
        ### BEGIN USAGE
        classes = [obj['category_id'] for obj in labels]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]
        entry = {
            "img": img,
            "labels": labels,
            "img_id": img_id,
            "usage": sample.get("usage", 0),
            "classes": list(set(classes)) if len(classes) else []
        }
        ### END USAGE

        if idx is None:
            self.buffer.append(entry)
        else:
            self.buffer[idx] = entry
    
    def register_sample_for_initial_buffer(self, sample, idx=None):
        img, labels, img_id = self.load_data(sample['file_name'], is_stream=False)
        ### BEGIN USAGE
        classes = [obj['category_id'] for obj in labels]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]
        entry = {
            "img": img,
            "labels": labels,
            "img_id": img_id,
            "usage": sample.get("usage", 0),
            "classes": list(set(classes)) if len(classes) else []
        }
        ### END USAGE

        if idx is None:
            self.buffer.append(entry)
        else:
            self.buffer[idx] = entry
    
    def register_stream(self, datalist):
        self.stream_data = []
        for data in datalist:
            data_class = data.get('label', None)
            img_path = data.get('file_name', data.get('filepath'))
            
            img, labels, img_id = self.load_data(img_path, is_stream=True, data_class=data_class)
            classes = [obj['category_id'] for obj in labels]
            classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                    for c in classes]
            entry = {
                "img": img,
                "labels": labels,
                "img_id": img_id,
                "usage": 0,
                "classes": list(set(classes)) if len(classes) else []
            }
            
            self.stream_data.append(entry)
    
    @torch.no_grad()
    def get_batch(self, batch_size, stream_batch_size=0, weight_method= None):
        assert batch_size >= stream_batch_size
        stream_batch_size = min(stream_batch_size, len(self.stream_data))
        batch_size = min(batch_size, stream_batch_size + len(self.buffer))
        memory_batch_size = batch_size - stream_batch_size

        data = []
        
        # append stream data to buffer for batch creation
        buffer_size = len(self.buffer)
        self.buffer.extend(self.stream_data)
 
        if stream_batch_size > 0:
            stream_indices = np.random.choice(range(len(self.stream_data)), size=stream_batch_size, replace=False)
            for i in stream_indices:
                if self.use_mosaic_mixup:
                    img, label, img_id = self.mosaic_wrapper.__getitem__((True, i + buffer_size))
                else:
                    entry = self.buffer[i + buffer_size]
                    img, anno, img_id = entry['img'], entry['labels'], entry['img_id']
                    anno = [obj for obj in anno if obj['iscrowd'] == 0]

                    boxes = [obj['bbox'] for obj in anno]
                    boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                    target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

                    classes = [obj['category_id'] for obj in anno]
                    classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                            for c in classes]

                    classes = torch.tensor(classes)
                    target.add_field('labels', classes)

                    target = target.clip_to_image(remove_empty=True)
                    
                    img = np.asarray(img)
                    img, label = self._transforms(img, target)
                
                data.append((img, label, img_id))

        # ───── Memory part ──────────────────────────────────────────────
        if memory_batch_size > 0 and len(self.buffer):
            for e in self.buffer:
                e['usage'] *= self.usage_decay
            for cls_idx in range(len(self.cls_train_cnt)):
                self.cls_train_cnt[cls_idx] *= self.usage_decay

            ### HYBRID WEIGHT BEGIN
            if weight_method == "cls_usage":          # new option
                # 1 / (usage+α)  ×  1 / (mean cls_trained + β)
                alpha = getattr(self, "alpha", 1.0)
                beta  = getattr(self, "beta", 1.0)       # you may set self.beta in __init__
                weights = []
                for entry in self.buffer:
                    u = entry["usage"]
                    # gather per-image class-trained counts
                    if entry["classes"]:
                        t = [self.cls_train_cnt[int(c)]    # safe: cls_dict maps real IDs
                            for c in entry["classes"]               # (skip if not yet in dict)
                            if c < len(self.cls_train_cnt)]
                        mean_t = np.mean(t) if t else 0.0
                    else:
                        mean_t = 0.0        # no GT boxes → neutral
                    weights.append(1.0 / (u + alpha) * 1.0 / (mean_t + beta))
                w = np.asarray(weights, dtype=np.float64)
                w /= w.sum()
            else:
                w = np.array([1.0 / (e["usage"] + self.alpha) for e in self.buffer],
                            dtype=np.float64)
                w /= w.sum()
            ### HYBRID WEIGHT END

            indices = np.random.choice(
                len(self.buffer),
                size=memory_batch_size,
                replace=len(self.buffer) < memory_batch_size,
                p=w,
            )

            for i in indices:
                # update usage counter *and* class-train counts
                self.buffer[i]["usage"] += 1
                for idx_cls in self.buffer[i]["classes"]:
                    if idx_cls < len(self.cls_train_cnt):
                        self.cls_train_cnt[int(idx_cls)] += 1

                if self.use_mosaic_mixup:
                    img, label, img_id = self.mosaic_wrapper.__getitem__((True, i))
                else:
                    entry = self.buffer[i]
                    img, anno, img_id = entry['img'], entry['labels'], entry['img_id']
                    anno = [obj for obj in anno if obj['iscrowd'] == 0]

                    boxes = [obj['bbox'] for obj in anno]
                    boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                    target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

                    classes = [obj['category_id'] for obj in anno]
                    classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                            for c in classes]

                    classes = torch.tensor(classes)
                    target.add_field('labels', classes)


                    target = target.clip_to_image(remove_empty=True)
                    
                    img = np.asarray(img)
                    img, label = self._transforms(img, target)
                data.append((img, label, img_id))

         # remove stream data from buffer
        self.buffer = self.buffer[:buffer_size]
        
        return self.batch_collator(data)

    def pull_item(self, idx):
        entry = self.buffer[idx]
        img, anno, img_id = entry['img'], entry['labels'], entry['img_id']
        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj['bbox'] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')
        
        target = target.clip_to_image(remove_empty=True)

        classes = [obj['category_id'] for obj in anno]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]

        obj_masks = []
        for obj in anno:
            obj_mask = []
            if 'segmentation' in obj:
                for mask in obj['segmentation']:
                    obj_mask += mask
                if len(obj_mask) > 0:
                    obj_masks.append(obj_mask)
        seg_masks = [
            np.array(obj_mask, dtype=np.float32).reshape(-1, 2)
            for obj_mask in obj_masks
        ]

        res = np.zeros((len(target.bbox), 5))
        for idx in range(len(target.bbox)):
            res[idx, 0:4] = target.bbox[idx]
            res[idx, 4] = classes[idx]

        img = np.asarray(img)  # rgb

        return img, res, seg_masks, img_id
    
    def load_anno(self, idx):
        entry = self.buffer[idx]
        anno = entry['labels']
        anno = [obj for obj in anno if obj['iscrowd'] == 0]
        classes = [obj['category_id'] for obj in anno]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]
        return classes

################################################################################################
# Class Balanced Dataset
class ClassBalancedDataset(MemoryDataset):
    def __init__(self, ann_file, root, transforms=None, class_names=None,
                 dataset=None, cls_list=None, device=None, data_dir=None, memory_size=None, 
                 init_buffer_size=None, image_size=(640, 640), aug=None
                 ):
        super(MemoryDataset, self).__init__(ann_file, root, transforms, class_names)
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_sizes = [int(image_size[0]), int(image_size[1])]
        self.memory_size = memory_size

        self.buffer = []
        self.stream_data = []

        self.dataset = dataset
        self.device = device
        self.data_dir = data_dir

        self.class_usage_cnt = []
        self.cls_list = cls_list if cls_list else []
        
        self.is_domain_incremental = ('SHIFT' in self.dataset) or ('BDD' in self.dataset) or ('Military' in self.dataset)
        
        if self.is_domain_incremental:
            self.cls_count = [0]
            self.cls_idx = [[]]
            
            self.new_exposed_classes = ['pretrained'] 
        else:
            self.cls_count = [0 for _ in range(len(cls_list))]
            self.cls_idx = [[] for _ in range(len(cls_list))]
            
            self.new_exposed_classes = self.cls_list #['pretrained'] 
        self.cls_train_cnt = np.array([0,]*len(cls_list)) if cls_list else np.array([])
        
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
            self.name2id = {re.sub(r'\.(jpg|jpeg|png)$', '', ann['file_name'], flags=re.I): ann['id'] for ann in annotations['images']}
            #print("[DEBUG] data_loader file:", __file__)
            #print("[DEBUG] ann_file:", ann_file)
            #print("[DEBUG] sample name2id key:", next(iter(self.name2id.keys())))

        
        if 'HS_TOD' not in self.dataset:
            self.build_initial_buffer(init_buffer_size)

        n_classes, image_dir, label_path = get_statistics(dataset=self.dataset)
        self.image_dir = image_dir
        self.label_path = label_path
        
        # set transform
        transforms = aug.transform
        transforms = build_transforms_memorydataset(**transforms)
        self._transforms = transforms
        
        print("mosaic_prob", aug.mosaic_mixup.mosaic_prob)
        print("mixup_prob", aug.mosaic_mixup.mixup_prob)
        print("degrees", aug.mosaic_mixup.degrees)
        print("translate", aug.mosaic_mixup.translate)
        print("mosaic_scale", aug.mosaic_mixup.mosaic_scale)
        
        
        self.mosaic_wrapper = MosaicWrapper(
            dataset=self,                      # <-- this class will provide pull_item/load_anno
            img_size=image_size,
            mosaic_prob=aug.mosaic_mixup.mosaic_prob,                   # always “allowed”; actual apply is still random inside wrapper
            mixup_prob=aug.mosaic_mixup.mixup_prob,
            transforms=self._transforms,                   # we'll run your own transforms after mosaic
            degrees=aug.mosaic_mixup.degrees,
            translate=aug.mosaic_mixup.translate,
            mosaic_scale=aug.mosaic_mixup.mosaic_scale,
            keep_ratio=True,
        )
        self.use_mosaic_mixup=True
        
        self.batch_collator = BatchCollator(size_divisible=32)
    
    def build_initial_buffer(self, buffer_size=None):
        n_classes, images_dir, label_path = get_pretrained_statistics(self.dataset)
        self.image_dir = images_dir
        self.label_path = label_path
        
        if self.dataset == 'VOC_10_10' or self.dataset == 'VOC_15_5':
            image_files = glob.glob(os.path.join(images_dir, "train","*/*.jpg"))
        elif self.dataset == 'VisDrone_3_4':
            image_files = glob.glob(os.path.join(images_dir, "trainval","*/*/*.jpg"))
        elif 'COCO' in self.dataset:
            image_files = glob.glob(os.path.join(images_dir, "train2017","*.jpg"))
        else:
            image_files = glob.glob(os.path.join(images_dir, "train","*.jpg"))
            
        if 'SHIFT' in self.dataset:
            cleaned_files = []
            for i in image_files:
                key = re.sub(r'\.(jpg|jpeg|png)$', '', i, flags=re.IGNORECASE)
                try:
                    temp = self.name2id[key]
                except KeyError:
                    try:
                        temp = self.name2id[key.split('/')[-1]]
                    except KeyError:
                        continue  # skip this file
                cleaned_files.append(i)
            image_files = cleaned_files
                    
        if self.is_domain_incremental:
            indices = np.random.choice(range(len(image_files)), size=buffer_size or self.memory_size, replace=False)
            for idx in indices:
                image_path = image_files[idx]
                split_name = image_path.split('/')[-2]
                base_name = image_path.split('/')[-1]
                self.register_sample_for_initial_buffer({'file_name': split_name + '/' + base_name, 'label': None})    
        else:
            for idx in range(len(image_files)):
                image_path = image_files[idx]
                split_name = image_path.split('/')[-2]
                base_name = image_path.split('/')[-1]
                if (buffer_size is not None and len(self.buffer) >= buffer_size) or (buffer_size is None and len(self.buffer) >= self.memory_size):
                    cls_to_replace = np.random.choice(
                        np.flatnonzero(np.array(self.cls_count) == np.array(self.cls_count).max()))
                    mem_index = np.random.choice(self.cls_idx[cls_to_replace])
                    labels = self.buffer[mem_index][1]
                else:
                    mem_index = None
                self.register_sample_for_initial_buffer({'file_name': split_name + '/' + base_name, 'label': None}, idx=mem_index)    
                if mem_index is not None:
                    classes = [obj['category_id'] for obj in labels]
                    classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                            for c in classes]
                    classes = list(set(classes))
                    for cls_ in classes:
                        self.cls_count[cls_] -= 1
                        self.cls_idx[cls_].remove(mem_index)
                    
                    new_labels = self.buffer[mem_index][1]
                    new_classes = [obj['category_id'] for obj in new_labels]
                    new_classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                            for c in new_classes]
                    new_classes = list(set(new_classes))
                    for cls_ in new_classes:
                        self.cls_idx[cls_].append(mem_index)
    
    def replace_sample(self, sample, idx=None):
        data_class = sample.get('label', None)
        img, labels, img_id = self.load_data(sample['file_name'], is_stream=True,data_class=data_class)
        data = (img, labels, img_id)

        if sample.get('klass', None):
            # self.cls_count[self.new_exposed_classes.index(sample['klass'])] += 1
            # sample_category = sample['klass']
            classes = [obj['category_id'] for obj in labels]
            classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                    for c in classes]
            classes = list(set(classes))
            for cls_ in classes:
                self.cls_count[cls_] += 1
            sample_category = classes
        elif sample.get('domain', None):
            self.cls_count[self.new_exposed_classes.index(sample['domain'])] += 1
            sample_category = sample['domain']
        else:
            self.cls_count[self.new_exposed_classes.index('pretrained')] += 1
            sample_category = 'pretrained'
        # self.cls_count[self.new_exposed_classes.index(sample.get('klass', 'pretrained'))] += 1
        if idx is None:
            if isinstance(sample_category, list):
                for cls_ in sample_category:
                    self.cls_idx[cls_].append(len(self.buffer))
            else:
                self.cls_idx[self.new_exposed_classes.index(sample_category)].append(len(self.buffer))
            self.buffer.append(data)
        else:
            self.buffer[idx] = data
    
    def register_sample_for_initial_buffer(self, sample, idx=None):
        data_class = sample.get('label', None)
        img, labels, img_id = self.load_data(sample['file_name'], is_stream=False)
        data = (img, labels, img_id)
        if not self.is_domain_incremental:
            # self.cls_count[self.new_exposed_classes.index(sample['klass'])] += 1
            # sample_category = sample['klass']
            classes = [obj['category_id'] for obj in labels]
            classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                    for c in classes]
            classes = list(set(classes))
            for cls_ in classes:
                self.cls_count[cls_] += 1
            sample_category = classes
        else:
            self.cls_count[self.new_exposed_classes.index('pretrained')] += 1
            sample_category = 'pretrained'
        # self.cls_count[self.new_exposed_classes.index(sample.get('klass', 'pretrained'))] += 1
        if idx is None:
            if isinstance(sample_category, list):
                for cls_ in sample_category:
                    self.cls_idx[cls_].append(len(self.buffer))
            else:
                self.cls_idx[self.new_exposed_classes.index(sample_category)].append(len(self.buffer))
            self.buffer.append(data)
        else:
            self.buffer[idx] = data

#################################################################################################
# Frequency + class balanced dataset
class FreqClsBalancedDataset(MemoryDataset):
    def __init__(self, ann_file, root, transforms=None, class_names=None,
                 dataset=None, cls_list=None, device=None, data_dir=None, memory_size=None, 
                 init_buffer_size=None, image_size=(640, 640), aug=None
                 ):
        super(MemoryDataset, self).__init__(ann_file, root, transforms, class_names)
        # self.args = args
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_sizes = [int(image_size[0]), int(image_size[1])]
        self.memory_size = memory_size

        self.buffer = []
        self.stream_data = []
        
        self.dataset = dataset
        self.device = device
        self.data_dir = data_dir

        self.class_usage_cnt = []
        self.cls_list = cls_list if cls_list else []
        
        self.is_domain_incremental = ('SHIFT' in self.dataset) or ('BDD' in self.dataset) or ('Military' in self.dataset)
        
        if self.is_domain_incremental:
            self.cls_count = [0]
            self.cls_idx = [[]]
            
            self.new_exposed_classes = ['pretrained'] 
        else:
            self.cls_count = [0 for _ in range(len(cls_list))]
            self.cls_idx = [[] for _ in range(len(cls_list))]
            
            self.new_exposed_classes = self.cls_list #['pretrained'] 
        self.cls_train_cnt = np.array([0,]*len(cls_list)) if cls_list else np.array([])
        
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
            self.name2id = {re.sub(r'\.(jpg|jpeg|png)$', '', ann['file_name'], flags=re.I): ann['id'] for ann in annotations['images']}
        
        if 'HS_TOD' not in self.dataset:
            self.build_initial_buffer(init_buffer_size)

        n_classes, image_dir, label_path = get_statistics(dataset=self.dataset)
        self.image_dir = image_dir
        self.label_path = label_path
        
        # set transform
        transforms = aug.transform
        transforms = build_transforms_memorydataset(**transforms)
        self._transforms = transforms
        
        print("mosaic_prob", aug.mosaic_mixup.mosaic_prob)
        print("mixup_prob", aug.mosaic_mixup.mixup_prob)
        print("degrees", aug.mosaic_mixup.degrees)
        print("translate", aug.mosaic_mixup.translate)
        print("mosaic_scale", aug.mosaic_mixup.mosaic_scale)
        
        
        self.mosaic_wrapper = MosaicWrapper(
            dataset=self,                      # <-- this class will provide pull_item/load_anno
            img_size=image_size,
            mosaic_prob=aug.mosaic_mixup.mosaic_prob,                   # always “allowed”; actual apply is still random inside wrapper
            mixup_prob=aug.mosaic_mixup.mixup_prob,
            transforms=self._transforms,                   # we'll run your own transforms after mosaic
            degrees=aug.mosaic_mixup.degrees,
            translate=aug.mosaic_mixup.translate,
            mosaic_scale=aug.mosaic_mixup.mosaic_scale,
            keep_ratio=True,
        )
        self.use_mosaic_mixup=False 
        
        self.batch_collator = BatchCollator(size_divisible=32)
        
        self.alpha = 1.0                  # smoothing constant in 1/(usage+α)
        self.beta = 1.0
        self.usage_decay = 0.995
    
    def build_initial_buffer(self, buffer_size=None):
        n_classes, images_dir, label_path = get_pretrained_statistics(self.dataset)
        self.image_dir = images_dir
        self.label_path = label_path
        
        if self.dataset == 'VOC_10_10' or self.dataset == 'VOC_15_5':
            image_files = glob.glob(os.path.join(images_dir, "train","*/*.jpg"))
        elif self.dataset == 'VisDrone_3_4':
            image_files = glob.glob(os.path.join(images_dir, "trainval","*/*/*.jpg"))
        elif 'COCO' in self.dataset:
            image_files = glob.glob(os.path.join(images_dir, "train2017","*.jpg"))
        else:
            image_files = glob.glob(os.path.join(images_dir, "train","*.jpg"))
            
        if 'SHIFT' in self.dataset:
            cleaned_files = []
            for i in image_files:
                key = re.sub(r'\.(jpg|jpeg|png)$', '', i, flags=re.IGNORECASE)
                try:
                    temp = self.name2id[key]
                except KeyError:
                    try:
                        temp = self.name2id[key.split('/')[-1]]
                    except KeyError:
                        continue  # skip this file
                cleaned_files.append(i)
            image_files = cleaned_files
                    
        if self.is_domain_incremental:
            indices = np.random.choice(range(len(image_files)), size=buffer_size or self.memory_size, replace=False)
            for idx in indices:
                image_path = image_files[idx]
                split_name = image_path.split('/')[-2]
                base_name = image_path.split('/')[-1]
                self.register_sample_for_initial_buffer({'file_name': split_name + '/' + base_name, 'label': None})    
        else:
            for idx in range(len(image_files)):
                image_path = image_files[idx]
                split_name = image_path.split('/')[-2]
                base_name = image_path.split('/')[-1]
                if (buffer_size is not None and len(self.buffer) >= buffer_size) or (buffer_size is None and len(self.buffer) >= self.memory_size):
                    cls_to_replace = np.random.choice(
                        np.flatnonzero(np.array(self.cls_count) == np.array(self.cls_count).max()))
                    mem_index = np.random.choice(self.cls_idx[cls_to_replace])
                    labels = self.buffer[mem_index]['labels']
                else:
                    mem_index = None
                self.register_sample_for_initial_buffer({'file_name': split_name + '/' + base_name, 'label': None}, idx=mem_index)    
                if mem_index is not None:
                    classes = [obj['category_id'] for obj in labels]
                    classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                            for c in classes]
                    classes = list(set(classes))
                    for cls_ in classes:
                        self.cls_count[cls_] -= 1
                        self.cls_idx[cls_].remove(mem_index)
                    
                    new_labels = self.buffer[mem_index]['labels']
                    new_classes = [obj['category_id'] for obj in new_labels]
                    new_classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                            for c in new_classes]
                    new_classes = list(set(new_classes))
                    for cls_ in new_classes:
                        self.cls_idx[cls_].append(mem_index)
        
    def replace_sample(self, sample, idx=None):
        data_class = sample.get('label', None)
        img, labels, img_id = self.load_data(sample['file_name'], is_stream=True,data_class=data_class)
        classes = [obj['category_id'] for obj in labels]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]
        ### BEGIN USAGE
        entry = {
            "img": img,
            "labels": labels,
            "img_id": img_id,
            "usage": sample.get("usage", 0),
            "classes": list(set(classes)) if len(classes) else []
        }
        ### END USAGE

        if sample.get('klass', None):
            # self.cls_count[self.new_exposed_classes.index(sample['klass'])] += 1
            # sample_category = sample['klass']
            classes = [obj['category_id'] for obj in labels]
            classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                    for c in classes]
            classes = list(set(classes))
            for cls_ in classes:
                self.cls_count[cls_] += 1
            sample_category = classes
        elif sample.get('domain', None):
            self.cls_count[self.new_exposed_classes.index(sample['domain'])] += 1
            sample_category = sample['domain']
        else:
            self.cls_count[self.new_exposed_classes.index('pretrained')] += 1
            sample_category = 'pretrained'
        # self.cls_count[self.new_exposed_classes.index(sample.get('klass', 'pretrained'))] += 1
        if idx is None:
            if isinstance(sample_category, list):
                for cls_ in sample_category:
                    self.cls_idx[cls_].append(len(self.buffer))
            else:
                self.cls_idx[self.new_exposed_classes.index(sample_category)].append(len(self.buffer))
            self.buffer.append(entry)
        else:
            self.buffer[idx] = entry
    
    def register_sample_for_initial_buffer(self, sample, idx=None):
        img, labels, img_id = self.load_data(sample['file_name'], is_stream=False)
        ### BEGIN USAGE
        classes = [obj['category_id'] for obj in labels]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]
        entry = {
            "img": img,
            "labels": labels,
            "img_id": img_id,
            "usage": sample.get("usage", 0),
            "classes": list(set(classes)) if len(classes) else []
        }
        ### END USAGE

        if not self.is_domain_incremental:
            # self.cls_count[self.new_exposed_classes.index(sample['klass'])] += 1
            # sample_category = sample['klass']
            classes = [obj['category_id'] for obj in labels]
            classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                    for c in classes]
            classes = list(set(classes))
            for cls_ in classes:
                self.cls_count[cls_] += 1
            sample_category = classes
        else:
            self.cls_count[self.new_exposed_classes.index('pretrained')] += 1
            sample_category = 'pretrained'
        # self.cls_count[self.new_exposed_classes.index(sample.get('klass', 'pretrained'))] += 1
        if idx is None:
            if isinstance(sample_category, list):
                for cls_ in sample_category:
                    self.cls_idx[cls_].append(len(self.buffer))
            else:
                self.cls_idx[self.new_exposed_classes.index(sample_category)].append(len(self.buffer))
            self.buffer.append(entry)
        else:
            self.buffer[idx] = entry
        
    
    def register_stream(self, datalist):
        self.stream_data = []
        for data in datalist:
            data_class = data.get('label', None)
            img_path = data.get('file_name', data.get('filepath'))
            
            img, labels, img_id = self.load_data(img_path, is_stream=True, data_class=data_class)
            classes = [obj['category_id'] for obj in labels]
            classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                    for c in classes]
            entry = {
                "img": img,
                "labels": labels,
                "img_id": img_id,
                "usage": 0,
                "classes": list(set(classes)) if len(classes) else []
            }
            
            self.stream_data.append(entry)

    @torch.no_grad()
    def get_batch(self, batch_size, stream_batch_size=0, weight_method= None):
        assert batch_size >= stream_batch_size
        stream_batch_size = min(stream_batch_size, len(self.stream_data))
        batch_size = min(batch_size, stream_batch_size + len(self.buffer))
        memory_batch_size = batch_size - stream_batch_size

        data = []
        
        # append stream data to buffer for batch creation
        buffer_size = len(self.buffer)
        self.buffer.extend(self.stream_data)
 
        if stream_batch_size > 0:
            stream_indices = np.random.choice(range(len(self.stream_data)), size=stream_batch_size, replace=False)
            for i in stream_indices:
                if self.use_mosaic_mixup:
                    img, label, img_id = self.mosaic_wrapper.__getitem__((True, i + buffer_size))
                else:
                    entry = self.buffer[i + buffer_size]
                    img, anno, img_id = entry['img'], entry['labels'], entry['img_id']
                    anno = [obj for obj in anno if obj['iscrowd'] == 0]

                    boxes = [obj['bbox'] for obj in anno]
                    boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                    target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

                    classes = [obj['category_id'] for obj in anno]
                    classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                            for c in classes]

                    classes = torch.tensor(classes)
                    target.add_field('labels', classes)

                    target = target.clip_to_image(remove_empty=True)
                    
                    img = np.asarray(img)
                    img, label = self._transforms(img, target)
                
                data.append((img, label, img_id))

        # ───── Memory part ──────────────────────────────────────────────
        if memory_batch_size > 0 and len(self.buffer):
            for e in self.buffer:
                e['usage'] *= self.usage_decay
            for cls_idx in range(len(self.cls_train_cnt)):
                self.cls_train_cnt[cls_idx] *= self.usage_decay

            ### HYBRID WEIGHT BEGIN
            if weight_method == "cls_usage":          # new option
                # 1 / (usage+α)  ×  1 / (mean cls_trained + β)
                alpha = getattr(self, "alpha", 1.0)
                beta  = getattr(self, "beta", 1.0)       # you may set self.beta in __init__
                weights = []
                for entry in self.buffer:
                    u = entry["usage"]
                    # gather per-image class-trained counts
                    if entry["classes"]:
                        t = [self.cls_train_cnt[int(c)]    # safe: cls_dict maps real IDs
                            for c in entry["classes"]               # (skip if not yet in dict)
                            if c < len(self.cls_train_cnt)]
                        mean_t = np.mean(t) if t else 0.0
                    else:
                        mean_t = 0.0        # no GT boxes → neutral
                    weights.append(1.0 / (u + alpha) * 1.0 / (mean_t + beta))
                w = np.asarray(weights, dtype=np.float64)
                w /= w.sum()
            else:
                w = np.array([1.0 / (e["usage"] + self.alpha) for e in self.buffer],
                            dtype=np.float64)
                w /= w.sum()
            ### HYBRID WEIGHT END

            indices = np.random.choice(
                len(self.buffer),
                size=memory_batch_size,
                replace=len(self.buffer) < memory_batch_size,
                p=w,
            )

            for i in indices:
                # update usage counter *and* class-train counts
                self.buffer[i]["usage"] += 1
                for idx_cls in self.buffer[i]["classes"]:
                    if idx_cls < len(self.cls_train_cnt):
                        self.cls_train_cnt[int(idx_cls)] += 1

                if self.use_mosaic_mixup:
                    img, label, img_id = self.mosaic_wrapper.__getitem__((True, i))
                else:
                    entry = self.buffer[i]
                    img, anno, img_id = entry['img'], entry['labels'], entry['img_id']
                    anno = [obj for obj in anno if obj['iscrowd'] == 0]

                    boxes = [obj['bbox'] for obj in anno]
                    boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                    target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

                    classes = [obj['category_id'] for obj in anno]
                    classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                            for c in classes]

                    classes = torch.tensor(classes)
                    target.add_field('labels', classes)


                    target = target.clip_to_image(remove_empty=True)
                    
                    img = np.asarray(img)
                    img, label = self._transforms(img, target)
                data.append((img, label, img_id))

         # remove stream data from buffer
        self.buffer = self.buffer[:buffer_size]
        
        return self.batch_collator(data)

    def pull_item(self, idx):
        entry = self.buffer[idx]
        img, anno, img_id = entry['img'], entry['labels'], entry['img_id']
        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj['bbox'] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')
        
        target = target.clip_to_image(remove_empty=True)

        classes = [obj['category_id'] for obj in anno]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]

        obj_masks = []
        for obj in anno:
            obj_mask = []
            if 'segmentation' in obj:
                for mask in obj['segmentation']:
                    obj_mask += mask
                if len(obj_mask) > 0:
                    obj_masks.append(obj_mask)
        seg_masks = [
            np.array(obj_mask, dtype=np.float32).reshape(-1, 2)
            for obj_mask in obj_masks
        ]

        res = np.zeros((len(target.bbox), 5))
        for idx in range(len(target.bbox)):
            res[idx, 0:4] = target.bbox[idx]
            res[idx, 4] = classes[idx]

        img = np.asarray(img)  # rgb

        return img, res, seg_masks, img_id
    
    def load_anno(self, idx):
        entry = self.buffer[idx]
        anno = entry['labels']
        anno = [obj for obj in anno if obj['iscrowd'] == 0]
        classes = [obj['category_id'] for obj in anno]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]
        return classes


# Selection-Based Retrieval
class SelectionMemoryDataset(MemoryDataset):
    def __init__(self, ann_file, root, transforms=None, class_names=None,
                 dataset=None, cls_list=None, device=None, data_dir=None, memory_size=None, 
                 init_buffer_size=None, image_size=(640, 640), aug=None, selection_method=None, priority_selection=None
                 ):
        super().__init__(ann_file, root, transforms, class_names,
            dataset,cls_list,device,data_dir,memory_size,
            init_buffer_size,image_size,aug)
        
        self.selection_method = selection_method
        self.priority_selection = priority_selection
        self.alpha = 1.0                  
        self.beta = 1.0
        self.use_mosaic_mixup=False
        self.info_ema = 0.9
    
    def update_initialinfo(self, info_list):
        assert len(info_list) == len(self.buffer)
        for i, info in enumerate(info_list):
            img, labels, img_id, _ = self.buffer[i]
            self.buffer[i] = (img, labels, img_id, info)
            
    def update_info(self, infos):
        assert len(infos) == len(self.indices)
        for i, info in enumerate(infos):
            img, labels, img_id, old_info = self.buffer[self.indices[i]]
            new_info = self.info_ema * old_info + (1 - self.info_ema) * info
            self.buffer[self.indices[i]] = (img, labels, img_id, new_info)
    
    def replace_sample(self, sample, info=0, idx=None):
        data_class = sample.get('label', None)
        img, labels, img_id = self.load_data(sample['file_name'], is_stream=True,data_class=data_class)
        data = (img, labels, img_id, info)

        if idx is None:
            self.buffer.append(data)
        else:
            self.buffer[idx] = data
    
    def register_sample_for_initial_buffer(self, sample, idx=None):
        img, labels, img_id = self.load_data(sample['file_name'], is_stream=False)
        data = (img, labels, img_id, 0)

        if idx is None:
            self.buffer.append(data)
        else:
            self.buffer[idx] = data
    
    def register_stream(self, datalist):
        self.stream_data = []
        for data in datalist:
            data_class = data.get('label', None)
            img_path = data.get('file_name', data.get('filepath'))
            
            img, labels, img_id = self.load_data(img_path, is_stream=True, data_class=data_class)
            
            self.stream_data.append((img, labels, img_id, 0))

    @torch.no_grad()
    def get_buffer_data(self, ind, batch_size):
        data = []
        
        # batch = self.buffer[ind:ind+batch_size]
        
        for i in range(ind, min(ind+batch_size, len(self.buffer))):
            # img, bboxes, img_id = entry['img'], entry['labels'], entry['img_id']
            # valid_mask = labels[:, 0] != -1
            # bboxes = labels[valid_mask]
            if self.use_mosaic_mixup:
                img, label, img_id = self.mosaic_wrapper.__getitem__((True, i))
            else:
                img, anno, img_id, score = self.buffer[i]
                anno = [obj for obj in anno if obj['iscrowd'] == 0]

                boxes = [obj['bbox'] for obj in anno]
                boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

                classes = [obj['category_id'] for obj in anno]
                classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                        for c in classes]

                classes = torch.tensor(classes)
                target.add_field('labels', classes)


                target = target.clip_to_image(remove_empty=True)
                
                img = np.asarray(img)
                img, label = self._transforms(img, target)
            data.append((img, label, img_id))
            
        return self.batch_collator(data)
    
    @torch.no_grad()
    def get_batch(self, batch_size, stream_batch_size=0, weight_method=None):
        assert batch_size >= stream_batch_size
        stream_batch_size = min(stream_batch_size, len(self.stream_data))
        batch_size = min(batch_size, stream_batch_size + len(self.buffer))
        memory_batch_size = batch_size - stream_batch_size

        data = []
        
        # append stream data to buffer for batch creation
        buffer_size = len(self.buffer)
        self.buffer.extend(self.stream_data)
 
        if stream_batch_size > 0:
            stream_indices = np.random.choice(range(len(self.stream_data)), size=stream_batch_size, replace=False)
            for i in stream_indices:
                
                
                if self.use_mosaic_mixup:
                    img, label, img_id = self.mosaic_wrapper.__getitem__((True, i + buffer_size))
                else:
                    img, anno, img_id, score = self.buffer[i + buffer_size]
                    anno = [obj for obj in anno if obj['iscrowd'] == 0]

                    boxes = [obj['bbox'] for obj in anno]
                    boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                    target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

                    classes = [obj['category_id'] for obj in anno]
                    classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                            for c in classes]

                    classes = torch.tensor(classes)
                    target.add_field('labels', classes)


                    target = target.clip_to_image(remove_empty=True)
                    
                    img = np.asarray(img)
                    img, label = self._transforms(img, target)
                
                data.append((img, label, img_id))

        if memory_batch_size > 0:
            # indices = np.random.choice(range(buffer_size), size=memory_batch_size, replace=False)
            info_list = np.array([e[-1] for e in self.buffer],
                        dtype=np.float64)
            info_list = info_list[:buffer_size]
            w = info_list / info_list.sum()
            ### HYBRID WEIGHT END
            if self.priority_selection == "high":
                indices = np.argpartition(w, -memory_batch_size)[-memory_batch_size:]
            elif self.priority_selection == "low":
                indices = np.argpartition(w, memory_batch_size)[:memory_batch_size] 
            elif self.priority_selection == "prob":
                indices = np.random.choice(
                    len(info_list),
                    size=memory_batch_size,
                    replace=len(info_list) < memory_batch_size,
                    p=w,
                )
            # print(nonzero_indices, indices, info_list, len(info_list), len(self.buffer))
            self.indices = indices
            for i in indices:
                if self.use_mosaic_mixup:
                    img, label, img_id = self.mosaic_wrapper.__getitem__((True, i))
                else:
                    img, anno, img_id, score = self.buffer[i]
                    anno = [obj for obj in anno if obj['iscrowd'] == 0]
                    

                    boxes = [obj['bbox'] for obj in anno]
                    boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                    target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

                    classes = [obj['category_id'] for obj in anno]
                    classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                            for c in classes]

                    classes = torch.tensor(classes)
                    target.add_field('labels', classes)


                    target = target.clip_to_image(remove_empty=True)
                    
                    img = np.asarray(img)
                    img, label = self._transforms(img, target)
                
                data.append((img, label, img_id))

        # remove stream data from buffer
        self.buffer = self.buffer[:buffer_size]
        
        return self.batch_collator(data)


    def pull_item(self, idx):
        img, anno, img_id, score = self.buffer[idx]
        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj['bbox'] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')
        
        target = target.clip_to_image(remove_empty=True)

        classes = [obj['category_id'] for obj in anno]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]

        obj_masks = []
        for obj in anno:
            obj_mask = []
            if 'segmentation' in obj:
                for mask in obj['segmentation']:
                    obj_mask += mask
                if len(obj_mask) > 0:
                    obj_masks.append(obj_mask)
        seg_masks = [
            np.array(obj_mask, dtype=np.float32).reshape(-1, 2)
            for obj_mask in obj_masks
        ]

        res = np.zeros((len(target.bbox), 5))
        for idx in range(len(target.bbox)):
            res[idx, 0:4] = target.bbox[idx]
            res[idx, 4] = classes[idx]

        img = np.asarray(img)  # rgb

        return img, res, seg_masks, img_id
    
    def load_anno(self, idx):
        _, anno, _,_ = self.buffer[idx]
        anno = [obj for obj in anno if obj['iscrowd'] == 0]
        classes = [obj['category_id'] for obj in anno]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]
        return classes
    
    
# Selection-Based Retrieval + Cls Balanced
class SelectionClsBalancedDataset(ClassBalancedDataset):
    def __init__(self, ann_file, root, transforms=None, class_names=None,
                 dataset=None, cls_list=None, device=None, data_dir=None, memory_size=None, 
                 init_buffer_size=None, image_size=(640, 640), aug=None, selection_method=None, priority_selection=None
                 ):
        super().__init__(ann_file, root, transforms, class_names,
            dataset,cls_list,device,data_dir,memory_size,
            init_buffer_size,image_size,aug)
        
        self.selection_method = selection_method
        self.priority_selection = priority_selection
        self.alpha = 1.0                  
        self.beta = 1.0
        self.use_mosaic_mixup=False
        self.info_ema = 0.9
        
    def update_initialinfo(self, info_list):
        assert len(info_list) == len(self.buffer)
        for i, info in enumerate(info_list):
            img, labels, img_id, _ = self.buffer[i]
            self.buffer[i] = (img, labels, img_id, info)
            
    def update_info(self, infos):
        assert len(infos) == len(self.indices)
        for i, info in enumerate(infos):
            img, labels, img_id, old_info = self.buffer[self.indices[i]]
            new_info = self.info_ema * old_info + (1 - self.info_ema) * info
            self.buffer[self.indices[i]] = (img, labels, img_id, new_info)
    
    def replace_sample(self, sample, info, idx=None):
        data_class = sample.get('label', None)
        img, labels, img_id = self.load_data(sample['file_name'], is_stream=True,data_class=data_class)
        data = (img, labels, img_id, info)
        classes = [obj['category_id'] for obj in labels]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]

        if sample.get('klass', None):
            # self.cls_count[self.new_exposed_classes.index(sample['klass'])] += 1
            # sample_category = sample['klass']
            classes = [obj['category_id'] for obj in labels]
            classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                    for c in classes]
            classes = list(set(classes))
            for cls_ in classes:
                self.cls_count[cls_] += 1
            sample_category = classes
        elif sample.get('domain', None):
            self.cls_count[self.new_exposed_classes.index(sample['domain'])] += 1
            sample_category = sample['domain']
        else:
            self.cls_count[self.new_exposed_classes.index('pretrained')] += 1
            sample_category = 'pretrained'
        # self.cls_count[self.new_exposed_classes.index(sample.get('klass', 'pretrained'))] += 1
        if idx is None:
            if isinstance(sample_category, list):
                for cls_ in sample_category:
                    self.cls_idx[cls_].append(len(self.buffer))
            else:
                self.cls_idx[self.new_exposed_classes.index(sample_category)].append(len(self.buffer))
            self.buffer.append(data)
        else:
            self.buffer[idx] = data
    
    def register_sample_for_initial_buffer(self, sample, idx=None):
        img, labels, img_id = self.load_data(sample['file_name'], is_stream=False)
        data = (img, labels, img_id, 0)
        classes = [obj['category_id'] for obj in labels]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]

        if idx is None:
            self.buffer.append(data)
        else:
            self.buffer[idx] = data
        
        ### END USAGE

        if not self.is_domain_incremental:
            # self.cls_count[self.new_exposed_classes.index(sample['klass'])] += 1
            # sample_category = sample['klass']
            classes = [obj['category_id'] for obj in labels]
            classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                    for c in classes]
            classes = list(set(classes))
            for cls_ in classes:
                self.cls_count[cls_] += 1
            sample_category = classes
        else:
            self.cls_count[self.new_exposed_classes.index('pretrained')] += 1
            sample_category = 'pretrained'
        # self.cls_count[self.new_exposed_classes.index(sample.get('klass', 'pretrained'))] += 1
        if idx is None:
            if isinstance(sample_category, list):
                for cls_ in sample_category:
                    self.cls_idx[cls_].append(len(self.buffer))
            else:
                self.cls_idx[self.new_exposed_classes.index(sample_category)].append(len(self.buffer))
            self.buffer.append(data)
        else:
            self.buffer[idx] = data
            
    
    def register_stream(self, datalist):
        self.stream_data = []
        for data in datalist:
            data_class = data.get('label', None)
            img_path = data.get('file_name', data.get('filepath'))
            
            img, labels, img_id = self.load_data(img_path, is_stream=True, data_class=data_class)
            
            self.stream_data.append((img, labels, img_id, 0))
            

    def get_buffer_data(self, ind, batch_size):
        data = []
        
        batch = self.buffer[ind:ind+batch_size]
        
        for i, entry in enumerate(batch):
            # img, bboxes, img_id = entry['img'], entry['labels'], entry['img_id']
            # valid_mask = labels[:, 0] != -1
            # bboxes = labels[valid_mask]
            if self.use_mosaic_mixup:
                img, label, img_id = self.mosaic_wrapper.__getitem__((True, i))
            else:
                img, anno, img_id, score = self.buffer[i]
                anno = [obj for obj in anno if obj['iscrowd'] == 0]

                boxes = [obj['bbox'] for obj in anno]
                boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

                classes = [obj['category_id'] for obj in anno]
                classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                        for c in classes]

                classes = torch.tensor(classes)
                target.add_field('labels', classes)


                target = target.clip_to_image(remove_empty=True)
                
                img = np.asarray(img)
                img, label = self._transforms(img, target)
            data.append((img, label, img_id))
            
        return self.batch_collator(data)
    
    
    @torch.no_grad()
    def get_batch(self, batch_size, stream_batch_size=0, weight_method=None):
        assert batch_size >= stream_batch_size
        stream_batch_size = min(stream_batch_size, len(self.stream_data))
        batch_size = min(batch_size, stream_batch_size + len(self.buffer))
        memory_batch_size = batch_size - stream_batch_size

        data = []
        
        # append stream data to buffer for batch creation
        buffer_size = len(self.buffer)
        self.buffer.extend(self.stream_data)
 
        if stream_batch_size > 0:
            stream_indices = np.random.choice(range(len(self.stream_data)), size=stream_batch_size, replace=False)
            for i in stream_indices:
                if self.use_mosaic_mixup:
                    img, label, img_id = self.mosaic_wrapper.__getitem__((True, i + buffer_size))
                else:
                    img, anno, img_id, score = self.buffer[i+buffer_size]
                    anno = [obj for obj in anno if obj['iscrowd'] == 0]

                    boxes = [obj['bbox'] for obj in anno]
                    boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                    target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

                    classes = [obj['category_id'] for obj in anno]
                    classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                            for c in classes]

                    classes = torch.tensor(classes)
                    target.add_field('labels', classes)

                    target = target.clip_to_image(remove_empty=True)
                    
                    img = np.asarray(img)
                    img, label = self._transforms(img, target)
                
                data.append((img, label, img_id))

        # ───── Memory part ──────────────────────────────────────────────
        if memory_batch_size > 0 and len(self.buffer):
            
            info_list = np.array([e[-1] for e in self.buffer],
                        dtype=np.float64)
            info_list = info_list[:buffer_size]
            w = info_list / info_list.sum()
            ### HYBRID WEIGHT END
            if self.priority_selection == "high":
                indices = np.argpartition(w, -memory_batch_size)[-memory_batch_size:]
            elif self.priority_selection == "low":
                indices = np.argpartition(w, memory_batch_size)[:memory_batch_size] 
            elif self.priority_selection == "prob":
                indices = np.random.choice(
                    len(info_list),
                    size=memory_batch_size,
                    replace=len(info_list) < memory_batch_size,
                    p=w,
                )
            self.indices = indices
            for i in indices:
                if self.use_mosaic_mixup:
                    img, label, img_id = self.mosaic_wrapper.__getitem__((True, i))
                else:
                    img, anno, img_id, score = self.buffer[i]
                    anno = [obj for obj in anno if obj['iscrowd'] == 0]

                    boxes = [obj['bbox'] for obj in anno]
                    boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                    target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

                    classes = [obj['category_id'] for obj in anno]
                    classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                            for c in classes]

                    classes = torch.tensor(classes)
                    target.add_field('labels', classes)


                    target = target.clip_to_image(remove_empty=True)
                    
                    img = np.asarray(img)
                    img, label = self._transforms(img, target)
                data.append((img, label, img_id))

         # remove stream data from buffer
        self.buffer = self.buffer[:buffer_size]
        return self.batch_collator(data)
    
    def pull_item(self, idx):
        img, anno, img_id, score = self.buffer[idx]
        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj['bbox'] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')
        
        target = target.clip_to_image(remove_empty=True)

        classes = [obj['category_id'] for obj in anno]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]

        obj_masks = []
        for obj in anno:
            obj_mask = []
            if 'segmentation' in obj:
                for mask in obj['segmentation']:
                    obj_mask += mask
                if len(obj_mask) > 0:
                    obj_masks.append(obj_mask)
        seg_masks = [
            np.array(obj_mask, dtype=np.float32).reshape(-1, 2)
            for obj_mask in obj_masks
        ]

        res = np.zeros((len(target.bbox), 5))
        for idx in range(len(target.bbox)):
            res[idx, 0:4] = target.bbox[idx]
            res[idx, 4] = classes[idx]

        img = np.asarray(img)  # rgb

        return img, res, seg_masks, img_id
    
    def load_anno(self, idx):
        _, anno, _,_ = self.buffer[idx]
        anno = [obj for obj in anno if obj['iscrowd'] == 0]
        classes = [obj['category_id'] for obj in anno]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]
        return classes



# Selection-Based Retrieval + Freq Retrieval
class SelectionFrequencyDataset(FreqDataset):
    def __init__(self, ann_file, root, transforms=None, class_names=None,
                 dataset=None, cls_list=None, device=None, data_dir=None, memory_size=None, 
                 init_buffer_size=None, image_size=(640, 640), aug=None, selection_method=None, priority_selection=None
                 ):
        super().__init__(ann_file, root, transforms, class_names,
            dataset,cls_list,device,data_dir,memory_size,
            init_buffer_size,image_size,aug)
        
        self.selection_method = selection_method
        self.priority_selection = priority_selection
        self.alpha = 1.0                  
        self.beta = 1.0
        self.use_mosaic_mixup=False
        self.info_ema = 0.9
        
    def update_initialinfo(self, info_list):
        assert len(info_list) == len(self.buffer)
        for i, info in enumerate(info_list):
            entry = self.buffer[i]
            entry["info"] = info
            self.buffer[i] = entry
            
    def update_info(self, infos):
        assert len(infos) == len(self.indices)
        for i, info in enumerate(infos):
            entry = self.buffer[self.indices[i]]
            entry["info"] = self.info_ema * entry["info"] + (1 - self.info_ema) * info
            self.buffer[self.indices[i]] = entry
    
    def replace_sample(self, sample, info, idx=None):
        data_class = sample.get('label', None)
        img, labels, img_id = self.load_data(sample['file_name'], is_stream=True,data_class=data_class)
        classes = [obj['category_id'] for obj in labels]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]

        entry = {
            "img": img,
            "labels": labels,
            "img_id": img_id,
            "usage": sample.get("usage", 0),
            "classes": list(set(classes)) if len(classes) else [],
            "info": info,
        }
        ### END USAGE

        if idx is None:
            self.buffer.append(entry)
        else:
            self.buffer[idx] = entry
    
    
    def register_sample_for_initial_buffer(self, sample, idx=None):
        img, labels, img_id = self.load_data(sample['file_name'], is_stream=False)
        classes = [obj['category_id'] for obj in labels]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]

        entry = {
            "img": img,
            "labels": labels,
            "img_id": img_id,
            "usage": sample.get("usage", 0),
            "classes": list(set(classes)) if len(classes) else [],
            "info": 0
        }
        ### END USAGE

        if idx is None:
            self.buffer.append(entry)
        else:
            self.buffer[idx] = entry
    
    def register_stream(self, datalist):
        self.stream_data = []
        for data in datalist:
            data_class = data.get('label', None)
            img_path = data.get('file_name', data.get('filepath'))
            
            img, labels, img_id = self.load_data(img_path, is_stream=True, data_class=data_class)
            
            classes = [obj['category_id'] for obj in labels]
            classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                    for c in classes]
            entry = {
                "img": img,
                "labels": labels,
                "img_id": img_id,
                "usage": 0,
                "classes": list(set(classes)) if len(classes) else [],
                "info": 0,
            }
            
            self.stream_data.append(entry)
            

    def get_buffer_data(self, ind, batch_size):
        data = []
        
        batch = self.buffer[ind:ind+batch_size]
        
        for i, entry in enumerate(batch):
            # img, bboxes, img_id = entry['img'], entry['labels'], entry['img_id']
            # valid_mask = labels[:, 0] != -1
            # bboxes = labels[valid_mask]
            if self.use_mosaic_mixup:
                img, label, img_id = self.mosaic_wrapper.__getitem__((True, i))
            else:
                entry = self.buffer[i]
                img, anno, img_id = entry['img'], entry['labels'], entry['img_id']
                anno = [obj for obj in anno if obj['iscrowd'] == 0]

                boxes = [obj['bbox'] for obj in anno]
                boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

                classes = [obj['category_id'] for obj in anno]
                classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                        for c in classes]

                classes = torch.tensor(classes)
                target.add_field('labels', classes)

                target = target.clip_to_image(remove_empty=True)
                
                img = np.asarray(img)
                img, label = self._transforms(img, target)
            data.append((img, label, img_id))
            
        return self.batch_collator(data)
    
    
    @torch.no_grad()
    def get_batch(self, batch_size, stream_batch_size=0, weight_method=None):
        assert batch_size >= stream_batch_size
        stream_batch_size = min(stream_batch_size, len(self.stream_data))
        batch_size = min(batch_size, stream_batch_size + len(self.buffer))
        memory_batch_size = batch_size - stream_batch_size

        data = []
        
        # append stream data to buffer for batch creation
        buffer_size = len(self.buffer)
        self.buffer.extend(self.stream_data)
 
        if stream_batch_size > 0:
            stream_indices = np.random.choice(range(len(self.stream_data)), size=stream_batch_size, replace=False)
            for i in stream_indices:
                if self.use_mosaic_mixup:
                    img, label, img_id = self.mosaic_wrapper.__getitem__((True, i + buffer_size))
                else:
                    entry = self.buffer[i + buffer_size]
                    img, anno, img_id = entry['img'], entry['labels'], entry['img_id']
                    anno = [obj for obj in anno if obj['iscrowd'] == 0]

                    boxes = [obj['bbox'] for obj in anno]
                    boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                    target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

                    classes = [obj['category_id'] for obj in anno]
                    classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                            for c in classes]

                    classes = torch.tensor(classes)
                    target.add_field('labels', classes)

                    target = target.clip_to_image(remove_empty=True)
                    
                    img = np.asarray(img)
                    img, label = self._transforms(img, target)
                
                data.append((img, label, img_id))

        # ───── Memory part ──────────────────────────────────────────────
        if memory_batch_size > 0 and len(self.buffer):
            
            for e in self.buffer:
                e['usage'] *= self.usage_decay
            for cls_idx in range(len(self.cls_train_cnt)):
                self.cls_train_cnt[cls_idx] *= self.usage_decay
            
            ### HYBRID WEIGHT BEGIN
            if weight_method == "cls_usage":          # new option
                # 1 / (usage+α)  ×  1 / (mean cls_trained + β)
                alpha = getattr(self, "alpha", 1.0)
                beta  = getattr(self, "beta", 1.0)       # you may set self.beta in __init__
                weights = []
                for entry in self.buffer:
                    u = entry["usage"]
                    # gather per-image class-trained counts
                    if entry["classes"]:
                        t = [self.cls_train_cnt[int(c)]    # safe: cls_dict maps real IDs
                            for c in entry["classes"]               # (skip if not yet in dict)
                            if c < len(self.cls_train_cnt)]
                        mean_t = np.mean(t) if t else 0.0
                    else:
                        mean_t = 0.0        # no GT boxes → neutral
                    weights.append(1.0 / (u + alpha) * 1.0 / (mean_t + beta))
                w = np.asarray(weights, dtype=np.float64)
                w /= w.sum()
            else:
                w = np.array([1.0 / (e["usage"] + self.alpha) for e in self.buffer],
                            dtype=np.float64)
                w /= w.sum()
            
            info_list = np.array([e["info"] for e in self.buffer],
                        dtype=np.float64)
            info_list = info_list[:buffer_size]
            w2 = np.array(info_list / info_list.sum())
            ### HYBRID WEIGHT END
            
            final_w = [(x + y) / 2 for x, y in zip(w, w2)]
            final_w = np.array(final_w)
            final_w /= final_w.sum()
            
            if self.priority_selection == "high":
                indices = np.argpartition(final_w, -memory_batch_size)[-memory_batch_size:]
            elif self.priority_selection == "low":
                indices = np.argpartition(final_w, memory_batch_size)[:memory_batch_size] 
            elif self.priority_selection == "prob":
                indices = np.random.choice(
                    len(info_list),
                    size=memory_batch_size,
                    replace=len(info_list) < memory_batch_size,
                    p=final_w,
                )
            self.indices = indices
            
            for i in indices:
                # update usage counter *and* class-train counts
                self.buffer[i]["usage"] += 1
                for idx_cls in self.buffer[i]["classes"]:
                    if idx_cls < len(self.cls_train_cnt):
                        self.cls_train_cnt[int(idx_cls)] += 1
                
                if self.use_mosaic_mixup:
                    img, label, img_id = self.mosaic_wrapper.__getitem__((True, i))
                else:
                    entry = self.buffer[i]
                    img, anno, img_id = entry['img'], entry['labels'], entry['img_id']
                    anno = [obj for obj in anno if obj['iscrowd'] == 0]

                    boxes = [obj['bbox'] for obj in anno]
                    boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                    target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

                    classes = [obj['category_id'] for obj in anno]
                    classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                            for c in classes]

                    classes = torch.tensor(classes)
                    target.add_field('labels', classes)


                    target = target.clip_to_image(remove_empty=True)
                    
                    img = np.asarray(img)
                    img, label = self._transforms(img, target)
                data.append((img, label, img_id))

         # remove stream data from buffer
        self.buffer = self.buffer[:buffer_size]
        return self.batch_collator(data)
    
    
    
# Selection-Based Retrieval + Freq Retrieval + Cls_Balanced
class SelectionFreqBalancedDataset(FreqClsBalancedDataset):
    def __init__(self, ann_file, root, transforms=None, class_names=None,
                 dataset=None, cls_list=None, device=None, data_dir=None, memory_size=None, 
                 init_buffer_size=None, image_size=(640, 640), aug=None, selection_method=None, priority_selection=None
                 ):
        super().__init__(ann_file, root, transforms, class_names,
            dataset,cls_list,device,data_dir,memory_size,
            init_buffer_size,image_size,aug)
        
        self.selection_method = selection_method
        self.priority_selection = priority_selection
        self.alpha = 1.0                  
        self.beta = 1.0
        self.use_mosaic_mixup=False
        self.info_ema = 0.9
        
    def update_initialinfo(self, info_list):
        assert len(info_list) == len(self.buffer)
        for i, info in enumerate(info_list):
            entry = self.buffer[i]
            entry["info"] = info
            self.buffer[i] = entry
            
    def update_info(self, infos):
        assert len(infos) == len(self.indices)
        for i, info in enumerate(infos):
            entry = self.buffer[self.indices[i]]
            entry["info"] = self.info_ema * entry["info"] + (1 - self.info_ema) * info
            self.buffer[self.indices[i]] = entry
    
    def replace_sample(self, sample, info, idx=None):
        data_class = sample.get('label', None)
        img, labels, img_id = self.load_data(sample['file_name'], is_stream=True,data_class=data_class)
        classes = [obj['category_id'] for obj in labels]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]

        entry = {
            "img": img,
            "labels": labels,
            "img_id": img_id,
            "usage": sample.get("usage", 0),
            "classes": list(set(classes)) if len(classes) else [],
            "info": info,
        }
        ### END USAGE

        if sample.get('klass', None):
            # self.cls_count[self.new_exposed_classes.index(sample['klass'])] += 1
            # sample_category = sample['klass']
            classes = [obj['category_id'] for obj in labels]
            classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                    for c in classes]
            classes = list(set(classes))
            for cls_ in classes:
                self.cls_count[cls_] += 1
            sample_category = classes
        elif sample.get('domain', None):
            self.cls_count[self.new_exposed_classes.index(sample['domain'])] += 1
            sample_category = sample['domain']
        else:
            self.cls_count[self.new_exposed_classes.index('pretrained')] += 1
            sample_category = 'pretrained'
        # self.cls_count[self.new_exposed_classes.index(sample.get('klass', 'pretrained'))] += 1
        if idx is None:
            if isinstance(sample_category, list):
                for cls_ in sample_category:
                    self.cls_idx[cls_].append(len(self.buffer))
            else:
                self.cls_idx[self.new_exposed_classes.index(sample_category)].append(len(self.buffer))
            self.buffer.append(entry)
        else:
            self.buffer[idx] = entry
    
    
    def register_sample_for_initial_buffer(self, sample, idx=None):
        img, labels, img_id = self.load_data(sample['file_name'], is_stream=False)
        classes = [obj['category_id'] for obj in labels]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]

        entry = {
            "img": img,
            "labels": labels,
            "img_id": img_id,
            "usage": sample.get("usage", 0),
            "classes": list(set(classes)) if len(classes) else [],
            "info": 0
        }
        ### END USAGE

        if not self.is_domain_incremental:
            # self.cls_count[self.new_exposed_classes.index(sample['klass'])] += 1
            # sample_category = sample['klass']
            classes = [obj['category_id'] for obj in labels]
            classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                    for c in classes]
            classes = list(set(classes))
            for cls_ in classes:
                self.cls_count[cls_] += 1
            sample_category = classes
        else:
            self.cls_count[self.new_exposed_classes.index('pretrained')] += 1
            sample_category = 'pretrained'
        # self.cls_count[self.new_exposed_classes.index(sample.get('klass', 'pretrained'))] += 1
        if idx is None:
            if isinstance(sample_category, list):
                for cls_ in sample_category:
                    self.cls_idx[cls_].append(len(self.buffer))
            else:
                self.cls_idx[self.new_exposed_classes.index(sample_category)].append(len(self.buffer))
            self.buffer.append(entry)
        else:
            self.buffer[idx] = entry
    
    def register_stream(self, datalist):
        self.stream_data = []
        for data in datalist:
            data_class = data.get('label', None)
            img_path = data.get('file_name', data.get('filepath'))
            
            img, labels, img_id = self.load_data(img_path, is_stream=True, data_class=data_class)
            
            classes = [obj['category_id'] for obj in labels]
            classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                    for c in classes]
            entry = {
                "img": img,
                "labels": labels,
                "img_id": img_id,
                "usage": 0,
                "classes": list(set(classes)) if len(classes) else [],
                "info": 0,
            }
            
            self.stream_data.append(entry)
            

    def get_buffer_data(self, ind, batch_size):
        data = []
        
        batch = self.buffer[ind:ind+batch_size]
        
        for i, entry in enumerate(batch):
            # img, bboxes, img_id = entry['img'], entry['labels'], entry['img_id']
            # valid_mask = labels[:, 0] != -1
            # bboxes = labels[valid_mask]
            if self.use_mosaic_mixup:
                img, label, img_id = self.mosaic_wrapper.__getitem__((True, i))
            else:
                entry = self.buffer[i]
                img, anno, img_id = entry['img'], entry['labels'], entry['img_id']
                anno = [obj for obj in anno if obj['iscrowd'] == 0]

                boxes = [obj['bbox'] for obj in anno]
                boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

                classes = [obj['category_id'] for obj in anno]
                classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                        for c in classes]

                classes = torch.tensor(classes)
                target.add_field('labels', classes)

                target = target.clip_to_image(remove_empty=True)
                
                img = np.asarray(img)
                img, label = self._transforms(img, target)
            data.append((img, label, img_id))
            
        return self.batch_collator(data)
    
    
    @torch.no_grad()
    def get_batch(self, batch_size, stream_batch_size=0, weight_method=None):
        assert batch_size >= stream_batch_size
        stream_batch_size = min(stream_batch_size, len(self.stream_data))
        batch_size = min(batch_size, stream_batch_size + len(self.buffer))
        memory_batch_size = batch_size - stream_batch_size

        data = []
        
        # append stream data to buffer for batch creation
        buffer_size = len(self.buffer)
        self.buffer.extend(self.stream_data)
 
        if stream_batch_size > 0:
            stream_indices = np.random.choice(range(len(self.stream_data)), size=stream_batch_size, replace=False)
            for i in stream_indices:
                if self.use_mosaic_mixup:
                    img, label, img_id = self.mosaic_wrapper.__getitem__((True, i + buffer_size))
                else:
                    entry = self.buffer[i + buffer_size]
                    img, anno, img_id = entry['img'], entry['labels'], entry['img_id']
                    anno = [obj for obj in anno if obj['iscrowd'] == 0]

                    boxes = [obj['bbox'] for obj in anno]
                    boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                    target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

                    classes = [obj['category_id'] for obj in anno]
                    classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                            for c in classes]

                    classes = torch.tensor(classes)
                    target.add_field('labels', classes)

                    target = target.clip_to_image(remove_empty=True)
                    
                    img = np.asarray(img)
                    img, label = self._transforms(img, target)
                
                data.append((img, label, img_id))

        # ───── Memory part ──────────────────────────────────────────────
        if memory_batch_size > 0 and len(self.buffer):
            
            for e in self.buffer:
                e['usage'] *= self.usage_decay
            for cls_idx in range(len(self.cls_train_cnt)):
                self.cls_train_cnt[cls_idx] *= self.usage_decay
            
            ### HYBRID WEIGHT BEGIN
            if weight_method == "cls_usage":          # new option
                # 1 / (usage+α)  ×  1 / (mean cls_trained + β)
                alpha = getattr(self, "alpha", 1.0)
                beta  = getattr(self, "beta", 1.0)       # you may set self.beta in __init__
                weights = []
                for entry in self.buffer:
                    u = entry["usage"]
                    # gather per-image class-trained counts
                    if entry["classes"]:
                        t = [self.cls_train_cnt[int(c)]    # safe: cls_dict maps real IDs
                            for c in entry["classes"]               # (skip if not yet in dict)
                            if c < len(self.cls_train_cnt)]
                        mean_t = np.mean(t) if t else 0.0
                    else:
                        mean_t = 0.0        # no GT boxes → neutral
                    weights.append(1.0 / (u + alpha) * 1.0 / (mean_t + beta))
                w = np.asarray(weights, dtype=np.float64)
                w /= w.sum()
            else:
                w = np.array([1.0 / (e["usage"] + self.alpha) for e in self.buffer],
                            dtype=np.float64)
                w /= w.sum()
            
            info_list = np.array([e["info"] for e in self.buffer],
                        dtype=np.float64)
            info_list = info_list[:buffer_size]
            w2 = info_list / info_list.sum()
            ### HYBRID WEIGHT END
            
            # final_w = [(x + y) / 2 for x, y in zip(w, w2)]
            final_w = [x*y for x, y in zip(w, w2)]
            final_w = np.array(final_w)
            final_w /= final_w.sum()
            
            if self.priority_selection == "high":
                indices = np.argpartition(final_w, -memory_batch_size)[-memory_batch_size:]
            elif self.priority_selection == "low":
                indices = np.argpartition(final_w, memory_batch_size)[:memory_batch_size] 
            elif self.priority_selection == "prob":
                indices = np.random.choice(
                    len(info_list),
                    size=memory_batch_size,
                    replace=len(info_list) < memory_batch_size,
                    p=final_w,
                )
            self.indices = indices
            
            for i in indices:
                # update usage counter *and* class-train counts
                self.buffer[i]["usage"] += 1
                for idx_cls in self.buffer[i]["classes"]:
                    if idx_cls < len(self.cls_train_cnt):
                        self.cls_train_cnt[int(idx_cls)] += 1
                
                if self.use_mosaic_mixup:
                    img, label, img_id = self.mosaic_wrapper.__getitem__((True, i))
                else:
                    entry = self.buffer[i]
                    img, anno, img_id = entry['img'], entry['labels'], entry['img_id']
                    anno = [obj for obj in anno if obj['iscrowd'] == 0]

                    boxes = [obj['bbox'] for obj in anno]
                    boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                    target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

                    classes = [obj['category_id'] for obj in anno]
                    classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                            for c in classes]

                    classes = torch.tensor(classes)
                    target.add_field('labels', classes)


                    target = target.clip_to_image(remove_empty=True)
                    
                    img = np.asarray(img)
                    img, label = self._transforms(img, target)
                data.append((img, label, img_id))

         # remove stream data from buffer
        self.buffer = self.buffer[:buffer_size]
        return self.batch_collator(data)



################################################################################################ pseudo
# MemoryPseudoDataset
from damo.dataset.datasets.mosaic_wrapper import PseudoMosaicWrapper
from damo.dataset.transforms import transforms as T
def generate_pseudo_labels(model, img, score_threshold=0.5, transform=None,image_sizes=(640,640),device='cuda'):
    # generate pseudo labels
    model.eval()
    if transform is not None:
        img_cv = np.asarray(img)
        height, width = img_cv.shape[:2]
        img_tensor = transform(img_cv)[0].unsqueeze(0).to(device)
    else:
        breakpoint()
        height, width = img.shape[1], img.shape[2]
        img_tensor = img.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)[0]

    boxes = outputs.bbox.detach().to('cpu').float().numpy()
    labels = outputs.extra_fields['labels'].detach().to('cpu').long().numpy()
    scores = outputs.extra_fields.get('scores').detach().to('cpu').float().numpy()

    # rescale boxes to original image size
    boxes[:, 0] = boxes[:, 0] * width / image_sizes[1]
    boxes[:, 1] = boxes[:, 1] * height / image_sizes[0]
    boxes[:, 2] = boxes[:, 2] * width / image_sizes[1]
    boxes[:, 3] = boxes[:, 3] * height / image_sizes[0]
    
    # filter boxes by score threshold
    keep_indices = np.where(scores >= score_threshold)[0]
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    labels = labels[keep_indices]
    return boxes, labels, scores

################################# better pseudo labels ###############################3
from ensemble_boxes import weighted_boxes_fusion
from damo.structures.image_list import to_image_list
SCALES = [(640, 640)]#, (704, 704), (768, 768)]
HFLIPS = [False]#, True]
def _apply_feature_dropout_to_backbone_feats(
    feats,
    p: float = 0.1,
) -> List[torch.Tensor]:
    """
    Apply dropout to backbone feature maps.
    Supports list/tuple/dict of tensors (common for FPN inputs).
    Uses F.dropout(..., training=True) to force stochasticity during inference.
    """
    def drop_one(x):
        if isinstance(x, torch.Tensor):
            return F.dropout(x, p=p, training=True)
        return x

    if isinstance(feats, (list, tuple)):
        return [drop_one(f) for f in feats]
    elif isinstance(feats, dict):
        return {k: drop_one(v) for k, v in feats.items()}
    else:
        # fallback (single tensor)
        return drop_one(feats)
def _forward_backbone_neck_head_eval(
    model,
    image_tensor: torch.Tensor,   # [1, C, H, W]
    dropout_p: float = 0.0,
    mc_passes: int = 0,
):
    """
    Splits forward like your snippet:
        new_feats_b = model.backbone(image_tensors)
        new_feats_n = model.neck(new_feats_b)
        out = model.head.forward_eval(new_feats_n, labels=None)
    If mc_passes>0, returns a list of outputs (length mc_passes) with
    dropout applied to backbone features before neck.
    Otherwise returns a single output.
    """
    # Backbone once (deterministic)
    with torch.no_grad():
        feats_b = model.backbone(image_tensor)
    img_list = to_image_list(image_tensor)
    if mc_passes and mc_passes > 0 and dropout_p > 0.0:
        outs = []
        with torch.no_grad():
            for _ in range(mc_passes):
                feats_b_do = _apply_feature_dropout_to_backbone_feats(feats_b, p=dropout_p)
                feats_n = model.neck(feats_b_do)
                out = model.head.forward_eval(feats_n, labels=None,imgs=img_list)[0]
                outs.append(out)
        return outs
    else:
        with torch.no_grad():
            feats_n = model.neck(feats_b)
            out = model.head.forward_eval(feats_n, labels=None,imgs=img_list)[0]
        return out
def _rescale_boxes_xyxy(boxes: np.ndarray, orig_w: int, orig_h: int, iw: int, ih: int):
    """Rescale boxes predicted on (iw, ih) back to original (orig_w, orig_h)."""
    boxes = boxes.copy()
    sx, sy = orig_w / iw, orig_h / ih
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy
    return boxes

def _maybe_hflip_boxes_inplace(boxes: np.ndarray, W: int):
    # invert a horizontal flip to original orientation
    x1 = boxes[:, 0].copy()
    x2 = boxes[:, 2].copy()
    boxes[:, 0] = W - x2
    boxes[:, 2] = W - x1

def _extract_np(out) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pull YOLO-style fields from your head.forward_eval output object:
      - out.bbox: [N,4] xyxy
      - out.extra_fields['scores']: [N]
      - out.extra_fields['labels']: [N]
    Returns np arrays.
    """
    boxes = out.bbox.detach().float().cpu().numpy()
    labels = out.extra_fields['labels'].detach().long().cpu().numpy()
    scores = out.extra_fields.get('scores').detach().float().cpu().numpy()
    return boxes, scores, labels

def _to_norm_xyxy(boxes: np.ndarray, W: int, H: int) -> np.ndarray:
    return np.stack([boxes[:,0]/W, boxes[:,1]/H, boxes[:,2]/W, boxes[:,3]/H], axis=1)

def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """
    IoU between two boxes (xyxy).
    """
    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, (a[2]-a[0])) * max(0.0, (a[3]-a[1]))
    area_b = max(0.0, (b[2]-b[0])) * max(0.0, (b[3]-b[1]))
    union = area_a + area_b - inter + 1e-9
    return inter / union

def generate_pseudo_labels_tta_mc_ugpl_first(
    models,
    img,                              # PIL or np image
    device='cuda',
    scales=SCALES,
    hflips=HFLIPS,
    # UGPL thresholds (SSAL)
    ugpl_prob_thr: float = 0.25,       # κ1: mean prob threshold
    ugpl_min_consistency: int = 15,    # κ2: min distinct views in T
    agg_iou_match: float = 0.5,       # γ: IoU for consistency
    # WBF for fusing inside each selected cluster
    wbf_iou_thr: float = 0.55,
    wbf_skip_box_thr: float = 0.0,    # we already filtered via UGPL
    wbf_conf_type: str = 'avg',
    # Optional caps/filters
    prefilter_score: float = 0.1,    # ignore very low-score raw dets early
    max_pseudo_per_image: int | None = None,
    # MC settings
    mc_passes: int = 10,
    dropout_p: float = 0.10,
):
    """
    SSAL-style pipeline:
      1) Build class-consistent clusters T across all TTA×MC×models by IoU≥γ.
      2) Compute p_hat (mean prob) and |T| (distinct views) for each cluster.
      3) Keep clusters with p_hat>=ugpl_prob_thr and |T|>=ugpl_min_consistency.
      4) Fuse *within each kept cluster* using WBF to get final pseudo labels.

    Returns:
      pl_boxes [M,4] float32 (xyxy, original image coords)
      pl_labels [M] int64
      pl_scores [M] float32  (you can choose to return p_hat here if preferred)
      pl_meta: dict with arrays 'p_hat', 'T_count'
    """

    img_cv = np.asarray(img)
    H, W = img_cv.shape[:2]

    # ===== 1) Collect raw detections across all views =====
    raw = []  # list of dicts with: box, score, label, view_id
    view_id = 0

    for model in models:
        model.eval()
        for (ih, iw) in scales:
            for flip in hflips:
                view_img = np.ascontiguousarray(img_cv[:, ::-1] if flip else img_cv)
                tfm = T.Compose(
                    [
                        T.Resize((ih, iw), keep_ratio=False),
                        T.ToTensor(),
                        T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
                    ]
                )
                inp = tfm(view_img)[0].unsqueeze(0).to(device)

                out = _forward_backbone_neck_head_eval(
                    model, inp, dropout_p=dropout_p, mc_passes=mc_passes
                )
                outs = out if isinstance(out, list) else [out]

                for out_i in outs:
                    boxes, scores, labels = _extract_np(out_i)
                    # drop tiny-conf detections up front
                    if prefilter_score is not None and prefilter_score > 0.0:
                        keep = scores >= prefilter_score
                        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

                    # back to original coords
                    boxes = _rescale_boxes_xyxy(boxes, W, H, iw, ih)
                    if flip:
                        _maybe_hflip_boxes_inplace(boxes, W)

                    # stash with view id
                    for b, s, c in zip(boxes, scores, labels):
                        raw.append({
                            'box': b.astype(np.float32),
                            'score': float(s),
                            'label': int(c),
                            'view': view_id
                        })
                    view_id += 1  # distinct per forward pass (TTA×MC×model)

    if len(raw) == 0:
        return (np.zeros((0, 4), np.float32),
                np.zeros((0,), np.int64),
                np.zeros((0,), np.float32),)
                #{'p_hat': np.zeros((0,), np.float32), 'T_count': np.zeros((0,), np.int32)})

    # ===== 2) Build class-wise clusters T via greedy IoU matching =====
    # Sort by score (desc) for stable clustering
    raw_sorted_idx = np.argsort([-r['score'] for r in raw])
    clusters = []  # list of dicts: {'label', 'members': [indices], 'views': set()}
    # For convenience, keep arrays
    raw_boxes = np.stack([raw[i]['box'] for i in range(len(raw))], axis=0)
    raw_scores = np.array([r['score'] for r in raw], dtype=np.float32)
    raw_labels = np.array([r['label'] for r in raw], dtype=np.int64)
    raw_views  = np.array([r['view']  for r in raw], dtype=np.int32)

    for idx in raw_sorted_idx:
        c = raw_labels[idx]
        b = raw_boxes[idx]
        # try to attach to an existing cluster of the same class
        attach_k = -1
        best_iou = -1.0
        for k, cl in enumerate(clusters):
            if cl['label'] != c:
                continue
            # compare with cluster's running representative: use highest-score member’s box
            rep_idx = cl['members'][0]
            iou = _iou_xyxy(b, raw_boxes[rep_idx])
            if iou >= agg_iou_match and iou > best_iou:
                attach_k, best_iou = k, iou
        if attach_k >= 0:
            clusters[attach_k]['members'].append(idx)
            clusters[attach_k]['views'].add(int(raw_views[idx]))
        else:
            clusters.append({
                'label': int(c),
                'members': [int(idx)],
                'views': {int(raw_views[idx])},
            })

    # ===== 3) Compute SSAL stats (p_hat, |T|) and select by UGPL =====
    p_hat_list, T_list, keep_mask = [], [], []
    for cl in clusters:
        ms = raw_scores[cl['members']]
        p_hat = float(ms.mean()) if ms.size > 0 else 0.0
        T_cnt = len(cl['views'])  # distinct views that produced consistent boxes
        p_hat_list.append(p_hat)
        T_list.append(T_cnt)
        keep_mask.append((p_hat >= ugpl_prob_thr) and (T_cnt >= ugpl_min_consistency))

    p_hat_arr = np.array(p_hat_list, dtype=np.float32)
    T_arr = np.array(T_list, dtype=np.int32)
    keep_idx = np.where(np.array(keep_mask, dtype=bool))[0]

    if keep_idx.size == 0:
        return (np.zeros((0, 4), np.float32),
                np.zeros((0,), np.int64),
                np.zeros((0,), np.float32),
                {'p_hat': np.zeros((0,), np.float32), 'T_count': np.zeros((0,), np.int32)})

    # Optional cap based on p_hat
    # if max_pseudo_per_image is not None and keep_idx.size > max_pseudo_per_image:
    #     order = np.argsort(-p_hat_arr[keep_idx])[:max_pseudo_per_image]
    #     keep_idx = keep_idx[order]

    # ===== 4) Fuse *within each kept cluster* using WBF =====
    pl_boxes, pl_labels, pl_scores = [], [], []
    pl_p_hat, pl_T = [], []

    for k in keep_idx:
        cl = clusters[k]
        memb = cl['members']
        # Prepare WBF inputs: treat each member as its own "model"
        boxes_list = []
        scores_list = []
        labels_list = []
        for m in memb:
            b = raw_boxes[m]
            # normalize to [0,1] for WBF
            bn = _to_norm_xyxy(b[None, :], W, H)[0].tolist()
            boxes_list.append([bn])                # one box in this "model"
            scores_list.append([float(raw_scores[m])])
            labels_list.append([int(raw_labels[m])])

        fb_n, fs_wbf, fl = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            iou_thr=wbf_iou_thr, skip_box_thr=wbf_skip_box_thr, conf_type=wbf_conf_type
        )

        # Back to absolute coords; keep the top fused box (there should be 1)
        fb_n = np.array(fb_n, dtype=np.float32)
        fb = fb_n.copy()
        fb[:, [0, 2]] *= W
        fb[:, [1, 3]] *= H

        # Append
        pl_boxes.append(fb[0])
        pl_labels.append(int(fl[0]))
        # choose the training weight/score you prefer (WBF score or p_hat). Here: p_hat.
        pl_scores.append(float(p_hat_arr[k]))
        pl_p_hat.append(float(p_hat_arr[k]))
        pl_T.append(int(T_arr[k]))

    pl_boxes = np.stack(pl_boxes, axis=0).astype(np.float32)
    pl_labels = np.array(pl_labels, dtype=np.int64)
    pl_scores = np.array(pl_scores, dtype=np.float32)
    
    return pl_boxes, pl_labels, pl_scores
#, {'p_hat': np.array(pl_p_hat, np.float32),
                                            #'T_count': np.array(pl_T, np.int32)}

class MemoryPseudoDataset(MemoryDataset):
    def __init__(self, ann_file, root, transforms=None, class_names=None,
                 dataset=None, cls_list=None, device=None, data_dir=None, memory_size=None, 
                 init_buffer_size=None, image_size=(640, 640), aug=None
                 ):
        super().__init__(ann_file, root, transforms, class_names,
                         dataset,cls_list,device,data_dir,memory_size,
                         init_buffer_size,image_size,aug)
        self.mosaic_wrapper = PseudoMosaicWrapper(
            dataset=self,                      # <-- this class will provide pull_item/load_anno
            img_size=image_size,
            mosaic_prob=aug.mosaic_mixup.mosaic_prob,                   # always “allowed”; actual apply is still random inside wrapper
            mixup_prob=aug.mosaic_mixup.mixup_prob,
            transforms=self._transforms,                   # we'll run your own transforms after mosaic
            degrees=aug.mosaic_mixup.degrees,
            translate=aug.mosaic_mixup.translate,
            mosaic_scale=aug.mosaic_mixup.mosaic_scale,
            keep_ratio=True,
        )
        self.use_mosaic_mixup=False
        self.test_transform = T.Compose(
            [
                T.Resize(image_size, keep_ratio=False),
                T.ToTensor(),
                T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ]
        )
    def register_stream(self, datalist):
        self.stream_data = []
        for data in datalist:
            data_class = data.get('label', None)
            img_path = data.get('file_name', data.get('filepath'))
            
            img, labels, img_id = self.load_data(img_path, is_stream=True, data_class=data_class)
            
            self.stream_data.append((img, None, img_id))
    def replace_sample(self, sample, idx=None, images_dir=None,label_path=None):
        data_class = sample.get('label', None)
        img, labels, img_id = self.load_data(sample['file_name'], is_stream=True,data_class=data_class)
        data = (img, None, img_id) # don't save label

        if idx is None:
            self.buffer.append(data)
        else:
            self.buffer[idx] = data

    @torch.no_grad()
    def get_batch(self, batch_size, stream_batch_size=0, weight_method=None,
                  model=None, score_threshold=0.5
                  ):
        assert batch_size >= stream_batch_size
        stream_batch_size = min(stream_batch_size, len(self.stream_data))
        batch_size = min(batch_size, stream_batch_size + len(self.buffer))
        memory_batch_size = batch_size - stream_batch_size

        data = []
        
        # append stream data to buffer for batch creation
        buffer_size = len(self.buffer)
        self.buffer.extend(self.stream_data)
 
        if stream_batch_size > 0:
            stream_indices = np.random.choice(range(len(self.stream_data)), size=stream_batch_size, replace=False)
            for i in stream_indices:
                if self.use_mosaic_mixup:
                    img, label, img_id = self.mosaic_wrapper.__getitem__((True, i + buffer_size, model, score_thresh))
                else:
                    img, anno, img_id = self.buffer[i + buffer_size]
                    
                    if anno is not None:
                    
                        anno = [obj for obj in anno if obj['iscrowd'] == 0]

                        boxes = [obj['bbox'] for obj in anno]
                        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

                        classes = [obj['category_id'] for obj in anno]
                        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                                for c in classes]

                        classes = torch.tensor(classes)
                        target.add_field('labels', classes)


                        target = target.clip_to_image(remove_empty=True)
                        
                        img = np.asarray(img)
                        img, label = self._transforms(img, target)
                    else:
                        # boxes, labels, scores = generate_pseudo_labels(model, img, score_thresh=score_thresh, transform=self.test_transform, image_sizes=self.image_sizes, device=self.device)
                        boxes, labels, scores = generate_pseudo_labels(model, img, transform=self.test_transform, device=self.device, score_threshold=score_threshold)

                        if len(boxes) > 0:
                            target = BoxList(torch.tensor(boxes), img.size, mode='xyxy')
                            target.add_field('labels', torch.tensor(labels))
                            target = target.clip_to_image(remove_empty=True)

                            img, label = self._transforms(np.asarray(img), target)
                        else:
                            # no valid boxes
                            # create empty label
                            target = BoxList(torch.zeros((0,4)), img.size, mode='xyxy')
                            target.add_field('labels', torch.tensor([]))
                            img, label = self._transforms(np.asarray(img), target)
                
                data.append((img, label, img_id))

        if memory_batch_size > 0:
            indices = np.random.choice(range(buffer_size), size=memory_batch_size, replace=False)
            for i in indices:
                if self.use_mosaic_mixup:
                    img, label, img_id = self.mosaic_wrapper.__getitem__((True, i, model, score_thresh))
                else:
                    img, label, img_id = self.buffer[i]
                    if label is not None:
                        label = [obj for obj in label if obj['iscrowd'] == 0]

                        boxes = [obj['bbox'] for obj in label]
                        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

                        classes = [obj['category_id'] for obj in label]
                        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                                for c in classes]

                        classes = torch.tensor(classes)
                        target.add_field('labels', classes)


                        target = target.clip_to_image(remove_empty=True)
                        
                        img = np.asarray(img)
                        img, label = self._transforms(img, target)
                    else:
                        # boxes, labels, scores = generate_pseudo_labels(model, img, score_thresh=score_thresh,transform=self.test_transform, image_sizes=self.image_sizes, device=self.device)
                        boxes, labels, scores = generate_pseudo_labels(model, img, transform=self.test_transform, device=self.device, score_threshold=score_threshold)

                        if len(boxes) > 0:
                            target = BoxList(torch.tensor(boxes), img.size, mode='xyxy')
                            target.add_field('labels', torch.tensor(labels))
                            target = target.clip_to_image(remove_empty=True)

                            img, label = self._transforms(np.asarray(img), target)
                        else:
                            # no valid boxes
                            # create empty label
                            target = BoxList(torch.zeros((0,4)), img.size, mode='xyxy')
                            target.add_field('labels', torch.tensor([]))
                            img, label = self._transforms(np.asarray(img), target)
                
                data.append((img, label, img_id))

        # remove stream data from buffer
        self.buffer = self.buffer[:buffer_size]
        
        return self.batch_collator(data)
    
    def pull_item(self, idx, model=None, score_thresh=0.7):
        img, anno, img_id = self.buffer[idx]
        
        if anno is None:
            boxes, labels, scores = generate_pseudo_labels(model, img, score_thresh=score_thresh, transform=self.test_transform, image_sizes=self.image_sizes, device=self.device)
            anno = []
            for box, label in zip(boxes, labels):
                anno.append({'bbox': box.tolist(), 'category_id': int(label), 'iscrowd': 0})
        
        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj['bbox'] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')
        
        target = target.clip_to_image(remove_empty=True)

        classes = [obj['category_id'] for obj in anno]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]

        obj_masks = []
        for obj in anno:
            obj_mask = []
            if 'segmentation' in obj:
                for mask in obj['segmentation']:
                    obj_mask += mask
                if len(obj_mask) > 0:
                    obj_masks.append(obj_mask)
        seg_masks = [
            np.array(obj_mask, dtype=np.float32).reshape(-1, 2)
            for obj_mask in obj_masks
        ]

        res = np.zeros((len(target.bbox), 5))
        for idx in range(len(target.bbox)):
            res[idx, 0:4] = target.bbox[idx]
            res[idx, 4] = classes[idx]

        img = np.asarray(img)  # rgb

        return img, res, seg_masks, img_id
    
    def load_anno(self, idx, model=None, score_thresh=0.7):
        img, anno, _ = self.buffer[idx]
        if anno is None:
            boxes, labels, scores = generate_pseudo_labels(model, img, score_thresh=score_thresh, transform=self.test_transform, image_sizes=self.image_sizes, device=self.device)
            anno = []
            for box, label in zip(boxes, labels):
                anno.append({'bbox': box.tolist(), 'category_id': int(label), 'iscrowd': 0})
            
        anno = [obj for obj in anno if obj['iscrowd'] == 0]
        classes = [obj['category_id'] for obj in anno]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]
        return classes

class FreqClsBalancedPseudoDataset(MemoryDataset):
    def __init__(self, ann_file, root, transforms=None, class_names=None,
                 dataset=None, cls_list=None, device=None, data_dir=None, memory_size=None, 
                 init_buffer_size=None, image_size=(640, 640), aug=None
                 ):
        super(MemoryDataset, self).__init__(ann_file, root, transforms, class_names)
        # self.args = args
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_sizes = [int(image_size[0]), int(image_size[1])]
        self.memory_size = memory_size

        self.buffer = []
        self.stream_data = []
        

        self.dataset = dataset
        self.device = device
        self.data_dir = data_dir

        
        self.class_usage_cnt = []
        
        
        # FIXME: fix for object detection class counting
        self.cls_list = cls_list if cls_list else []
        
        
        self.cls_count = [0]
        self.cls_idx = [[]]
        
        self.new_exposed_classes = ['pretrained']
        self.cls_train_cnt = np.array([0,]*len(cls_list)) if cls_list else np.array([])
        # self.cls_train_cnt = np.array([])
        
        
        
        
        
        
        
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
            self.name2id = {re.sub(r'\.(jpg|jpeg|png)$', '', ann['file_name'], flags=re.I): ann['id'] for ann in annotations['images']}
            
        if 'HS_TOD' not in self.dataset:
            self.build_initial_buffer(init_buffer_size)

        n_classes, image_dir, label_path = get_statistics(dataset=self.dataset)
        self.image_dir = image_dir
        self.label_path = label_path
        
        # set transform
        transforms = aug.transform
        transforms = build_transforms_memorydataset(**transforms)
        self._transforms = transforms
        
        # aug.mosaic_mixup.mosaic_prob = 0
        # aug.mosaic_mixup.mixup_prob = 0
        # aug.mosaic_mixup.degrees = 0
        # aug.mosaic_mixup.translate = 0
        # aug.mosaic_mixup.mosaic_scale = (0.8, 1.2)
        
        print("mosaic_prob", aug.mosaic_mixup.mosaic_prob)
        print("mixup_prob", aug.mosaic_mixup.mixup_prob)
        print("degrees", aug.mosaic_mixup.degrees)
        print("translate", aug.mosaic_mixup.translate)
        print("mosaic_scale", aug.mosaic_mixup.mosaic_scale)
        
        
        self.mosaic_wrapper = PseudoMosaicWrapper(
            dataset=self,                      # <-- this class will provide pull_item/load_anno
            img_size=image_size,
            mosaic_prob=aug.mosaic_mixup.mosaic_prob,                   # always “allowed”; actual apply is still random inside wrapper
            mixup_prob=aug.mosaic_mixup.mixup_prob,
            transforms=self._transforms,                   # we'll run your own transforms after mosaic
            degrees=aug.mosaic_mixup.degrees,
            translate=aug.mosaic_mixup.translate,
            mosaic_scale=aug.mosaic_mixup.mosaic_scale,
            keep_ratio=True,
        )
        self.use_mosaic_mixup=False 
        
        # self.batch_collator = BatchCollator(size_divisible=32)
        self.batch_collator = HarmoniousBatchCollator(size_divisible=32)
        
        self.alpha = 1.0                  # smoothing constant in 1/(usage+α)
        self.beta = 1.0
        self.usage_decay = 0.995
        
        self.test_transform = T.Compose(
            [
                T.Resize(image_size, keep_ratio=False),
                T.ToTensor(),
                T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ]
        )
        
        assert self.use_mosaic_mixup == False, "Not support mosaic with pseudo label now."
    
        
    def replace_sample(self, sample, idx=None, images_dir=None,label_path=None):
        data_class = sample.get('label', None)
        img, labels, img_id = self.load_data(sample['file_name'], is_stream=True,data_class=data_class)
        classes = [obj['category_id'] for obj in labels]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]
        ### BEGIN USAGE
        entry = {
            "img": img,
            "labels": None,
            "img_id": img_id,
            "usage": sample.get("usage", 0),
            "classes": []
        }
        ### END USAGE

        if sample.get('klass', None):
            self.cls_count[self.new_exposed_classes.index(sample['klass'])] += 1
            sample_category = sample['klass']
        elif sample.get('domain', None):
            self.cls_count[self.new_exposed_classes.index(sample['domain'])] += 1
            sample_category = sample['domain']
        else:
            self.cls_count[self.new_exposed_classes.index('pretrained')] += 1
            sample_category = 'pretrained'
        # self.cls_count[self.new_exposed_classes.index(sample.get('klass', 'pretrained'))] += 1
        if idx is None:
            self.cls_idx[self.new_exposed_classes.index(sample_category)].append(len(self.buffer))
            self.buffer.append(entry)
        else:
            self.buffer[idx] = entry
    
    def register_sample_for_initial_buffer(self, sample, idx=None, images_dir=None,label_path=None):
        img, labels, img_id = self.load_data(sample['file_name'], is_stream=False)
        ### BEGIN USAGE
        classes = [obj['category_id'] for obj in labels]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]
        entry = {
            "img": img,
            "labels": labels,
            "img_id": img_id,
            "usage": sample.get("usage", 0),
            "classes": list(set(classes)) if len(classes) else []
        }
        ### END USAGE

        if sample.get('klass', None):
            self.cls_count[self.new_exposed_classes.index(sample['klass'])] += 1
            sample_category = sample['klass']
        elif sample.get('domain', None):
            self.cls_count[self.new_exposed_classes.index(sample['domain'])] += 1
            sample_category = sample['domain']
        else:
            self.cls_count[self.new_exposed_classes.index('pretrained')] += 1
            sample_category = 'pretrained'
        # self.cls_count[self.new_exposed_classes.index(sample.get('klass', 'pretrained'))] += 1
        if idx is None:
            self.cls_idx[self.new_exposed_classes.index(sample_category)].append(len(self.buffer))
            self.buffer.append(entry)
        else:
            self.buffer[idx] = entry
    
    def register_stream(self, datalist):
        self.stream_data = []
        for data in datalist:
            data_class = data.get('label', None)
            img_path = data.get('file_name', data.get('filepath'))
            
            img, labels, img_id = self.load_data(img_path, is_stream=True, data_class=data_class)
            classes = [obj['category_id'] for obj in labels]
            classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                    for c in classes]
            entry = {
                "img": img,
                "labels": None,
                "img_id": img_id,
                "usage": 0,
                "classes": []
            }
            
            self.stream_data.append(entry)

    @torch.no_grad()
    def get_batch(self, batch_size, stream_batch_size=0, use_weight=None, transform=None, weight_method= "cls_usage",
                  model=None, score_thresh=0.7
                  ):
        assert batch_size >= stream_batch_size
        stream_batch_size = min(stream_batch_size, len(self.stream_data))
        batch_size = min(batch_size, stream_batch_size + len(self.buffer))
        memory_batch_size = batch_size - stream_batch_size

        data = []
        
        # append stream data to buffer for batch creation
        buffer_size = len(self.buffer)
        self.buffer.extend(self.stream_data)
 
        if stream_batch_size > 0:
            stream_indices = np.random.choice(range(len(self.stream_data)), size=stream_batch_size, replace=False)
            for i in stream_indices:
                if self.use_mosaic_mixup:
                    img, label, img_id = self.mosaic_wrapper.__getitem__((True, i + buffer_size, model, score_thresh))
                else:
                    entry = self.buffer[i + buffer_size]
                    img, anno, img_id = entry['img'], entry['labels'], entry['img_id']
                    if anno is not None:
                        anno = [obj for obj in anno if obj['iscrowd'] == 0]

                        boxes = [obj['bbox'] for obj in anno]
                        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

                        classes = [obj['category_id'] for obj in anno]
                        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                                for c in classes]

                        classes = torch.tensor(classes)
                        target.add_field('labels', classes)


                        target = target.clip_to_image(remove_empty=True)
                        
                        img = np.asarray(img)
                        img, label = self._transforms(img, target)
                    else:
                        # boxes, labels, scores = generate_pseudo_labels(model, img, score_thresh=score_thresh, transform=self.test_transform, image_sizes=self.image_sizes, device=self.device)
                        boxes, labels, scores = generate_pseudo_labels_tta_mc_ugpl_first(model, img, device=self.device)

                        if len(boxes) > 0:
                            target = BoxList(torch.tensor(boxes), img.size, mode='xyxy')
                            target.add_field('labels', torch.tensor(labels))
                            target = target.clip_to_image(remove_empty=True)

                            img, label = self._transforms(np.asarray(img), target)
                        else:
                            # no valid boxes
                            # create empty label
                            target = BoxList(torch.zeros((0,4)), img.size, mode='xyxy')
                            target.add_field('labels', torch.tensor([]))
                            img, label = self._transforms(np.asarray(img), target)
                        scores = torch.tensor(scores)
                
                data.append((img, label, img_id, scores))

        # ───── Memory part ──────────────────────────────────────────────
        if memory_batch_size > 0 and len(self.buffer):
            for e in self.buffer:
                e['usage'] *= self.usage_decay
            for cls_idx in range(len(self.cls_train_cnt)):
                self.cls_train_cnt[cls_idx] *= self.usage_decay

            ### HYBRID WEIGHT BEGIN
            if weight_method == "cls_usage":          # new option
                # 1 / (usage+α)  ×  1 / (mean cls_trained + β)
                alpha = getattr(self, "alpha", 1.0)
                beta  = getattr(self, "beta", 1.0)       # you may set self.beta in __init__
                weights = []
                for entry in self.buffer:
                    u = entry["usage"]
                    # gather per-image class-trained counts
                    if entry["classes"]:
                        t = [self.cls_train_cnt[int(c)]    # safe: cls_dict maps real IDs
                            for c in entry["classes"]               # (skip if not yet in dict)
                            if c < len(self.cls_train_cnt)]
                        mean_t = np.mean(t) if t else 0.0
                    else:
                        mean_t = 0.0        # no GT boxes → neutral
                    weights.append(1.0 / (u + alpha) * 1.0 / (mean_t + beta))
                w = np.asarray(weights, dtype=np.float64)
                w /= w.sum()
            else:
                w = np.array([1.0 / (e["usage"] + self.alpha) for e in self.buffer],
                            dtype=np.float64)
                w /= w.sum()
            ### HYBRID WEIGHT END

            indices = np.random.choice(
                len(self.buffer),
                size=memory_batch_size,
                replace=len(self.buffer) < memory_batch_size,
                p=w,
            )

            for i in indices:
                # update usage counter *and* class-train counts
                self.buffer[i]["usage"] += 1
                for idx_cls in self.buffer[i]["classes"]:
                    if idx_cls < len(self.cls_train_cnt):
                        self.cls_train_cnt[int(idx_cls)] += 1

                if self.use_mosaic_mixup:
                    img, label, img_id = self.mosaic_wrapper.__getitem__((True, i, model, score_thresh))
                else:
                    entry = self.buffer[i]
                    img, label, img_id = entry['img'], entry['labels'], entry['img_id']
                    if label is not None:
                        label = [obj for obj in label if obj['iscrowd'] == 0]

                        boxes = [obj['bbox'] for obj in label]
                        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

                        classes = [obj['category_id'] for obj in label]
                        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                                for c in classes]

                        classes = torch.tensor(classes)
                        target.add_field('labels', classes)


                        target = target.clip_to_image(remove_empty=True)
                        
                        img = np.asarray(img)
                        img, label = self._transforms(img, target)
                        scores = torch.ones(len(boxes))
                    else:
                        # boxes, labels, scores = generate_pseudo_labels(model, img, score_thresh=score_thresh,transform=self.test_transform, image_sizes=self.image_sizes, device=self.device)
                        boxes, labels, scores = generate_pseudo_labels_tta_mc_ugpl_first(model, img, device=self.device)

                        if len(boxes) > 0:
                            target = BoxList(torch.tensor(boxes), img.size, mode='xyxy')
                            target.add_field('labels', torch.tensor(labels))
                            target = target.clip_to_image(remove_empty=True)

                            img, label = self._transforms(np.asarray(img), target)
                        else:
                            # no valid boxes
                            # create empty label
                            target = BoxList(torch.zeros((0,4)), img.size, mode='xyxy')
                            target.add_field('labels', torch.tensor([]))
                            img, label = self._transforms(np.asarray(img), target)
                        scores = torch.tensor(scores)
                data.append((img, label, img_id, scores))

         # remove stream data from buffer
        self.buffer = self.buffer[:buffer_size]
        
        return self.batch_collator(data)


# Harmonious Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def draw_boxes_on_image(img, boxes, labels, scores=None, save_path="output.jpg"):
    """
    img: PIL.Image or numpy array
    boxes: [N, 4] numpy array, (x1, y1, x2, y2)
    labels: [N] numpy array
    scores: [N] numpy array or None
    save_path: str, where to save the image
    """
    # PIL 변환
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1

        # 박스 그리기
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                 edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        # 텍스트(label + score)
        caption = f"{labels[i]}"
        if scores is not None:
            caption += f" ({scores[i]:.2f})"
        ax.text(x1, y1 - 5, caption, color='yellow',
                fontsize=10, weight='bold',
                bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved with boxes at {save_path}")


def generate_pseudo_labels_harmonious(model, img, score_thresh=0.7, transform=None,image_sizes=(640,640),device='cuda'):
    # generate pseudo labels
    model.eval()
    img_cv = np.asarray(img)
    height, width = img_cv.shape[:2]
    if transform is not None:
        img_tensor = transform(img_cv)[0].unsqueeze(0).to(device)
    else:
        img_tensor = T.ToTensor()(img_cv)[0].unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)[0]

    boxes = outputs.bbox.detach().to('cpu').float().numpy()
    labels = outputs.extra_fields['labels'].detach().to('cpu').long().numpy()
    scores = outputs.extra_fields.get('scores').detach().to('cpu').float().numpy()
    
    
    # rescale boxes to original image size
    boxes[:, 0] = boxes[:, 0] * width / image_sizes[1]
    boxes[:, 1] = boxes[:, 1] * height / image_sizes[0]
    boxes[:, 2] = boxes[:, 2] * width / image_sizes[1]
    boxes[:, 3] = boxes[:, 3] * height / image_sizes[0]
    
    # filter boxes by score threshold
    keep_indices = np.where(scores >= score_thresh)[0]
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    labels = labels[keep_indices]
    
    return boxes, labels, scores



class HarmoniousDataset(MemoryDataset):
    def __init__(self, ann_file, root, transforms=None, class_names=None,
                 dataset=None, cls_list=None, device=None, data_dir=None, memory_size=None, 
                 init_buffer_size=None, image_size=(640, 640), aug=None
                 ):
        super(MemoryDataset, self).__init__(ann_file, root, transforms, class_names)
        # self.args = args
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_sizes = [int(image_size[0]), int(image_size[1])]
        self.memory_size = memory_size

        self.buffer = []
        self.stream_data = []
        

        self.dataset = dataset
        self.device = device
        self.data_dir = data_dir

        
        self.class_usage_cnt = []
        
        
        # FIXME: fix for object detection class counting
        self.cls_list = cls_list if cls_list else []
        
        
        self.cls_count = [0]
        self.cls_idx = [[]]
        
        self.new_exposed_classes = ['pretrained']
        # self.cls_train_cnt = np.array([0,]*len(cls_list)) if cls_list else np.array([])
        self.cls_train_cnt = np.array([])
        
        
        
        
        
        
        
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
            self.name2id = {re.sub(r'\.(jpg|jpeg|png)$', '', ann['file_name'], flags=re.I): ann['id'] for ann in annotations['images']}
            
        if 'HS_TOD' not in self.dataset:
            self.build_initial_buffer(init_buffer_size)

        n_classes, image_dir, label_path = get_statistics(dataset=self.dataset)
        self.image_dir = image_dir
        self.label_path = label_path
        
        # set transform
        transforms = aug.transform
        transforms = build_transforms_memorydataset(**transforms)
        self._transforms = transforms
        
        # aug.mosaic_mixup.mosaic_prob = 0
        # aug.mosaic_mixup.mixup_prob = 0
        # aug.mosaic_mixup.degrees = 0
        # aug.mosaic_mixup.translate = 0
        # aug.mosaic_mixup.mosaic_scale = (0.8, 1.2)
        
        print("mosaic_prob", aug.mosaic_mixup.mosaic_prob)
        print("mixup_prob", aug.mosaic_mixup.mixup_prob)
        print("degrees", aug.mosaic_mixup.degrees)
        print("translate", aug.mosaic_mixup.translate)
        print("mosaic_scale", aug.mosaic_mixup.mosaic_scale)
        
        self.mosaic_wrapper = MosaicWrapper(
            dataset=self,                      # <-- this class will provide pull_item/load_anno
            img_size=image_size,
            mosaic_prob=aug.mosaic_mixup.mosaic_prob,                   # always “allowed”; actual apply is still random inside wrapper
            mixup_prob=aug.mosaic_mixup.mixup_prob,
            transforms=self._transforms,                   # we'll run your own transforms after mosaic
            degrees=aug.mosaic_mixup.degrees,
            translate=aug.mosaic_mixup.translate,
            mosaic_scale=aug.mosaic_mixup.mosaic_scale,
            keep_ratio=True,
        )
        self.use_mosaic_mixup=False 
        
        
        self.test_transform = T.Compose(
            [
                T.Resize(image_size, keep_ratio=False),
                T.ToTensor(),
                T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ]
        )
        
        
        self.batch_collator = HarmoniousBatchCollator(size_divisible=32)
        
        self.alpha = 1.0                  # smoothing constant in 1/(usage+α)
        self.beta = 1.0
        self.usage_decay = 0.995
        
        
    def load_valid_labels(self, label, is_stream, data_class):
        indices_to_keep = []
        
        for i in range(len(label)):
            if is_stream:
                pass
            else:
                if self.dataset == 'VOC_10_10':
                    if self.contiguous_class2id[self.ori_id2class[label[i]['category_id']]] < 10:
                        indices_to_keep.append(i)
                elif self.dataset == 'VOC_15_5':
                    if self.contiguous_class2id[self.ori_id2class[label[i]['category_id']]] < 15:
                        indices_to_keep.append(i)
                else:
                    indices_to_keep.append(i)
                    
                    
        label = [label[i] for i in indices_to_keep]
        
        if len(label) == 0:
            return None
        else:
            return label
        
    def load_data(self, img_path, is_stream, data_class=None):
        img_path = re.sub(r'\.(jpg|jpeg|png|JPG|JPEG|PNG)$', '', img_path)
        try:
            image_id = self.name2id[img_path]
        except:
            image_id = self.name2id[img_path.split('/')[-1]]
        idx = self.ids.index(image_id)
        img, label, img_id = self.__getitem__(idx)
    
        label = self.load_valid_labels(label, is_stream, data_class=data_class)
        
        return img, label, img_id
    
        
    def replace_sample(self, sample, idx=None, images_dir=None,label_path=None):
        data_class = sample.get('label', None)
        img, labels, img_id = self.load_data(sample['file_name'], is_stream=True,data_class=data_class)
        classes = []
        if labels is not None: # if data is not stream  
            classes = [obj['category_id'] for obj in labels]
            classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                    for c in classes]
        ### BEGIN USAGE
        entry = {
            "img": img,
            "labels": labels,
            "img_id": img_id,
            "usage": sample.get("usage", 0),
            "classes": list(set(classes)) if len(classes) else [],
            "stream": True
        }
        ### END USAGE

        if sample.get('klass', None):
            self.cls_count[self.new_exposed_classes.index(sample['klass'])] += 1
            sample_category = sample['klass']
        elif sample.get('domain', None):
            self.cls_count[self.new_exposed_classes.index(sample['domain'])] += 1
            sample_category = sample['domain']
        else:
            self.cls_count[self.new_exposed_classes.index('pretrained')] += 1
            sample_category = 'pretrained'
        # self.cls_count[self.new_exposed_classes.index(sample.get('klass', 'pretrained'))] += 1
        if idx is None:
            self.cls_idx[self.new_exposed_classes.index(sample_category)].append(len(self.buffer))
            self.buffer.append(entry)
        else:
            self.buffer[idx] = entry
    
    def register_sample_for_initial_buffer(self, sample, idx=None, images_dir=None,label_path=None):
        img, labels, img_id = self.load_data(sample['file_name'], is_stream=False)
        
        
        ### BEGIN USAGE
        
        classes = []
        if labels is not None: # if data is not stream  
            classes = [obj['category_id'] for obj in labels]
            classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                    for c in classes]
            
        entry = {
            "img": img,
            "labels": labels,
            "img_id": img_id,
            "usage": sample.get("usage", 0),
            "classes": list(set(classes)) if len(classes) else [],
            "stream": False
        }
        ### END USAGE

        if sample.get('klass', None):
            self.cls_count[self.new_exposed_classes.index(sample['klass'])] += 1
            sample_category = sample['klass']
        elif sample.get('domain', None):
            self.cls_count[self.new_exposed_classes.index(sample['domain'])] += 1
            sample_category = sample['domain']
        else:
            self.cls_count[self.new_exposed_classes.index('pretrained')] += 1
            sample_category = 'pretrained'
        # self.cls_count[self.new_exposed_classes.index(sample.get('klass', 'pretrained'))] += 1
        if idx is None:
            self.cls_idx[self.new_exposed_classes.index(sample_category)].append(len(self.buffer))
            self.buffer.append(entry)
            
        else:
            self.buffer[idx] = entry
    
    def register_stream(self, datalist):
        self.stream_data = []
        for data in datalist:
            data_class = data.get('label', None)
            img_path = data.get('file_name', data.get('filepath'))
            
            img, labels, img_id = self.load_data(img_path, is_stream=True, data_class=data_class)
            
            classes = []
            if labels is not None: # if data is not stream
                classes = [obj['category_id'] for obj in labels]
                classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                        for c in classes]
                
            entry = {
                "img": img,
                "labels": labels,
                "img_id": img_id,
                "usage": 0,
                "classes": list(set(classes)) if len(classes) else [],
                "stream": True
            }
            self.stream_data.append(entry)

    @torch.no_grad()
    def get_batch(self, batch_size, stream_batch_size=0, use_weight=None, transform=None, weight_method= "cls_usage", model=None, score_thresh=0.7):
        assert batch_size >= stream_batch_size
        stream_batch_size = min(stream_batch_size, len(self.stream_data))
        batch_size = min(batch_size, stream_batch_size + len(self.buffer))
        memory_batch_size = batch_size - stream_batch_size

        data = []
        
        # append stream data to buffer for batch creation
        buffer_size = len(self.buffer)
        self.buffer.extend(self.stream_data)
 
        if stream_batch_size > 0:
            stream_indices = np.random.choice(range(len(self.stream_data)), size=stream_batch_size, replace=False)
            for i in stream_indices:
                if self.use_mosaic_mixup:
                    img, label, img_id = self.mosaic_wrapper.__getitem__((True, i + buffer_size))
                else:
                    entry = self.buffer[i + buffer_size]
                    img, anno, img_id = entry['img'], entry['labels'], entry['img_id']
                    
                    if anno is not None: # will not be executed as target data is unlabeled
                        
                        anno = [obj for obj in anno if obj['iscrowd'] == 0]

                        boxes = [obj['bbox'] for obj in anno]
                        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

                        classes = [obj['category_id'] for obj in anno]
                        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                                for c in classes]

                        classes = torch.tensor(classes)
                        target.add_field('labels', classes)

                        target = target.clip_to_image(remove_empty=True)
                        
                        img = np.asarray(img)
                        img, label = self._transforms(img, target)
                        
                        
                    else:
                        boxes, labels, scores = generate_pseudo_labels_harmonious(model, img, score_thresh=score_thresh, transform=self.test_transform, image_sizes=self.image_sizes, device=self.device)
                        # boxes, labels, scores = generate_pseudo_labels_tta_mc_ugpl_first(model, img, device=self.device)
                        
                        if len(boxes) > 0:
                            target = BoxList(torch.tensor(boxes), img.size, mode='xyxy')
                            target.add_field('labels', torch.tensor(labels))
                            target = target.clip_to_image(remove_empty=True)

                            img, label = self._transforms(np.asarray(img), target)
                        else:
                            # no valid boxes
                            # create empty label
                            target = BoxList(torch.zeros((0,4)), img.size, mode='xyxy')
                            target.add_field('labels', torch.tensor([]))
                            img, label = self._transforms(np.asarray(img), target)
                            
                        score_weight = torch.tensor(scores)
                
                # data.append((img, label, img_id))
                data.append((img, label, img_id, score_weight))

        # ───── Memory part ──────────────────────────────────────────────
        if memory_batch_size > 0 and len(self.buffer):
            indices = np.random.choice(range(buffer_size), size=memory_batch_size, replace=False)
            for i in indices:
                if self.use_mosaic_mixup:
                    img, label, img_id = self.mosaic_wrapper.__getitem__((True, i, model, score_thresh))
                else:
                    entry = self.buffer[i]
                    img, anno, img_id = entry['img'], entry['labels'], entry['img_id']
                    
                    if anno is not None:
                        anno = [obj for obj in anno if obj['iscrowd'] == 0]

                        boxes = [obj['bbox'] for obj in anno]
                        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                        
                        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

                        classes = [obj['category_id'] for obj in anno]
                        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                                for c in classes]

                        classes = torch.tensor(classes)
                        target.add_field('labels', classes)

                        target = target.clip_to_image(remove_empty=True)
                        
                        img = np.asarray(img)
                        img, label = self._transforms(img, target)
                        
                        score_weight = torch.ones(len(boxes))

                        
                    else: 
                        boxes, labels, scores = generate_pseudo_labels_harmonious(model, img, score_thresh=score_thresh, transform=self.test_transform, image_sizes=self.image_sizes, device=self.device)
                        # boxes, labels, scores = generate_pseudo_labels_tta_mc_ugpl_first(model, img, device=self.device)
                        
                        if len(boxes) > 0:
                            target = BoxList(torch.tensor(boxes), img.size, mode='xyxy')
                            target.add_field('labels', torch.tensor(labels))
                            target = target.clip_to_image(remove_empty=True)

                            img, label = self._transforms(np.asarray(img), target)
                        else:
                            # no valid boxes
                            # create empty label
                            target = BoxList(torch.zeros((0,4)), img.size, mode='xyxy')
                            target.add_field('labels', torch.tensor([]))
                            img, label = self._transforms(np.asarray(img), target)
                            
                        score_weight = torch.tensor(scores)
                
                # data.append((img, label, img_id))    
                data.append((img, label, img_id, score_weight))

                            

         # remove stream data from buffer
        self.buffer = self.buffer[:buffer_size]
        
        return self.batch_collator(data)

    # # might need to modify for pseudolabel
    # def pull_item(self, idx):
    #     entry = self.buffer[idx]
    #     img, anno, img_id = entry['img'], entry['labels'], entry['img_id']
    #     # filter crowd annotations
    #     # TODO might be better to add an extra field
    #     anno = [obj for obj in anno if obj['iscrowd'] == 0]

    #     boxes = [obj['bbox'] for obj in anno]
    #     boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
    #     target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')
        
    #     target = target.clip_to_image(remove_empty=True)

    #     classes = [obj['category_id'] for obj in anno]
    #     classes = [self.contiguous_class2id[self.ori_id2class[c]] 
    #                for c in classes]

    #     obj_masks = []
    #     for obj in anno:
    #         obj_mask = []
    #         if 'segmentation' in obj:
    #             for mask in obj['segmentation']:
    #                 obj_mask += mask
    #             if len(obj_mask) > 0:
    #                 obj_masks.append(obj_mask)
    #     seg_masks = [
    #         np.array(obj_mask, dtype=np.float32).reshape(-1, 2)
    #         for obj_mask in obj_masks
    #     ]

    #     res = np.zeros((len(target.bbox), 5))
    #     for idx in range(len(target.bbox)):
    #         res[idx, 0:4] = target.bbox[idx]
    #         res[idx, 4] = classes[idx]

    #     img = np.asarray(img)  # rgb

    #     return img, res, seg_masks, img_id
    
    def load_anno(self, idx):
        entry = self.buffer[idx]
        anno = entry['labels']
        anno = [obj for obj in anno if obj['iscrowd'] == 0]
        classes = [obj['category_id'] for obj in anno]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]
        return classes

