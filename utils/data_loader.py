import logging.config
import random
import os
import json
from typing import List
import copy
import PIL
import math
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
from collections import Counter
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn as nn
import torch.multiprocessing as multiprocessing

# from utils.data_worker import worker_loop
import glob

logger = logging.getLogger()


from statistics import mean
from typing import Generator, List, Tuple, Union
from torch import Tensor
from pathlib import Path
from damo.dataset.datasets.mosaic_wrapper import MosaicWrapper
from damo.dataset.datasets.coco import COCODataset
from damo.dataset.collate_batch import BatchCollator
from damo.dataset.transforms import build_transforms, build_transforms_memorydataset
from damo.structures.bounding_box import BoxList

from collections import defaultdict
import cv2

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
    elif dataset == 'BDD_domain':
        return 13, 'data/bdd100k/images', 'data/bdd100k/annotations'
    elif dataset == 'SHIFT_domain':
        return 6, 'data/shift/images', 'data/shift/annotations'
    elif 'MILITARY_SYNTHETIC_domain' in dataset:
        return 9, 'data/military_synthetic/images', 'data/military_synthetic/annotations'
    else:
        raise ValueError("Wrong dataset name")

def get_pretrained_statistics(dataset: str):
    if dataset == 'VOC_10_10':
        return 10, 'data/voc_10/images', 'data/voc_10/annotations' ## 경로 수정
    elif dataset == 'BDD_domain':
        return 13, 'data/bdd100k_source/images', 'data/bdd100k_source/annotations'
    elif dataset == 'SHIFT_domain':
        return 6, 'data/shift_source/images', 'data/shift_source/annotations'
    elif 'MILITARY_SYNTHETIC_domain' in dataset:
        return 9, 'data/military_synthetic_domain_source/images', 'data/military_synthetic_domain_source/annotations'
    else:
        raise ValueError("Wrong dataset name")
    
def get_exposed_classes(dataset: str):
    if dataset == 'VOC_10_10':
        return ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow']
    elif dataset == 'BDD_domain':
        return ['pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'traffic light', 'traffic sign', 'train', 'trailer', 'other person', 'other vehicle']
    elif dataset == 'SHIFT_domain':
        return ['pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']
    elif 'MILITARY_SYNTHETIC_domain' in dataset:
        return ['fishing vessel', 'warship', 'merchant vessel', 'fixed-wing aircraft', 'rotary-wing aircraft', 'Unmanned Aerial Vehicle', 'bird', 'leaflet', 'waste bomb']
        
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
        self.logits = []

        self.dataset = dataset
        self.device = device
        self.data_dir = data_dir

        self.counts = []
        self.class_usage_cnt = []
        self.tasks = []
        
        # FIXME: fix for object detection class counting
        self.cls_list = cls_list if cls_list else []
        self.cls_used_times = []
        self.cls_dict = {}
        self.cls_count = []
        self.cls_idx = []
        self.cls_train_cnt = np.array([0,]*len(cls_list)) if cls_list else np.array([])
        self.score = []
        self.others_loss_decrease = np.array([])
        self.previous_idx = np.array([], dtype=int)
        self.usage_cnt = []
        self.sample_weight = []
        self.data = {}
        
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
            self.name2id = {ann['file_name'].split('.')[0]: ann['id'] for ann in annotations['images']}
        
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
        self.use_mosaic_mixup=True 
        
        self.batch_collator = BatchCollator(size_divisible=32)
    
    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, inp):
        if type(inp) is tuple:
            idx = inp[1]
        else:
            idx = inp
        img, anno = super(COCODataset, self).__getitem__(idx)
        # PIL to numpy array
        return img, anno, idx

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
    
    def load_valid_labels(self, label, is_stream, data_class):
        indices_to_keep = []
        
        for i in range(len(label)):
            if is_stream:
                if self.dataset == 'VOC_10_10':
                    if self.contiguous_class2id[self.ori_id2class[label[i]['category_id']]] <= data_class:# == data_class: # VOC_10_10 labels start from 0
                        indices_to_keep.append(i)
                    # if self.contiguous_class2id[self.ori_id2class[label[i]['category_id']]] < len(self.cls_list): # VOC_10_10 labels start from 0
                    #     indices_to_keep.append(i)
                else:
                    indices_to_keep.append(i)
            else:
                if self.dataset == 'VOC_10_10':
                    if self.contiguous_class2id[self.ori_id2class[label[i]['category_id']]] < 10:
                        indices_to_keep.append(i)
                else:
                    indices_to_keep.append(i)
        label = [label[i] for i in indices_to_keep]
        
        return label
    
    def build_initial_buffer(self, buffer_size=None):
        n_classes, images_dir, label_path = get_pretrained_statistics(self.dataset)
        self.image_dir = images_dir
        self.label_path = label_path
        if self.dataset == 'VOC_10_10':
            image_files = glob.glob(os.path.join(images_dir, "train","*/*.jpg"))
        else:
            image_files = glob.glob(os.path.join(images_dir, "train","*.jpg"))

        indices = np.random.choice(range(len(image_files)), size=buffer_size or self.memory_size, replace=False)

        for idx in indices:
            image_path = image_files[idx]
            split_name = image_path.split('/')[-2]
            base_name = image_path.split('/')[-1]
            self.register_sample_for_initial_buffer({'file_name': split_name + '/' + base_name, 'label': None}, images_dir=images_dir,label_path=label_path)    
            

    def replace_sample(self, sample, idx=None, images_dir=None,label_path=None):
        data_class = sample.get('label', None)
        img, labels, img_id = self.load_data(sample['file_name'], is_stream=True,data_class=data_class)
        data = (img, labels, img_id)

        if idx is None:
            self.buffer.append(data)
        else:
            self.buffer[idx] = data
            
    def register_sample_for_initial_buffer(self, sample, idx=None, images_dir=None,label_path=None):
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
    def get_batch(self, batch_size, stream_batch_size=0, use_weight=None, transform=None, weight_method=None):
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

    def get_stream_batch(self, batch_size):
        if not hasattr(self, 'stream_data') or len(self.stream_data) == 0:
            raise ValueError("stream_data is empty. Cannot get a stream batch.")

        current_batch_size = min(batch_size, len(self.stream_data))
        stream_indices = np.random.choice(range(len(self.stream_data)), size=current_batch_size, replace=False)
        
        data_to_collate = []
        for i in stream_indices:
            buffer_size = len(self.buffer)
            self.buffer.append(self.stream_data[i])
            
            img, label, img_id = self.mosaic_wrapper.__getitem__((True, buffer_size))
            data_to_collate.append((img, label, img_id))

            self.buffer.pop()

        return self.batch_collator(data_to_collate)
    
    def add_new_class(self, cls_list, sample=None):
        self.cls_list = cls_list
        self.cls_count.append(0)
        # self.cls_loss.append(None)
        # if sample is not None:
            # self.cls_times.append(sample['time'])
        # else:
            # self.cls_times.append(None)
        #self.cls_used_times.append(max(self.cls_times))
        # self.cls_weight.append(1)
        self.cls_idx.append([])
        self.class_usage_cnt.append(0)
        self.cls_dict = {self.cls_list[i]:i for i in range(len(self.cls_list))}
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
    
    
def get_train_datalist(dataset, sigma, repeat, init_cls, rnd_seed):
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

    def replace_sample(self, sample, idx=None, images_dir=None,label_path=None):
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
    def get_batch(self, batch_size, stream_batch_size=0, use_weight=None, transform=None, weight_method=None):
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
        # self.args = args
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_sizes = [int(image_size[0]), int(image_size[1])]
        self.memory_size = memory_size

        self.buffer = []
        self.stream_data = []
        self.logits = []

        self.dataset = dataset
        self.device = device
        self.data_dir = data_dir

        self.counts = []
        self.class_usage_cnt = []
        self.tasks = []
        
        # FIXME: fix for object detection class counting
        self.cls_list = cls_list if cls_list else []
        self.cls_used_times = []
        self.cls_dict = {}
        self.cls_count = [0]
        self.cls_idx = [[]]
        
        self.new_exposed_classes = ['pretrained']
        # self.cls_train_cnt = np.array([0,]*len(cls_list)) if cls_list else np.array([])
        self.cls_train_cnt = np.array([])
        self.score = []
        self.others_loss_decrease = np.array([])
        self.previous_idx = np.array([], dtype=int)
        self.usage_cnt = []
        self.sample_weight = []
        self.data = {}
        
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
            self.name2id = {ann['file_name'].split('.')[0]: ann['id'] for ann in annotations['images']}
        
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
        self.use_mosaic_mixup=True 
        
        self.batch_collator = BatchCollator(size_divisible=32)

    def add_new_class(self, cls_list, sample=None):
        self.cls_list = cls_list
        self.cls_count.append(0)
        self.cls_idx.append([])

    def replace_sample(self, sample, idx=None, images_dir=None,label_path=None):
        data_class = sample.get('label', None)
        img, labels, img_id = self.load_data(sample['file_name'], is_stream=True,data_class=data_class)
        data = (img, labels, img_id)

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
            self.buffer.append(data)
        else:
            self.buffer[idx] = data
    def register_sample_for_initial_buffer(self, sample, idx=None, images_dir=None, label_path=None):
        data_class = sample.get('label', None)
        img, labels, img_id = self.load_data(sample['file_name'], is_stream=False)
        data = (img, labels, img_id)
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
            self.buffer.append(data)
        else:
            self.buffer[idx] = data

#################################################################################################
# Frequency + class balanced dataset

# Frequency-based Dataset
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
        self.logits = []

        self.dataset = dataset
        self.device = device
        self.data_dir = data_dir

        self.counts = []
        self.class_usage_cnt = []
        self.tasks = []
        
        # FIXME: fix for object detection class counting
        self.cls_list = cls_list if cls_list else []
        self.cls_used_times = []
        self.cls_dict = {}
        self.cls_count = [0]
        self.cls_idx = [[]]
        
        self.new_exposed_classes = ['pretrained']
        # self.cls_train_cnt = np.array([0,]*len(cls_list)) if cls_list else np.array([])
        self.cls_train_cnt = np.array([])
        self.score = []
        self.others_loss_decrease = np.array([])
        self.previous_idx = np.array([], dtype=int)
        self.usage_cnt = []
        self.sample_weight = []
        self.data = {}
        
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
            self.name2id = {ann['file_name'].split('.')[0]: ann['id'] for ann in annotations['images']}
        
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
        self.use_mosaic_mixup=True 
        
        self.batch_collator = BatchCollator(size_divisible=32)
        
        self.alpha = 1.0                  # smoothing constant in 1/(usage+α)
        self.beta = 1.0
        self.usage_decay = 0.9
    
        
    def replace_sample(self, sample, idx=None, images_dir=None,label_path=None):
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
                "labels": labels,
                "img_id": img_id,
                "usage": 0,
                "classes": list(set(classes)) if len(classes) else []
            }
            
            self.stream_data.append(entry)

    @torch.no_grad()
    def get_batch(self, batch_size, stream_batch_size=0, use_weight=None, transform=None, weight_method=None):
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
#################################################################################################
########## Below is original code that we don't use now



class MultiProcessLoader():
    def __init__(self, n_workers, cls_dict, transform, data_dir, transform_on_gpu=False, cpu_transform=None, device='cpu', use_kornia=False, transform_on_worker=True, init=True):
        self.n_workers = n_workers
        self.cls_dict = cls_dict
        self.transform = transform
        self.transform_on_gpu = transform_on_gpu
        self.transform_on_worker = transform_on_worker
        self.use_kornia = use_kornia
        self.cpu_transform = cpu_transform
        self.device = device
        self.result_queues = []
        self.workers = []
        self.index_queues = []
        if init:
            for i in range(self.n_workers):
                index_queue = multiprocessing.Queue()
                index_queue.cancel_join_thread()
                result_queue = multiprocessing.Queue()
                result_queue.cancel_join_thread()
                w = multiprocessing.Process(target=worker_loop, args=(index_queue, result_queue, data_dir, self.transform, self.transform_on_gpu, self.cpu_transform, self.device, use_kornia, transform_on_worker))
                w.daemon = True
                w.start()
                self.workers.append(w)
                self.index_queues.append(index_queue)
                self.result_queues.append(result_queue)
        if self.use_kornia:
            if 'cifar100' in data_dir:
                mean, std, n_classes, inp_size, _ = get_statistics(dataset='cifar100')
            elif 'cifar10' in data_dir:
                mean, std, n_classes, inp_size, _ = get_statistics(dataset='cifar10')
            elif 'tinyimagenet' in data_dir:
                mean, std, n_classes, inp_size, _ = get_statistics(dataset='tinyimagenet')
            elif 'imagenet' in data_dir:
                mean, std, n_classes, inp_size, _ = get_statistics(dataset='imagenet')
            self.transform = DataAugmentation(inp_size, mean, std)

    def add_new_class(self, cls_dict):
        self.cls_dict = cls_dict

    def load_batch(self, batch):
        for sample in batch:
            sample["label"] = self.cls_dict[sample["klass"]]
            
        for i in range(self.n_workers):
            self.index_queues[i].put(batch[len(batch)*i//self.n_workers:len(batch)*(i+1)//self.n_workers])

    @torch.no_grad()
    def get_batch(self):
        data = dict()
        images = []
        labels = []
        for i in range(self.n_workers):
            loaded_samples = self.result_queues[i].get(timeout=3000.0)
            if loaded_samples is not None:
                images.append(loaded_samples["image"])
                labels.append(loaded_samples["label"])
        if len(images) > 0:
            images = torch.cat(images)
            labels = torch.cat(labels)
            if self.transform_on_gpu and not self.transform_on_worker:
                images = self.transform(images.to(self.device))
            data['image'] = images
            data['label'] = labels
            return data
        else:
            return None
    
    # @torch.no_grad()
    # def get_batch(self):
    #     data = dict()
    #     images = []
    #     labels = []
    #     for i in range(self.n_workers):
    #         loaded_samples = self.result_queues[i].get(timeout=3000.0)
    #         # if loaded_samples is not None:
    #         images.append(torch.rand(4, 3, 32, 32).to(self.device))
    #         labels.append(torch.LongTensor([0, 0, 0, 0]).to(self.device))
    #     if len(images) > 0:
    #         images = torch.cat(images)
    #         labels = torch.cat(labels)
    #         if self.transform_on_gpu and not self.transform_on_worker:
    #             images = self.transform(images.to(self.device))
    #         data['image'] = images
    #         data['label'] = labels
    #         return data
    #     else:
    #         return None


class CCLDCLoader(MultiProcessLoader):
    def __init__(self, n_workers, cls_dict, transform, data_dir, transform_on_gpu=False, cpu_transform=None, device='cpu', use_kornia=False, transform_on_worker=True, test_transform=None, scl=False, init=True, transform_1=None, transform_2=None, transform_3=None, base_transform=None, normalize=None):
        super().__init__(n_workers, cls_dict, transform, data_dir, transform_on_gpu, cpu_transform, device, use_kornia, transform_on_worker, init=False)
        self.n_workers = n_workers
        self.cls_dict = cls_dict
        self.transform = transform
        self.transform_on_gpu = transform_on_gpu
        self.transform_on_worker = transform_on_worker
        self.use_kornia = use_kornia
        self.cpu_transform = cpu_transform
        self.device = device
        self.result_queues = []
        self.workers = []
        self.index_queues = []
        for i in range(self.n_workers):
            index_queue = multiprocessing.Queue()
            index_queue.cancel_join_thread()
            result_queue = multiprocessing.Queue()
            result_queue.cancel_join_thread()
            w = multiprocessing.Process(target=worker_loop, args=(index_queue, result_queue, data_dir, self.transform, self.transform_on_gpu, self.cpu_transform, self.device, use_kornia, transform_on_worker, test_transform, scl, transform_1, transform_2, transform_3, base_transform, normalize))
            w.daemon = True
            w.start()
            self.workers.append(w)
            self.index_queues.append(index_queue)
            self.result_queues.append(result_queue)

    @torch.no_grad()
    def get_batch(self):
        data = dict()
        images = []
        labels = []
        transform_1_image = []
        transform_2_image = []
        transform_3_image = []
        not_aug_image = []
        for i in range(self.n_workers):
            loaded_samples = self.result_queues[i].get(timeout=3000.0)
            if loaded_samples is not None:
                images.append(loaded_samples["image"])
                labels.append(loaded_samples["label"])
                transform_1_image.append(loaded_samples["transform_1_image"])
                transform_2_image.append(loaded_samples["transform_2_image"])
                transform_3_image.append(loaded_samples["transform_3_image"])
                not_aug_image.append(loaded_samples["not_aug_image"])
                
        if len(images) > 0:
            images = torch.cat(images)
            labels = torch.cat(labels)
            transform_1_image = torch.cat(transform_1_image)
            transform_2_image = torch.cat(transform_2_image)
            transform_3_image = torch.cat(transform_3_image)
            not_aug_image = torch.cat(not_aug_image)
            if self.transform_on_gpu and not self.transform_on_worker:
                images = self.transform(images.to(self.device))
            data['image'] = images
            data['label'] = labels
            data['transform_1_image'] = transform_1_image
            data['transform_2_image'] = transform_2_image
            data['transform_3_image'] = transform_3_image
            data['not_aug_image'] = not_aug_image
            return data
        else:
            return None


class XDERLoader(MultiProcessLoader):
    def __init__(self, n_workers, cls_dict, transform, data_dir, transform_on_gpu=False, cpu_transform=None, device='cpu', use_kornia=False, transform_on_worker=True, test_transform=None, scl=False, init=True):
        super().__init__(n_workers, cls_dict, transform, data_dir, transform_on_gpu, cpu_transform, device, use_kornia, transform_on_worker, init=False)
        self.n_workers = n_workers
        self.cls_dict = cls_dict
        self.transform = transform
        self.transform_on_gpu = transform_on_gpu
        self.transform_on_worker = transform_on_worker
        self.use_kornia = use_kornia
        self.cpu_transform = cpu_transform
        self.device = device
        self.result_queues = []
        self.workers = []
        self.index_queues = []
        for i in range(self.n_workers):
            index_queue = multiprocessing.Queue()
            index_queue.cancel_join_thread()
            result_queue = multiprocessing.Queue()
            result_queue.cancel_join_thread()
            w = multiprocessing.Process(target=worker_loop, args=(index_queue, result_queue, data_dir, self.transform, self.transform_on_gpu, self.cpu_transform, self.device, use_kornia, transform_on_worker, test_transform, scl))
            w.daemon = True
            w.start()
            self.workers.append(w)
            self.index_queues.append(index_queue)
            self.result_queues.append(result_queue)
        if self.use_kornia:
            if 'cifar100' in data_dir:
                mean, std, n_classes, inp_size, _ = get_statistics(dataset='cifar100')
            elif 'cifar10' in data_dir:
                mean, std, n_classes, inp_size, _ = get_statistics(dataset='cifar10')
            elif 'tinyimagenet' in data_dir:
                mean, std, n_classes, inp_size, _ = get_statistics(dataset='tinyimagenet')
            elif 'imagenet' in data_dir:
                mean, std, n_classes, inp_size, _ = get_statistics(dataset='imagenet')
            self.transform = DataAugmentation(inp_size, mean, std)

    @torch.no_grad()
    def get_batch(self):
        data = dict()
        images = []
        labels = []
        not_aug_img = []
        for i in range(self.n_workers):
            loaded_samples = self.result_queues[i].get(timeout=3000.0)
            if loaded_samples is not None:
                images.append(loaded_samples["image"])
                labels.append(loaded_samples["label"])
                if "not_aug_img" in loaded_samples.keys():
                    not_aug_img.append(loaded_samples["not_aug_img"])
        if len(images) > 0:
            images = torch.cat(images)
            labels = torch.cat(labels)
            if self.transform_on_gpu and not self.transform_on_worker:
                images = self.transform(images.to(self.device))
            data['image'] = images
            data['label'] = labels
            if "not_aug_img" in loaded_samples.keys():
                not_aug_img = torch.cat(not_aug_img)
                data['not_aug_img'] = not_aug_img
            return data
        else:
            return None
    


def nonzero_indices(bool_mask_tensor):
    # Returns tensor which contains indices of nonzero elements in bool_mask_tensor
    return bool_mask_tensor.nonzero(as_tuple=True)[0]

def partial_distill_loss(model, net_partial_features: list, pret_partial_features: list,
                         targets, device, teacher_forcing: list = None, extern_attention_maps: list = None):

    assert len(net_partial_features) == len(
        pret_partial_features), f"{len(net_partial_features)} - {len(pret_partial_features)}"

    if teacher_forcing is None or extern_attention_maps is None:
        assert teacher_forcing is None
        assert extern_attention_maps is None

    loss = 0
    attention_maps = []

    for i, (net_feat, pret_feat) in enumerate(zip(net_partial_features, pret_partial_features)):
        assert net_feat.shape == pret_feat.shape, f"{net_feat.shape} - {pret_feat.shape}"

        adapter = getattr(
            model, f"adapter_{i+1}")

        pret_feat = pret_feat.detach()

        if teacher_forcing is None:
            curr_teacher_forcing = torch.zeros(
                len(net_feat,)).bool().to(device)
            curr_ext_attention_map = torch.ones(
                (len(net_feat), adapter.c)).to(device)
        else:
            curr_teacher_forcing = teacher_forcing
            curr_ext_attention_map = torch.stack(
                [b[i] for b in extern_attention_maps], dim=0).float()

        adapt_loss, adapt_attention = adapter(net_feat, pret_feat, targets,
                                              teacher_forcing=curr_teacher_forcing, attention_map=curr_ext_attention_map)

        loss += adapt_loss
        attention_maps.append(adapt_attention.detach().cpu().clone().data)

    return loss / (i + 1), attention_maps

class Preprocess(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""
    def __init__(self, input_size=32):
        super().__init__()
        self.input_size = input_size

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x):
        x_tmp = np.array(x)  # HxWxC
        x_out = image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        x_out = resize(x_out.float() / 255.0, (self.input_size, self.input_size))
        return x_out

class ImageDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, dataset: str, transform=None, cls_list=None, data_dir=None,
                 preload=False, device=None, transform_on_gpu=False, use_kornia=False):
        self.use_kornia = use_kornia
        self.data_frame = data_frame
        self.dataset = dataset
        self.transform = transform
        self.cls_list = cls_list
        self.data_dir = data_dir
        self.preload = preload
        self.device = device
        self.transform_on_gpu = transform_on_gpu
        if self.preload:
            mean, std, n_classes, inp_size, _ = get_statistics(dataset=self.dataset)
            self.preprocess = Preprocess(input_size=inp_size)
            if self.transform_on_gpu:
                self.transform_cpu = transforms.Compose(
                    [
                        transforms.Resize((inp_size, inp_size)),
                        transforms.PILToTensor()
                    ])
                self.transform_gpu = self.transform
            self.loaded_images = []
            for idx in range(len(self.data_frame)):
                sample = dict()
                try:
                    img_name = self.data_frame.iloc[idx]["file_name"]
                except KeyError:
                    img_name = self.data_frame.iloc[idx]["filepath"]
                if self.cls_list is None:
                    label = self.data_frame.iloc[idx].get("label", -1)
                else:
                    label = self.cls_list.index(self.data_frame.iloc[idx]["klass"])
                if self.data_dir is None:
                    img_path = os.path.join("dataset", self.dataset, img_name)
                else:
                    img_path = os.path.join(self.data_dir, img_name)
                image = PIL.Image.open(img_path).convert("RGB")
                if self.use_kornia:
                    image = self.preprocess(PIL.Image.open(img_path).convert('RGB'))
                elif self.transform_on_gpu:
                    image = self.transform_cpu(PIL.Image.open(img_path).convert('RGB'))
                elif self.transform:
                    image = self.transform(image)
                sample["image"] = image
                sample["label"] = label
                sample["image_name"] = img_name
                self.loaded_images.append(sample)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if self.preload:
            return self.loaded_images[idx]
        else:
            sample = dict()
            if torch.is_tensor(idx):
                idx = idx.tolist()
            img_name = self.data_frame.iloc[idx]["file_name"]
            if self.cls_list is None:
                label = self.data_frame.iloc[idx].get("label", -1)
            else:
                label = self.cls_list.index(self.data_frame.iloc[idx]["klass"])

            if self.data_dir is None:
                img_path = os.path.join("dataset", self.dataset, img_name)
            else:
                img_path = os.path.join(self.data_dir, img_name)
            image = PIL.Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            sample["image"] = image
            sample["label"] = label
            sample["image_name"] = img_name
            return sample

    def get_image_class(self, y):
        return self.data_frame[self.data_frame["label"] == y]

    def generate_idx(self, batch_size):
        if self.preload:
            arr = np.arange(len(self.loaded_images))
        else:
            arr = np.arange(len(self.data_frame))
        np.random.shuffle(arr)
        if batch_size >= len(arr):
            return [arr]
        else:
            return np.split(arr, np.arange(batch_size, len(arr), batch_size))

    def get_data_gpu(self, indices):
        images = []
        labels = []
        data = {}
        if self.use_kornia:
            images = [self.loaded_images[i]["image"] for i in indices]
            images = torch.stack(images).to(self.device)
            images = self.transform_gpu(images)
            data["image"] = images

            for i in indices:
            # labels
                labels.append(self.loaded_images[i]["label"])
        else:
            for i in indices:
                if self.preload:
                    if self.transform_on_gpu:
                        images.append(self.transform_gpu(self.loaded_images[i]["image"].to(self.device)))
                    else:
                        images.append(self.transform(self.loaded_images[i]["image"]).to(self.device))
                    labels.append(self.loaded_images[i]["label"])
                else:
                    try:
                        img_name = self.data_frame.iloc[i]["file_name"]
                    except KeyError:
                        img_name = self.data_frame.iloc[i]["filepath"]
                    if self.cls_list is None:
                        label = self.data_frame.iloc[i].get("label", -1)
                    else:
                        label = self.cls_list.index(self.data_frame.iloc[i]["klass"])
                    if self.data_dir is None:
                        img_path = os.path.join("dataset", self.dataset, img_name)
                    else:
                        img_path = os.path.join(self.data_dir, img_name)
                    image = PIL.Image.open(img_path).convert("RGB")
                    image = self.transform(image)
                    images.append(image.to(self.device))
                    labels.append(label)
            data['image'] = torch.stack(images)
        data['label'] = torch.LongTensor(labels).to(self.device)
        return data


class StreamDataset(Dataset):
    def __init__(self, datalist, dataset, transform, cls_list, data_dir=None, device=None, transform_on_gpu=False, use_kornia=True):
        self.use_kornia = use_kornia
        self.images = []
        self.labels = []
        self.dataset = dataset
        self.transform = transform
        self.cls_list = cls_list
        self.data_dir = data_dir
        self.device = device

        self.transform_on_gpu = transform_on_gpu
        mean, std, n_classes, inp_size, _ = get_statistics(dataset=self.dataset)

        self.preprocess = Preprocess(input_size=inp_size)
        if self.transform_on_gpu:
            self.transform_cpu = transforms.Compose(
                [
                    transforms.Resize((inp_size, inp_size)),
                    transforms.PILToTensor()
                ])
            self.transform_gpu = transform
        for data in datalist:
            try:
                img_name = data['file_name']
            except KeyError:
                img_name = data['filepath']
            if self.data_dir is None:
                img_path = os.path.join("dataset", self.dataset, img_name)
            else:
                img_path = os.path.join(self.data_dir, img_name)
            if self.use_kornia:
                self.images.append(self.preprocess(PIL.Image.open(img_path).convert('RGB')))
            elif self.transform_on_gpu:
                self.images.append(self.transform_cpu(PIL.Image.open(img_path).convert('RGB')))
            else:
                self.images.append(PIL.Image.open(img_path).convert('RGB'))
            self.labels.append(self.cls_list.index(data['klass']))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        sample["image"] = image
        sample["label"] = label
        return sample

    @torch.no_grad()
    def get_data(self):
        data = dict()
        images = []
        labels = []
        if self.use_kornia:
            # images
            images = torch.stack(self.images).to(self.device)
            data['image'] = self.transform_gpu(images)

        if not self.use_kornia:
            for i, image in enumerate(self.images):
                if self.transform_on_gpu:
                    images.append(self.transform_gpu(image.to(self.device)))
                else:
                    images.append(self.transform(image))
            data['image'] = torch.stack(images)
        data['label'] = torch.LongTensor(self.labels)
        return data

class ASERMemory(MemoryDataset):
    def __init__(self, dataset, transform=None, cls_list=None, device=None, test_transform=None,
                 data_dir=None, transform_on_gpu=True, save_test=None, keep_history=False, use_kornia=True, memory_size=None):
        super().__init__(dataset, transform, cls_list, device, test_transform,
                 data_dir, transform_on_gpu, save_test, keep_history)
        
        self.memory_size = memory_size

    def replace_samples(self, ind_cur, ind_buffer):

        if len(ind_cur) > 1:
            for i, stream_index in enumerate(ind_cur):
                buffer_index = ind_buffer[i]
                self.cls_count[self.labels[buffer_index]] -= 1
                self.cls_count[self.stream_labels[stream_index]] += 1
                self.cls_idx[self.labels[buffer_index]].remove(buffer_index)
                self.cls_idx[self.stream_labels[stream_index]].append(buffer_index)

                # replace
                self.images[buffer_index] = self.stream_images[stream_index]
                self.labels[buffer_index] = self.stream_labels[stream_index]
        else:
            buffer_index = ind_buffer
            stream_index = ind_cur[0]
            self.cls_count[self.labels[buffer_index]] -= 1
            self.cls_count[self.stream_labels[stream_index]] += 1
            self.cls_idx[self.labels[buffer_index]].remove(buffer_index)
            self.cls_idx[self.stream_labels[stream_index]].append(buffer_index)

            # replace
            self.images[buffer_index] = self.stream_images[stream_index]
            self.labels[buffer_index] = self.stream_labels[stream_index]            


    def get_sampling_candidates(self, n_smp_cls, n_cand_sample, spare_size):

        ##### for current data #####
        current_data = dict()
        images = []
        labels = []

        stream_indices = np.arange(len(self.stream_images)) 

        if spare_size>0:
            stream_indices = stream_indices[spare_size:]
        
        for i in stream_indices:
            # self.stream_images, self.images에는 이미 test_transform이 다 적용되어 있음
            images.append(self.test_transform(self.stream_images[i]))
            labels.append(self.stream_labels[i])
                
        images = torch.stack(images).to(self.device)
        labels = torch.LongTensor(labels)
        current_data['image'] = images
        current_data['label'] = labels
        #current_data['image'] = self.transform_gpu(images) # 이건 aser에 못쓰임 augmentation 되어 있으므로 

        ##### for minority data #####  
        threshold = torch.tensor(1).float().uniform_(0, 1 / len(self.cls_idx)).item()
        minority_data = dict()
        
        cls_proportion = torch.zeros(len(self.cls_idx))
        for i in range(len(self.cls_idx)):
            cls_proportion[i] = len(self.cls_idx[i])

        cls_proportion = cls_proportion / len(self.images)
        minority_ind = nonzero_indices(cls_proportion[current_data['label']] < threshold)
        '''
        print("cls_proportion")
        print(cls_proportion)
        print("threshold")
        print(threshold)
        print("minority_ind")
        print(minority_ind)
        '''
        minority_batch_x = current_data['image'][minority_ind]
        minority_batch_y = current_data['label'][minority_ind]
        minority_data['image'] = minority_batch_x
        minority_data['label'] = minority_batch_y
        min
        ##### for eval data #####    
        eval_data = dict()       
        eval_indices = self.get_class_balance_indices(n_smp_cls, candidate_size=None)
        images = []
        labels = []
        for i in eval_indices:
            images.append(self.test_transform(self.images[i]))
            labels.append(self.labels[i])
        #print("eval", Counter(labels))
        eval_data['image'] = torch.stack(images)
        eval_data['label'] = torch.LongTensor(labels)

        ##### for candidate data #####
        candidate_data = dict()
        candidate_indices = self.get_random_indices(n_cand_sample, excl_indices=eval_indices)
        images = []
        labels = []
        for i in candidate_indices:
            images.append(self.test_transform(self.images[i]))
            labels.append(self.labels[i])
        #print("eval", Counter(labels))
        candidate_data['image'] = torch.stack(images)
        candidate_data['label'] = torch.LongTensor(labels)
        candidate_data['index'] = candidate_indices

        return minority_data, current_data, eval_data, candidate_data

    def get_random_indices(self, num_samples, excl_indices=None):
        total_indices = range(len(self.images))
        candidate_indices = list(set(total_indices) - set(excl_indices))
        indices = np.random.choice(candidate_indices, size=num_samples, replace=False)
        return indices

    def get_aser_train_batches(self):
        data = dict()
        images = []
        labels = []
        stream_indices = np.arange(len(self.stream_images)) 
        
        for i in stream_indices:
            images.append(self.stream_images[i])
            labels.append(self.stream_labels[i])
            self.class_usage_cnt[self.stream_labels[i]] += 1
        
        for i in self.batch_indices:
            images.append(self.images[i])
            labels.append(self.labels[i])
            self.class_usage_cnt[self.labels[i]] += 1

        images = torch.stack(images).to(self.device)
        labels = torch.LongTensor(labels)
        
        '''
        print("self.batch_indices")
        print(self.batch_indices)
        print("now")
        print(Counter(list(labels.numpy())))
        print("class total")
        print(self.class_usage_cnt)
        print("sampl total")
        print(self.usage_cnt)
        '''
        
        # use_kornia=True라고 가정되어 있음
        data['image'] = self.transform_gpu(images)
        data['label'] = labels    
        
        return data

    def get_aser_calculate_batches(self, n_smp_cls, candidate_size):
        #print("whole", Counter(self.labels))
        #print("current", Counter(self.stream_labels))
        current_data = dict()
        candidate_data = None
        eval_data = None
        
        ##### for current data #####
        images = []
        labels = []
        stream_indices = np.arange(len(self.stream_images)) 
        
        for i in stream_indices:
            # self.stream_images, self.images에는 이미 test_transform이 다 적용되어 있음
            images.append(self.test_transform(self.stream_images[i]))
            labels.append(self.stream_labels[i])
                
        images = torch.stack(images).to(self.device)
        labels = torch.LongTensor(labels)
        #current_data['image'] = self.transform_gpu(images) # 이건 aser에 못쓰임 augmentation 되어 있으므로 
        
        current_data['image'] = images
        current_data['label'] = labels

        if len(self.images) > 0:  
            candidate_data = dict()
            eval_data = dict()
            
            ##### for candidate data #####
            candidate_indices = self.get_class_balance_indices(n_smp_cls, candidate_size)
            images = []
            labels = []
            for i in candidate_indices:
                images.append(self.test_transform(self.images[i]))
                labels.append(self.labels[i])
            candidate_data['image'] = torch.stack(images)
            candidate_data['label'] = torch.LongTensor(labels)
            candidate_data['index'] = torch.LongTensor(candidate_indices)
            
            ##### for eval data #####           
            # discard indices는 겹치는 애들을 의미하며, 해당 index eval indices를 뽑을 때 빼주어야 한다.
            eval_indices = self.get_class_balance_indices(n_smp_cls, candidate_size=candidate_size, discard_indices = candidate_indices)
            images = []
            labels = []
            for i in eval_indices:
                images.append(self.test_transform(self.images[i]))
                labels.append(self.labels[i])
            #print("eval", Counter(labels))
            eval_data['image'] = torch.stack(images)
            eval_data['label'] = torch.LongTensor(labels)
        
        return current_data, candidate_data, eval_data

    def register_batch_indices(self, batch_indices=None, batch_size=None):
        if batch_indices is not None:
            self.batch_indices = batch_indices
        else:
            batch_indices = np.random.choice(range(len(self.images)), size=batch_size, replace=False)
            self.batch_indices = batch_indices

    def get_class_balance_indices(self, n_smp_cls, candidate_size=None, discard_indices = None):
        if candidate_size is not None:
            #real_candidate_size = min(n_smp_cls, candidate_size // len(self.cls_idx))
            real_candidate_size = candidate_size // len(self.cls_idx)
            indices = []

            # balanced sampling
            for klass in range(len(self.cls_idx)):
                candidates = self.cls_idx[klass]
                if discard_indices is not None:
                    candidates = list(set(candidates) - set(discard_indices))
                indices.extend(np.random.choice(candidates, size=min(real_candidate_size, len(candidates)), replace=False))

            # additional sampling for match candidate_size
            additional_size = candidate_size % len(self.cls_idx)
            candidates = list(set(range(len(self.images))) - set(indices))
            indices.extend(np.random.choice(candidates, size=additional_size, replace=False))
            '''
            # balanced 맞추고 candidate_size를 다 채우지 못했다면, random하게 나머지 채우기
            candidates = list(set(range(len(self.images))) - set(indices))
            indices.extend(np.random.choice(candidates, size=min(candidate_size - len(indices), len(candidates)), replace=False))
            '''

        else:
            indices = []
            # balanced sampling
            for klass in range(len(self.cls_idx)):
                candidates = self.cls_idx[klass]
                if discard_indices is not None:
                    candidates = list(set(candidates) - set(discard_indices))
                indices.extend(np.random.choice(candidates, size=min(n_smp_cls, len(candidates)), replace=False))

        #print(Counter(np.array(self.labels)[indices]))
        return indices


class DistillationMemory(MemoryDataset):
    def __init__(self, dataset, memory_size, transform=None, cls_list=None, device=None, test_transform=None,
                 data_dir=None, transform_on_gpu=True, save_test=None, keep_history=False, use_logit=True, use_feature=False, use_kornia=True):
        super().__init__(dataset, transform, cls_list, device, test_transform,
                 data_dir, transform_on_gpu, save_test, keep_history, use_kornia=use_kornia)
        self.logits = []
        self.features = []
        self.logits_mask = []
        self.use_logit = use_logit
        self.use_feature = use_feature
        self.logit_budget = 0.0
        self.memory_size = memory_size
        if self.dataset in ['cifar10', 'cifar100']:
            self.img_size = 32*32*3
        elif self.dataset == "tinyimagenet":
            self.img_size = 64*64*3
        elif self.dataset == "imagenet":
            self.img_size == 224*224*3
        else:
            raise NotImplementedError(
            "Please select the appropriate datasets (cifar10, cifar100, tinyimagenet, imagenet)"
            )
            
    def save_logit(self, logit, idx=None):
        if idx is None:
            self.logits.append(logit)
        else:
            self.logits[idx] = logit
        self.logits_mask.append(torch.ones_like(logit))

    def save_feature(self, feature, idx=None):
        if idx is None:
            self.features.append(feature)
        else:
            self.features[idx] = feature

    def add_new_class(self, cls_list):
        self.cls_list = cls_list
        self.cls_count.append(0)
        self.cls_idx.append([])
        self.cls_dict = {self.cls_list[i]:i for i in range(len(self.cls_list))}
        self.cls_train_cnt = np.append(self.cls_train_cnt, 0)
        for i, logit in enumerate(self.logits):
            self.logits[i] = torch.cat([logit, torch.zeros(1).to(self.device)])
            self.logits_mask[i] = torch.cat([self.logits_mask[i], torch.zeros(1).to(self.device)])
        if len(self.logits)>0:
            self.logit_budget = (len(self.logits) * len(self.logits[0])) / self.img_size
            print("logit_budget", self.logit_budget, "memory size", len(self.images))
            
            total_resource = math.ceil(self.logit_budget + len(self.images))
            num_discard_image = total_resource - self.memory_size
            if num_discard_image > 0:
                self.discard_images(num_discard_image)
            
    def discard_images(self, num_discard_image):
        
        print("num_discard_image", num_discard_image)
        target_index = random.sample(range(len(self.labels)), num_discard_image)
        target_index.sort()
        print("target_index")
        print(target_index)
        real_target_index = [idx-i for i, idx in enumerate(target_index)]
        print("real_target_index")
        print(real_target_index)
        
        for del_idx in real_target_index:
            print("del idx", del_idx)
            print(self.cls_idx[self.labels[del_idx]])
            self.cls_idx[self.labels[del_idx]].remove(del_idx)
            self.cls_count[self.labels[del_idx]] -= 1
            
            del self.images[del_idx]
            del self.labels[del_idx]
            del self.logits[del_idx]
            del self.logits_mask[del_idx]
            
            if self.use_feature:
                del self.features[del_idx]
        '''
        # step 1) class balanced를 맞춰서 discard
        per_klass = num_discard_image // len(self.cls_list) 
        cls_list = list(set(self.labels))
        # step 2) 나머지는 discard할 klass select하고 거기서만 제거
        # self.memory.discard_images(per_klass)
        additional_per_klass = random.sample(cls_list, num_discard_image % len(cls_list))

        print("per_klass", per_klass)
        print("additional_per_klass", additional_per_klass)
        
        for klass in range(len(self.cls_list)):
            klass_index = np.where(klass == np.array(self.labels))[0]
            print("klass_index")
            print(klass_index)
            if klass in additional_per_klass:
                num_discard = per_klass
            else:
                num_discard = per_klass + 1
                
            target_index = random.sample(list(klass_index), num_discard)
            target_index.sort()
            
            print("target_index")
            print(target_index)
            
            real_target_index = [idx-i for i, idx in enumerate(target_index)]
            print("real_target_index")
            print(real_target_index)
            
            for del_idx in real_target_index:
                del self.images[del_idx]
                del self.labels[del_idx]
                del self.logits[del_idx]
                del self.logits_mask[del_idx]
                
                if self.use_feature:
                    del self.features[del_idx]
            
        '''
            

    @torch.no_grad()
    def get_batch(self, batch_size, stream_batch_size=0, use_weight=False, transform=None):

        assert batch_size >= stream_batch_size
        stream_batch_size = min(stream_batch_size, len(self.stream_images))
        batch_size = min(batch_size, stream_batch_size + len(self.images))
        memory_batch_size = batch_size - stream_batch_size
        if memory_batch_size > 0:
            if use_weight:
                weight = self.get_weight()
                indices = np.random.choice(range(len(self.images)), size=memory_batch_size, p=weight / np.sum(weight),
                                           replace=False)
            else:
                indices = np.random.choice(range(len(self.images)), size=memory_batch_size, replace=False)
        if stream_batch_size > 0:
            if len(self.stream_images) > stream_batch_size:
                stream_indices = np.random.choice(range(len(self.stream_images)), size=stream_batch_size, replace=False)
            else:
                stream_indices = np.arange(len(self.stream_images))

        data = dict()
        images = []
        labels = []
        logits = []
        features = []
        logit_masks = []
        if self.use_kornia:
            # images
            if stream_batch_size > 0:
                for i in stream_indices:
                    images.append(self.stream_images[i])
                    labels.append(self.stream_labels[i])
            if memory_batch_size > 0:
                for i in indices:
                    images.append(self.images[i])
                    labels.append(self.labels[i])
                    if self.use_logit:
                        logits.append(self.logits[i])
                        logit_masks.append(self.logits_mask[i])
                    if self.use_feature:
                        features.append(self.features[i])
            images = torch.stack(images).to(self.device)
            images = self.transform_gpu(images)
        else:
            if stream_batch_size > 0:
                for i in stream_indices:
                    if transform is None:
                        if self.transform_on_gpu:
                            images.append(self.transform_gpu(self.stream_images[i].to(self.device)))
                        else:
                            images.append(self.transform(self.stream_images[i]))
                    else:
                        if self.transform_on_gpu:
                            images.append(transform(self.stream_images[i].to(self.device)))
                        else:
                            images.append(transform(self.stream_images[i]))
                    labels.append(self.stream_labels[i])
            if memory_batch_size > 0:
                for i in indices:
                    if transform is None:
                        if self.transform_on_gpu:
                            images.append(self.transform_gpu(self.images[i].to(self.device)))
                        else:
                            images.append(self.transform(self.images[i]))
                    else:
                        if self.transform_on_gpu:
                            images.append(transform(self.images[i].to(self.device)))
                        else:
                            images.append(transform(self.images[i]))
                    labels.append(self.labels[i])
                    if self.use_logit:
                        logits.append(self.logits[i])
                        logit_masks.append(self.logits_mask[i])
                    if self.use_feature:
                        features.append(self.features[i])

            images = torch.stack(images)

        data['image'] = images
        data['label'] = torch.LongTensor(labels)
        if memory_batch_size > 0:
            if self.use_logit:
                data['logit'] = torch.stack(logits)
                data['logit_mask'] = torch.stack(logit_masks)
            if self.use_feature:
                data['feature'] = torch.stack(features)
        else:
            if self.use_logit:
                data['logit'] = torch.zeros(1)
                data['logit_mask'] = torch.zeros(1)
            if self.use_feature:
                data['feature'] =torch.zeros(1)
        if self.keep_history:
            self.previous_idx = np.append(self.previous_idx, indices)
        return data



'''
def get_statistics(dataset: str):
    """
    Returns statistics of the dataset given a string of dataset name. To add new dataset, please add required statistics here
    """
    if dataset == 'imagenet':
        dataset = 'imagenet1000'
    assert dataset in [
        "mnist",
        "KMNIST",
        "EMNIST",
        "FashionMNIST",
        "SVHN",
        "cifar10",
        "cifar100",
        "clear10",
        "clear100",
        "CINIC10",
        "imagenet100",
        "imagenet1000",
        "tinyimagenet",
    ]
    mean = {
        "mnist": (0.1307,),
        "KMNIST": (0.1307,),
        "EMNIST": (0.1307,),
        "FashionMNIST": (0.1307,),
        "SVHN": (0.4377, 0.4438, 0.4728),
        "cifar10": (0.4914, 0.4822, 0.4465),
        "cifar100": (0.5071, 0.4867, 0.4408),
        "clear10": (0.485, 0.456, 0.406),
        "clear100": (0.485, 0.456, 0.406),
        "CINIC10": (0.47889522, 0.47227842, 0.43047404),
        "tinyimagenet": (0.4802, 0.4481, 0.3975),
        "imagenet100": (0.485, 0.456, 0.406),
        "imagenet1000": (0.485, 0.456, 0.406),
    }

    std = {
        "mnist": (0.3081,),
        "KMNIST": (0.3081,),
        "EMNIST": (0.3081,),
        "FashionMNIST": (0.3081,),
        "SVHN": (0.1969, 0.1999, 0.1958),
        "cifar10": (0.2023, 0.1994, 0.2010),
        "cifar100": (0.2675, 0.2565, 0.2761),
        "clear10": (0.229, 0.224, 0.225),
        "std": (0.229, 0.224, 0.225),
        "CINIC10": (0.24205776, 0.23828046, 0.25874835),
        "tinyimagenet": (0.2302, 0.2265, 0.2262),
        "imagenet100": (0.229, 0.224, 0.225),
        "imagenet1000": (0.229, 0.224, 0.225),
    }

    classes = {
        "mnist": 10,
        "KMNIST": 10,
        "EMNIST": 49,
        "FashionMNIST": 10,
        "SVHN": 10,
        "cifar10": 10,
        "cifar100": 100,
        "clear10": 11,
        "clear100": 100,
        "CINIC10": 10,
        "tinyimagenet": 200,
        "imagenet100": 100,
        "imagenet1000": 1000,
    }

    in_channels = {
        "mnist": 1,
        "KMNIST": 1,
        "EMNIST": 1,
        "FashionMNIST": 1,
        "SVHN": 3,
        "cifar10": 3,
        "cifar100": 3,
        "clear10": 3,
        "clear100": 3,
        "CINIC10": 3,
        "tinyimagenet": 3,
        "imagenet100": 3,
        "imagenet1000": 3,
    }

    inp_size = {
        "mnist": 28,
        "KMNIST": 28,
        "EMNIST": 28,
        "FashionMNIST": 28,
        "SVHN": 32,
        "cifar10": 32,
        "cifar100": 32,
        "clear10": 224,
        "clear100": 224,
        "CINIC10": 32,
        "tinyimagenet": 64,
        "imagenet100": 224,
        "imagenet1000": 224,
    }
    return (
        mean[dataset],
        std[dataset],
        classes[dataset],
        inp_size[dataset],
        in_channels[dataset],
    )
    '''

# from https://github.com/drimpossible/GDumb/blob/74a5e814afd89b19476cd0ea4287d09a7df3c7a8/src/utils.py#L102:5
def cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5, z=None):
    assert alpha > 0
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    if z is not None:
        z_a, z_b = z, z[index]
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    if z is None:
        return x, y_a, y_b, lam
    else:
        return x, y_a, y_b, lam, z_a, z_b

def cutmix_feature(x, y, feature, prob, weight, alpha=1.0, cutmix_prob=0.5):
    assert alpha > 0
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    y_a, y_b = y, y[index]
    feature_a, feature_b = feature, feature[index]
    prob_a, prob_b = prob, prob[index]
    weight_a, weight_b = weight, weight[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, feature_a, feature_b, prob_a, prob_b, weight_a, weight_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2