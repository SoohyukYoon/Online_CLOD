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


from yolo.tools.data_augmentation import *
from statistics import mean
from typing import Generator, List, Tuple, Union
from torch import Tensor
from pathlib import Path
from yolo.utils.dataset_utils import create_image_metadata, scale_segmentation

from collections import defaultdict
import cv2

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
    else:
        raise ValueError("Wrong dataset name")

def get_pretrained_statistics(dataset: str):
    if dataset == 'VOC_10_10':
        return 10, 'data/voc_10/images', 'data/voc_10/annotations' ## 경로 수정
    elif dataset == 'BDD_domain':
        return 13, 'data/bdd100k_source/images', 'data/bdd100k_source/annotations'
    elif dataset == 'SHIFT_domain':
        return 6, 'data/shift_source/images', 'data/shift_source/annotations'
    else:
        raise ValueError("Wrong dataset name")
    
def get_exposed_classes(dataset: str):
    if dataset == 'VOC_10_10':
        return ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow']
    elif dataset == 'BDD_domain':
        return ['pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'traffic light', 'traffic sign', 'train', 'trailer', 'other person', 'other vehicle']
    elif dataset == 'SHIFT_domain':
        return ['pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']

def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tensor]]:
    """
    A collate function to handle batching of images and their corresponding targets.

    Args:
        batch (list of tuples): Each tuple contains:
            - image (Tensor): The image tensor.
            - labels (Tensor): The tensor of labels for the image.

    Returns:
        Tuple[Tensor, List[Tensor]]: A tuple containing:
            - A tensor of batched images.
            - A list of tensors, each corresponding to bboxes for each image in the batch.
    """
    batch_size = len(batch)
    target_sizes = [item[1].size(0) for item in batch]
    # TODO: Improve readability of these process
    # TODO: remove maxBbox or reduce loss function memory usage
    batch_targets = torch.zeros(batch_size, min(max(target_sizes), 100), 5)
    batch_targets[:, :, 0] = -1
    for idx, target_size in enumerate(target_sizes):
        batch_targets[idx, : min(target_size, 100)] = batch[idx][1][:100]

    batch_images, _, batch_reverse, batch_path = zip(*batch)
    batch_images = torch.stack(batch_images)
    batch_reverse = torch.stack(batch_reverse)

    return batch_size, batch_images, batch_targets, batch_reverse, batch_path

class MemoryDataset(Dataset):
    def __init__(self, args, dataset, cls_list=None, device=None, data_dir=None, memory_size=None, 
                 init_buffer_size=None, mosaic_prob=1.0, mixup_prob=0.0):
        self.args = args
        self.image_sizes = args.image_size  # [640, 640]
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
        
        self.build_initial_buffer(init_buffer_size)

        n_classes, image_dir, label_path = get_statistics(dataset=self.dataset)
        self.image_dir = image_dir
        self.label_path = label_path

        self.augment = True

        transforms = {
            "Mosaic": mosaic_prob,
            "MixUp": mixup_prob,
            "HorizontalFlip": 0.5,
            "RandomCrop": 1,
            "RemoveOutliers": 1e-8,
        }
        transforms = [eval(k)(v) for k, v in transforms.items()]
        self.base_size = mean(self.image_sizes)
        self.transform = AugmentationComposer(transforms, self.image_sizes, self.base_size)
        self.transform.get_more_data = self.get_more_data
        
        self.padResize = AugmentationComposer([], self.image_sizes, self.base_size)
        
        # Load all metadata once during initialization
        self.metadata_cache = {}
        self._load_all_metadata()
    
    def _load_all_metadata(self):
        """Load metadata for all possible JSON files"""
        if os.path.isdir(self.label_path):
            json_files = [
                'instances_train2012.json',
                'instances_train2007.json', 
                'instances_val2012.json',
                'instances_val2007.json',
                'instances_test2007.json',
                'instances_train.json'  # fallback
            ]
            
            for json_file in json_files:
                full_path = os.path.join(self.label_path, json_file)
                if os.path.exists(full_path):
                    annotations_index, image_info_dict = create_image_metadata(full_path)
                    self.metadata_cache[json_file] = (annotations_index, image_info_dict)
        else:
            # Single JSON file
            annotations_index, image_info_dict = create_image_metadata(self.label_path)
            self.metadata_cache[self.label_path] = (annotations_index, image_info_dict)

    def ft_get_more_data(self, num: int = 1):
        indices = torch.randint(0, len(self.stream_data), (num,))
        return [self.stream_data[idx][:2] for idx in indices]

    def get_more_data(self, num: int = 1):
        indices = torch.randint(0, len(self), (num,))
        return [self.get_data(idx)[:2] for idx in indices]

    def build_initial_buffer(self, buffer_size=None):
        n_classes, images_dir, label_path = get_pretrained_statistics(self.dataset)
        self.image_dir = images_dir
        self.label_path = label_path
        self.metadata_cache = {}
        self._load_all_metadata()
        if self.dataset == 'VOC_10_10':
            image_files = glob.glob(os.path.join(images_dir, "train2012", "*.jpg")) \
                        + glob.glob(os.path.join(images_dir, "train2007", "*.jpg")) \
                        + glob.glob(os.path.join(images_dir, "val2012", "*.jpg")) \
                        + glob.glob(os.path.join(images_dir, "val2007", "*.jpg"))
        else:
            image_files = glob.glob(os.path.join(images_dir, "train","*.jpg"))

        indices = np.random.choice(range(len(image_files)), size=buffer_size or self.memory_size, replace=False)

        for idx in indices:
            image_path = image_files[idx]
            split_name = image_path.split('/')[-2]
            base_name = image_path.split('/')[-1]
            self.replace_sample({'file_name': split_name + '/' + base_name, 'label': None}, images_dir=images_dir,label_path=label_path)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        img, bboxes, img_path = self.get_data(idx)
        img, bboxes, rev_tensor = self.transform(img, bboxes)
        bboxes[:, [1, 3]] *= self.image_sizes[0]
        bboxes[:, [2, 4]] *= self.image_sizes[1]
        return img, bboxes, rev_tensor, img_path

    def get_data(self, idx):
        img, labels, img_path, _ = self.buffer[idx]
        valid_mask = labels[:, 0] != -1

        return img, labels[valid_mask], img_path


    def load_data(self, img_name, image_dir, label_path, cls_type = None):
        image_path = os.path.join(image_dir, img_name)
        image_id = Path(image_path).stem
        
        if os.path.isdir(label_path):
            if 'train2012' in img_name:
                json_file = 'instances_train2012.json'
            elif 'train2007' in img_name:
                json_file = 'instances_train2007.json'
            elif 'val2012' in img_name:
                json_file = 'instances_val2012.json'
            elif 'val2007' in img_name:
                json_file = 'instances_val2007.json'
            elif 'test2007' in img_name:
                json_file = 'instances_test2007.json'
            else:
                # raise ValueError(f"Cannot determine JSON file for {img_name}")
                json_file = 'instances_train.json'
        else:
            json_file = label_path

        annotations_index, image_info_dict = self.metadata_cache[json_file]
        image_info = image_info_dict.get(image_id, None)
        if image_info is None:
            raise ValueError(f"Image info not found for {image_id}")

        annotations = annotations_index.get(image_info["id"], [])
        image_seg_annotations = scale_segmentation(annotations, image_info)
        labels = self.load_valid_labels(image_id, image_seg_annotations, cls_type)

        if '.jpg' not in image_path:
            image_path += '.jpg'

        img = Image.open(image_path)
        w, h = img.size

        return img, labels, image_path, w / h

    def load_valid_labels(self, label_path: str, seg_data_one_img: list, cls_type: str = None) -> Union[torch.Tensor, None]:
        bboxes = []
        for seg_data in seg_data_one_img:
            cls = seg_data[0]
            if cls >= len(self.cls_list):
                continue
            if cls_type is not None and cls != self.cls_list.index(cls_type): # only one class is allowed per image
                continue
            
            points = np.array(seg_data[1:]).reshape(-1, 2)
            valid_points = points[(points >= 0) & (points <= 1)].reshape(-1, 2)
            if valid_points.size > 1:
                bbox = torch.tensor([cls, *valid_points.min(axis=0), *valid_points.max(axis=0)])
                bboxes.append(bbox)

        if bboxes:
            return torch.stack(bboxes)
        else:
            return torch.zeros((0, 5))

    def register_stream(self, datalist):
        self.stream_data = []
        for data in datalist:
            img_name = data.get('file_name', data.get('filepath'))
            img, labels, image_path, ratio = self.load_data(img_name, image_dir=self.image_dir, label_path=self.label_path, cls_type = data.get('klass', None))
            self.stream_data.append((img, labels, image_path, ratio))

    def replace_sample(self, sample, idx=None, images_dir=None,label_path=None):
        img, labels, image_path, ratio = self.load_data(sample['file_name'], image_dir=images_dir or self.image_dir, label_path=label_path or self.label_path, cls_type = sample.get('klass', None))
        data = (img, labels, image_path, ratio)
        if idx is None:
            self.buffer.append(data)
        else:
            self.buffer[idx] = data

    def get_stream_data(self, sample, transform=False):
        batch = []
        for data in sample:
            img_name = data.get('file_name', data.get('filepath'))
            img, labels, image_path, _ = self.load_data(img_name, image_dir=self.image_dir, label_path=self.label_path)
            if transform:
                img, labels, rev_tensor = self.transform(img, labels)
            else:
                img, labels, rev_tensor = self.padResize(img, labels)
            labels[:, [1, 3]] *= self.image_sizes[0]
            labels[:, [2, 4]] *= self.image_sizes[1]
            batch.append((img, labels, rev_tensor, image_path))
        return collate_fn(batch)
    
    @torch.no_grad()
    def get_batch(self, batch_size, stream_batch_size=0, use_weight=None, transform=None, weight_method=None):
        assert batch_size >= stream_batch_size
        stream_batch_size = min(stream_batch_size, len(self.stream_data))
        batch_size = min(batch_size, stream_batch_size + len(self.buffer))
        memory_batch_size = batch_size - stream_batch_size

        data = []

        if stream_batch_size > 0:
            stream_indices = np.random.choice(range(len(self.stream_data)), size=stream_batch_size, replace=False)
            for i in stream_indices:
                img, bboxes, img_path, _ = self.stream_data[i]
                img, bboxes, rev_tensor = self.transform(img, bboxes)
                bboxes[:, [1, 3]] *= self.image_sizes[0]
                bboxes[:, [2, 4]] *= self.image_sizes[1]
                data.append((img, bboxes, rev_tensor, img_path))

        if memory_batch_size > 0:
            indices = np.random.choice(range(len(self.buffer)), size=memory_batch_size, replace=False)
            for i in indices:
                img, bboxes, img_path = self.get_data(i)
                img, bboxes, rev_tensor = self.transform(img, bboxes)
                bboxes[:, [1, 3]] *= self.image_sizes[0]
                bboxes[:, [2, 4]] *= self.image_sizes[1]
                data.append((img, bboxes, rev_tensor, img_path))

        return collate_fn(data)

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


    def update_gss_score(self, score, idx=None):
        if idx is None:
            self.score.append(score)
        else:
            self.score[idx] = score

    def batch_iterate(size: int, batch_size: int):
        n_chunks = size // batch_size
        if size % batch_size != 0:
            n_chunks += 1

        for i in range(n_chunks):
            yield list(range(i * batch_size, min((i + 1) * batch_size), size))

    def time_update(self, label, time):
        if self.cls_times[label] is not None:
            self.cls_times[label] = self.cls_times[label] * (1-self.weight_ema_ratio) + time * self.weight_ema_ratio
        else:
            self.cls_times[label] = time

    def update_loss_history(self, loss, prev_loss, ema_ratio=0.90, dropped_idx=None):
        if dropped_idx is None:
            loss_diff = np.mean(loss - prev_loss)
        elif len(prev_loss) > 0:
            mask = np.ones(len(loss), bool)
            mask[dropped_idx] = False
            loss_diff = np.mean((loss[:len(prev_loss)] - prev_loss)[mask[:len(prev_loss)]])
        else:
            loss_diff = 0
        difference = loss_diff - np.mean(self.others_loss_decrease[self.previous_idx]) / len(self.previous_idx)
        self.others_loss_decrease[self.previous_idx] -= (1 - ema_ratio) * difference
        self.previous_idx = np.array([], dtype=int)

    
    def get_two_batches(self, batch_size, test_transform):
        indices = np.random.choice(range(len(self.images)), size=batch_size, replace=False)
        data_1 = dict()
        data_2 = dict()
        images = []
        labels = []
        if self.use_kornia:
            # images
            for i in indices:
                images.append(self.images[i])
                labels.append(self.labels[i])
            images = torch.stack(images).to(self.device)
            data_1['image'] = self.transform_gpu(images)

        else:
            for i in indices:
                if self.transform_on_gpu:
                    images.append(self.transform_gpu(self.images[i].to(self.device)))
                else:
                    images.append(self.transform(self.images[i]))
                labels.append(self.labels[i])
            data_1['image'] = torch.stack(images)
        data_1['label'] = torch.LongTensor(labels)
        data_1['index'] = torch.LongTensor(indices)
        images = []
        labels = []
        for i in indices:
            images.append(self.test_transform(self.images[i]))
            labels.append(self.labels[i])
        data_2['image'] = torch.stack(images)
        data_2['label'] = torch.LongTensor(labels)
        
        return data_1, data_2

    def update_std(self, y_list):
        for y in y_list:
            self.class_usage_cnt[y] += 1
            
    def make_cls_dist_set(self, labels, transform=None):
        if transform is None:
            transform = self.transform
        indices = []
        for label in labels:
            indices.append(np.random.choice(self.cls_idx[label]))
        indices = np.array(indices)
        data = dict()
        images = []
        labels = []
        for i in indices:
            images.append(transform(self.images[i]))
            labels.append(self.labels[i])
        data['image'] = torch.stack(images)
        data['label'] = torch.LongTensor(labels)
        return data

    def make_val_set(self, size=None, transform=None):
        if size is None:
            size = int(0.1*len(self.images))
        if transform is None:
            transform = self.transform
        size_per_cls = size//len(self.cls_list)
        indices = []
        for cls_list in self.cls_idx:
            if len(cls_list) >= size_per_cls:
                indices.append(np.random.choice(cls_list, size=size_per_cls, replace=False))
            else:
                indices.append(np.random.choice(cls_list, size=size_per_cls, replace=True))
        indices = np.concatenate(indices)
        data = dict()
        images = []
        labels = []
        for i in indices:
            images.append(transform(self.images[i]))
            labels.append(self.labels[i])
        data['image'] = torch.stack(images)
        data['label'] = torch.LongTensor(labels)
        return data

    def is_balanced(self):
        mem_per_cls = len(self.images)//len(self.cls_list)
        for cls in self.cls_count:
            if cls < mem_per_cls or cls > mem_per_cls+1:
                return False
        return True

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
    def __init__(self, args, dataset, cls_list=None, device=None, data_dir=None, memory_size=None,
                 init_buffer_size=None, mosaic_prob=0.5, mixup_prob=0.0):
        super().__init__(args, dataset, cls_list, device, data_dir, memory_size,
                         init_buffer_size, mosaic_prob, mixup_prob)
        self.alpha = 1.0                  # smoothing constant in 1/(usage+α)
        self.beta = 1.0
        self.usage_decay = 0.95

    def build_initial_buffer(self, buffer_size=None):
        n_classes, images_dir, label_path = get_pretrained_statistics(self.dataset)
        self.image_dir = images_dir
        self.label_path = label_path
        self.metadata_cache = {}
        self._load_all_metadata()
        if self.dataset == 'VOC_10_10':
            image_files = glob.glob(os.path.join(images_dir, "train2012", "*.jpg")) \
                        + glob.glob(os.path.join(images_dir, "train2007", "*.jpg")) \
                        + glob.glob(os.path.join(images_dir, "val2012", "*.jpg")) \
                        + glob.glob(os.path.join(images_dir, "val2007", "*.jpg"))
        else:
            image_files = glob.glob(os.path.join(images_dir, "train", "*.jpg"))

        indices = np.random.choice(range(len(image_files)), size=self.memory_size, replace=False)

        for idx in indices:
            image_path = image_files[idx]
            split_name = image_path.split('/')[-2]
            base_name = image_path.split('/')[-1]
            self.replace_sample({'file_name': split_name + '/' + base_name, 'label': None}, images_dir=images_dir,label_path=label_path)

        for idx in indices:
            image_path = image_files[idx]
            split_name = image_path.split('/')[-2]
            base_name = image_path.split('/')[-1]
            self.replace_sample(
                {
                    'file_name': split_name + '/' + base_name,
                    'label': None,
                    'usage': 0,
                    'classes': [],   # we fill it in replace_sample() after we know labels
                },
                images_dir=images_dir, label_path=label_path
            )
    
    def replace_sample(self, sample, idx=None, images_dir=None, label_path=None):
        img, labels, image_path, ratio = self.load_data(
            sample['file_name'],
            # cls_label=sample['label'],
            image_dir=images_dir or self.image_dir,
            label_path=label_path or self.label_path,
            cls_type = sample.get('klass', None)
        )

        ### BEGIN USAGE
        entry = {
            "img": img,
            "labels": labels,
            "img_path": image_path,
            "ratio": ratio,
            "usage": sample.get("usage", 0),
            "classes": torch.unique(labels[:, 0]).tolist() if len(labels) else []
        }
        ### END USAGE

        if idx is None:
            self.buffer.append(entry)
        else:
            self.buffer[idx] = entry
    
    def get_data(self, idx):
        entry = self.buffer[idx]
        img, labels, img_path = entry["img"], entry["labels"], entry["img_path"]
        valid_mask = labels[:, 0] != -1

        return img, labels[valid_mask], img_path

    @torch.no_grad()
    def get_batch(self, batch_size, stream_batch_size=0, use_weight=None,
                  transform=None, weight_method=None):
        assert batch_size >= stream_batch_size
        stream_batch_size = min(stream_batch_size, len(self.stream_data))
        batch_size = min(batch_size, stream_batch_size + len(self.buffer))
        memory_batch_size = batch_size - stream_batch_size

        data = []
        if stream_batch_size > 0:
            stream_indices = np.random.choice(range(len(self.stream_data)), size=stream_batch_size, replace=False)
            for i in stream_indices:
                img, bboxes, img_path, _ = self.stream_data[i]
                img, bboxes, rev_tensor = self.transform(img, bboxes)
                bboxes[:, [1, 3]] *= self.image_sizes[0]
                bboxes[:, [2, 4]] *= self.image_sizes[1]
                data.append((img, bboxes, rev_tensor, img_path))

        # ───── Memory part ──────────────────────────────────────────────
        if memory_batch_size > 0 and len(self.buffer):

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
                # for e in self.buffer:
                #     e['usage'] *= self.usage_decay
                
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

                img, bboxes, img_path = self.get_data(i)
                img, bboxes, rev_tensor = self.transform(img, bboxes)
                bboxes[:, [1, 3]] *= self.image_sizes[0]
                bboxes[:, [2, 4]] *= self.image_sizes[1]
                data.append((img, bboxes, rev_tensor, img_path))

        return collate_fn(data)

################################################################################################
# Class Balanced Dataset
class ClassBalancedDataset(MemoryDataset):
    def __init__(self, args, dataset, cls_list=None, device=None, data_dir=None, memory_size=None, 
                 init_buffer_size=None, mosaic_prob=1.0, mixup_prob=0.0):
        self.args = args
        self.image_sizes = args.image_size  # [640, 640]
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
        self.cls_train_cnt = np.array([])
        self.score = []
        self.others_loss_decrease = np.array([])
        self.previous_idx = np.array([], dtype=int)
        self.usage_cnt = []
        self.sample_weight = []
        self.data = {}
        
        self.build_initial_buffer(init_buffer_size)

        n_classes, image_dir, label_path = get_statistics(dataset=self.dataset)
        self.image_dir = image_dir
        self.label_path = label_path

        self.augment = True

        transforms = {
            "Mosaic": mosaic_prob,
            "MixUp": mixup_prob,
            "HorizontalFlip": 0.5,
            "RandomCrop": 1,
            "RemoveOutliers": 1e-8,
        }
        transforms = [eval(k)(v) for k, v in transforms.items()]
        self.base_size = mean(self.image_sizes)
        self.transform = AugmentationComposer(transforms, self.image_sizes, self.base_size)
        self.transform.get_more_data = self.get_more_data
        
        self.padResize = AugmentationComposer([], self.image_sizes, self.base_size)
        
        # Load all metadata once during initialization
        self.metadata_cache = {}
        self._load_all_metadata()
        
    
    def build_initial_buffer(self, buffer_size=None):
        n_classes, images_dir, label_path = get_pretrained_statistics(self.dataset)
        self.image_dir = images_dir
        self.label_path = label_path
        self.metadata_cache = {}
        self._load_all_metadata()
        if self.dataset == 'VOC_10_10':
            image_files = glob.glob(os.path.join(images_dir, "train2012", "*.jpg")) \
                        + glob.glob(os.path.join(images_dir, "train2007", "*.jpg")) \
                        + glob.glob(os.path.join(images_dir, "val2012", "*.jpg")) \
                        + glob.glob(os.path.join(images_dir, "val2007", "*.jpg"))
        else:
            image_files = glob.glob(os.path.join(images_dir, "train","*.jpg"))

        indices = np.random.choice(range(len(image_files)), size=buffer_size or self.memory_size, replace=False)

        for idx in indices:
            image_path = image_files[idx]
            split_name = image_path.split('/')[-2]
            base_name = image_path.split('/')[-1]
            self.replace_sample({'file_name': split_name + '/' + base_name, 'label': None}, images_dir=images_dir,label_path=label_path)
    
    def add_new_class(self, cls_list, sample=None):
        self.cls_list = cls_list
        self.cls_count.append(0)
        self.cls_idx.append([])
    
    def replace_sample(self, sample, idx=None, images_dir=None, label_path=None):
        img, labels, image_path, ratio = self.load_data(sample['file_name'], image_dir=images_dir or self.image_dir, label_path=label_path or self.label_path, cls_type = sample.get('klass', None))
        data = (img, labels, image_path, ratio)
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
    def __init__(self, args, dataset, cls_list=None, device=None, data_dir=None, memory_size=None,
                 init_buffer_size=None, mosaic_prob=0.5, mixup_prob=0.0):
        self.args = args
        self.image_sizes = args.image_size  # [640, 640]
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
        self.cls_train_cnt = np.array([])
        self.score = []
        self.others_loss_decrease = np.array([])
        self.previous_idx = np.array([], dtype=int)
        self.usage_cnt = []
        self.sample_weight = []
        self.data = {}
        
        self.build_initial_buffer(init_buffer_size)

        n_classes, image_dir, label_path = get_statistics(dataset=self.dataset)
        self.image_dir = image_dir
        self.label_path = label_path

        self.augment = True

        transforms = {
            "Mosaic": mosaic_prob,
            "MixUp": mixup_prob,
            "HorizontalFlip": 0.5,
            "RandomCrop": 1,
            "RemoveOutliers": 1e-8,
        }
        transforms = [eval(k)(v) for k, v in transforms.items()]
        self.base_size = mean(self.image_sizes)
        self.transform = AugmentationComposer(transforms, self.image_sizes, self.base_size)
        self.transform.get_more_data = self.get_more_data
        
        self.padResize = AugmentationComposer([], self.image_sizes, self.base_size)
        
        # Load all metadata once during initialization
        self.metadata_cache = {}
        self._load_all_metadata()
        
        self.alpha = 1.0                  # smoothing constant in 1/(usage+α)
        self.beta = 1.0
        self.usage_decay = 0.95
    
    def build_initial_buffer(self, buffer_size=None):
        n_classes, images_dir, label_path = get_pretrained_statistics(self.dataset)
        self.image_dir = images_dir
        self.label_path = label_path
        self.metadata_cache = {}
        self._load_all_metadata()
        if self.dataset == 'VOC_10_10':
            image_files = glob.glob(os.path.join(images_dir, "train2012", "*.jpg")) \
                        + glob.glob(os.path.join(images_dir, "train2007", "*.jpg")) \
                        + glob.glob(os.path.join(images_dir, "val2012", "*.jpg")) \
                        + glob.glob(os.path.join(images_dir, "val2007", "*.jpg"))
        else:
            image_files = glob.glob(os.path.join(images_dir, "train", "*.jpg"))

        indices = np.random.choice(range(len(image_files)), size=self.memory_size, replace=False)

        for idx in indices:
            image_path = image_files[idx]
            split_name = image_path.split('/')[-2]
            base_name = image_path.split('/')[-1]
            self.replace_sample({'file_name': split_name + '/' + base_name, 'label': None}, images_dir=images_dir,label_path=label_path)

        for idx in indices:
            image_path = image_files[idx]
            split_name = image_path.split('/')[-2]
            base_name = image_path.split('/')[-1]
            self.replace_sample(
                {
                    'file_name': split_name + '/' + base_name,
                    'label': None,
                    'usage': 0,
                    'classes': [],   # we fill it in replace_sample() after we know labels
                },
                images_dir=images_dir, label_path=label_path
            )
    
    def replace_sample(self, sample, idx=None, images_dir=None, label_path=None):
        img, labels, image_path, ratio = self.load_data(
            sample['file_name'],
            # cls_label=sample['label'],
            image_dir=images_dir or self.image_dir,
            label_path=label_path or self.label_path,
            cls_type = sample.get('klass', None)
        )

        ### BEGIN USAGE
        entry = {
            "img": img,
            "labels": labels,
            "img_path": image_path,
            "ratio": ratio,
            "usage": sample.get("usage", 0),
            "classes": torch.unique(labels[:, 0]).tolist() if len(labels) else []
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
    
    def get_data(self, idx):
        entry = self.buffer[idx]
        img, labels, img_path = entry["img"], entry["labels"], entry["img_path"]
        valid_mask = labels[:, 0] != -1

        return img, labels[valid_mask], img_path

    @torch.no_grad()
    def get_batch(self, batch_size, stream_batch_size=0, use_weight=None,
                  transform=None, weight_method=None):
        assert batch_size >= stream_batch_size
        stream_batch_size = min(stream_batch_size, len(self.stream_data))
        batch_size = min(batch_size, stream_batch_size + len(self.buffer))
        memory_batch_size = batch_size - stream_batch_size

        data = []
        if stream_batch_size > 0:
            stream_indices = np.random.choice(range(len(self.stream_data)), size=stream_batch_size, replace=False)
            for i in stream_indices:
                img, bboxes, img_path, _ = self.stream_data[i]
                img, bboxes, rev_tensor = self.transform(img, bboxes)
                bboxes[:, [1, 3]] *= self.image_sizes[0]
                bboxes[:, [2, 4]] *= self.image_sizes[1]
                data.append((img, bboxes, rev_tensor, img_path))

        # ───── Memory part ──────────────────────────────────────────────
        if memory_batch_size > 0 and len(self.buffer):

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
                # for e in self.buffer:
                #     e['usage'] *= self.usage_decay
                
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

                img, bboxes, img_path = self.get_data(i)
                img, bboxes, rev_tensor = self.transform(img, bboxes)
                bboxes[:, [1, 3]] *= self.image_sizes[0]
                bboxes[:, [2, 4]] *= self.image_sizes[1]
                data.append((img, bboxes, rev_tensor, img_path))

        return collate_fn(data)

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