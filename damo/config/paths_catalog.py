# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
"""Centralized catalog of paths."""
import os


class DatasetCatalog(object):
    DATA_DIR = 'data'
    DATASETS = {
        # --- 기존 COCO 데이터셋 ---
        'coco_2017_train': {
            'img_dir': 'coco/images/train2017',
            'ann_file': 'coco/annotations/instances_train2017.json'
        },
        'coco_2017_val': {
            'img_dir': 'coco/images/val2017',
            'ann_file': 'coco/annotations/instances_val2017.json'
        },
        'coco_2017_test_dev': {
            'img_dir': 'coco/test2017',
            'ann_file': 'coco/annotations/image_info_test-dev2017.json'
        },
        
        'coco_60_train': {
            'img_dir': 'coco_60/images/train2017',
            'ann_file': 'coco_60/annotations/instances_train2017.json'
        },
        'coco_60_val': {
            'img_dir': 'coco_60/images/val2017',
            'ann_file': 'coco_60/annotations/instances_val2017.json'
        },
        
        'coco_70_train': {
            'img_dir': 'coco_70/images/train2017',
            'ann_file': 'coco_70/annotations/instances_train2017.json'
        },
        'coco_70_val': {
            'img_dir': 'coco_70/images/val2017',
            'ann_file': 'coco_70/annotations/instances_val2017.json'
        },
        
        # --- VOC 데이터셋 ---
        'voc_10_train': {
            'img_dir': 'voc_10/images/train_merged',
            'ann_file': 'voc_10/annotations/instances_train.json'
        },
        'voc_10_val': {
            'img_dir': 'voc_10/images/test2007',
            'ann_file': 'voc_10/annotations/instances_test2007.json'
        },
        
        'voc_train': {
            'img_dir': 'voc/images/train_merged',
            'ann_file': 'voc/annotations/instances_train.json'
        },
        'voc_val': {
            'img_dir': 'voc/images/test2007',
            'ann_file': 'voc/annotations/instances_test2007.json'
        },
        
        # --- SHIFT 데이터셋 ---
        'shift_train_pretrain': {
            'img_dir': '/home/vision/mjlee/Online_CLOD/data/shift_source/images/train',
            'ann_file': '/home/vision/mjlee/Online_CLOD/data/shift_source/annotations/instances_train.json'
        } ,
        'shift_val_pretrain': {
            'img_dir': '/home/vision/mjlee/Online_CLOD/data/shift_source/images/val',
            'ann_file': '/home/vision/mjlee/Online_CLOD/data/shift_source/annotations/instances_val.json'
        },
        
        'shift_train': {
            'img_dir': '/home/vision/mjlee/Online_CLOD/data/shift/images/train',
            'ann_file': '/home/vision/mjlee/Online_CLOD/data/shift/annotations/instances_train.json'
        },
        'shift_val': {
            'img_dir': '/home/vision/mjlee/Online_CLOD/data/shift/images/val',
            'ann_file': '/home/vision/mjlee/Online_CLOD/data/shift/annotations/instances_val.json'
        },
        
        # --- MILITARY SYNTHETIC 데이터셋 ---
        'military_synthetic_train_pretrain': {
            'img_dir': '/home/vision/mjlee/Online_CLOD/data/military_synthetic_domain_source/images/train',
            'ann_file': '/home/vision/mjlee/Online_CLOD/data/military_synthetic_domain_source/annotations/instances_train_area.json'
        },
        'military_synthetic_val_pretrain': {
            'img_dir': '/home/vision/mjlee/Online_CLOD/data/military_synthetic_domain_source/images/val',
            'ann_file': '/home/vision/mjlee/Online_CLOD/data/military_synthetic_domain_source/annotations/instances_val_area.json'
        },
        
        'military_synthetic_train': {
            'img_dir': '/home/vision/mjlee/Online_CLOD/data/military_synthetic/images/train',
            'ann_file': '/home/vision/mjlee/Online_CLOD/data/military_synthetic/annotations/instances_train_area.json'
        },
        'military_synthetic_val': {
            'img_dir': '/home/vision/mjlee/Online_CLOD/data/military_synthetic/images/val',
            'ann_file': '/home/vision/mjlee/Online_CLOD/data/military_synthetic/annotations/instances_val_area.json'
        }
        }

    @staticmethod
    def get(name):
        keywords = ['coco', 'voc', 'shift', 'military']
        if any(keyword in name for keyword in keywords):
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs['img_dir']),
                ann_file=os.path.join(data_dir, attrs['ann_file']),
            )
            return dict(
                factory='COCODataset',
                args=args,
            )
        else:
            raise RuntimeError(f'Unknown dataset name: {name}')