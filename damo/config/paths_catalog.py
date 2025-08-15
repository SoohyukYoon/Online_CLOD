# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
"""Centralized catalog of paths."""
import os


class DatasetCatalog(object):
    DATA_DIR = 'data'
    DATASETS = {
        # --- 기존 COCO 데이터셋 ---
        'coco_2017_train': {
            'img_dir': 'coco/train2017',
            'ann_file': 'coco/annotations/instances_train2017.json'
        },
        'coco_2017_val': {
            'img_dir': 'coco/val2017',
            'ann_file': 'coco/annotations/instances_val2017.json'
        },
        'coco_2017_test_dev': {
            'img_dir': 'coco/test2017',
            'ann_file': 'coco/annotations/image_info_test-dev2017.json'
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
        
        # --- BDD100K 데이터셋 ---

        
        # --- SHIFT 데이터셋 ---
        
        }

    @staticmethod
    def get(name):
        keywords = ['coco', 'voc', 'bdd', 'shift']
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