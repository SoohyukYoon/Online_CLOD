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
            'img_dir': 'voc_10/images/train',
            'ann_file': 'voc_10/annotations/instances_train.json'
        },
        'voc_10_val': {
            'img_dir': 'voc_10/images/val/test2007',
            'ann_file': 'voc_10/annotations/instances_test2007.json'
        },
        
        'voc_15_train': {
            'img_dir': 'voc_15/images/train',
            'ann_file': 'voc_15/annotations/instances_train.json'
        },
        'voc_15_val': {
            'img_dir': 'voc_15/images/val',
            'ann_file': 'voc_15/annotations/instances_val.json'
        },
        
        'voc_train': {
            'img_dir': 'voc/images/train',
            'ann_file': 'voc/annotations/instances_train.json'
        },
        'voc_val': {
            'img_dir': 'voc/images/val/test2007',
            'ann_file': 'voc/annotations/instances_test2007.json'
        },
        
        # --- SHIFT 데이터셋 ---
        'shift_train_pretrain': {
            # 'img_dir': 'shift_source/images/train',
            'img_dir': '/disk1/jhpark/clod/data/shift_source/images/train',
            'ann_file': 'shift_source/annotations/instances_train.json'
        } ,
        'shift_val_pretrain': {
            # 'img_dir': 'shift_source/images/val',
            'img_dir': '/disk1/jhpark/clod/data/shift_source/images/val',
            'ann_file': 'shift_source/annotations/instances_val.json'
        },
        
        'shift_train': {
            # 'img_dir': 'shift/images/train',
            'img_dir': '/disk1/jhpark/clod/data/shift/images/train',
            'ann_file': 'shift/annotations/instances_train.json'
        },
        'shift_val': {
            # 'img_dir': 'shift/images/val',
            'img_dir': '/disk1/jhpark/clod/data/shift/images/val',
            'ann_file': 'shift/annotations/instances_val.json'
        },
        'shift_source_val': {
            # 'img_dir': 'shift_source/images/val',
            'img_dir': '/disk1/jhpark/clod/data/shift_source/images/val',
            'ann_file': 'shift_source/annotations/instances_val.json'
        },
        'shift_cloudy_val': {
            # 'img_dir': 'shift_cloudy/images/val',
            'img_dir': '/disk1/jhpark/clod/data/shift_cloudy/images/val',
            'ann_file': 'shift_cloudy/annotations/instances_val.json'
        },
        'shift_dawndusk_val': {
            # 'img_dir': 'shift_dawndusk/images/val',
            'img_dir': '/disk1/jhpark/clod/data/shift_dawndusk/images/val',
            'ann_file': 'shift_dawndusk/annotations/instances_val.json'
        },
        'shift_foggy_val': {
            'img_dir': '/disk1/jhpark/clod/data/shift_foggy/images/val',
            'ann_file': 'shift_foggy/annotations/instances_val.json'
        },
        'shift_night_val': {
            # 'img_dir': 'shift_night/images/val',
            'img_dir': '/disk1/jhpark/clod/data/shift_night/images/val',
            'ann_file': 'shift_night/annotations/instances_val.json'
        },
        'shift_overcast_val': {
            # 'img_dir': 'shift_overcast/images/val',
            'img_dir': '/disk1/jhpark/clod/data/shift_overcast/images/val',
            'ann_file': 'shift_overcast/annotations/instances_val.json'
        },
        'shift_rainy_val': {
            # 'img_dir': 'shift_rainy/images/val',
            'img_dir': '/disk1/jhpark/clod/data/shift_rainy/images/val',
            'ann_file': 'shift_rainy/annotations/instances_val.json'
        },
        
        # --- BDD 데이터셋 ---
        'bdd_train_pretrain': {
            'img_dir': 'bdd100k_source/images/train',
            'ann_file': 'bdd100k_source/annotations/instances_train.json'
        } ,
        'bdd_val_pretrain': {
            'img_dir': 'bdd100k_source/images/val',
            'ann_file': 'bdd100k_source/annotations/instances_val.json'
        },
        'bdd_train': {
            'img_dir': 'bdd100k/images/train',
            'ann_file': 'bdd100k/annotations/instances_train.json'
        } ,
        'bdd_val': {
            'img_dir': 'bdd100k/images/val',
            'ann_file': 'bdd100k/annotations/instances_val.json'
        },
        
        'bdd100k_source_val': {
            'img_dir': 'bdd100k_source/images/val',
            'ann_file': 'bdd100k_source/annotations/instances_val.json'
        },
        'bdd100k_cloudy_val': {
            'img_dir': 'bdd100k_cloudy/images/val',
            'ann_file': 'bdd100k_cloudy/annotations/instances_val.json'
        },
        'bdd100k_dawndusk_val': {
            'img_dir': 'bdd100k_dawndusk/images/val',
            'ann_file': 'bdd100k_dawndusk/annotations/instances_val.json'
        },
        'bdd100k_night_val': {
            'img_dir': 'bdd100k_night/images/val',
            'ann_file': 'bdd100k_night/annotations/instances_val.json'
        },
        'bdd100k_overcast_val': {
            'img_dir': 'bdd100k_overcast/images/val',
            'ann_file': 'bdd100k_overcast/annotations/instances_val.json'
        },
        
        
        
        # --- MILITARY SYNTHETIC 데이터셋 ---
        'military_synthetic_train_pretrain': {
            'img_dir': 'military_synthetic_domain_source/images/train',
            'ann_file': 'military_synthetic_domain_source/annotations/instances_train_area.json'
        },
        'military_synthetic_val_pretrain': {
            'img_dir': 'military_synthetic_domain_source/images/val',
            'ann_file': 'military_synthetic_domain_source/annotations/instances_val_area.json'
        },
        
        'military_synthetic_train': {
            'img_dir': 'military_synthetic/images/train',
            'ann_file': 'military_synthetic/annotations/instances_train_area.json'
        },
        'military_synthetic_val': {
            'img_dir': 'military_synthetic/images/val',
            'ann_file': 'military_synthetic/annotations/instances_val_area.json'
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