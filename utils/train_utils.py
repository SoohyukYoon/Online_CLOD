from torch import optim
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
    
from damo.detectors.detector import build_local_model
from damo.config.base import parse_config
import os

from torchmetrics.detection import MeanAveragePrecision
import contextlib
import io
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from torchmetrics.detection.helpers import CocoBackend, _get_classes
from lightning_utilities import apply_to_collection


def select_model(dataset, cfg):
    print("Building DAMO-YOLO model")

    model = build_local_model(cfg, device='cuda')
    
    # Load pretrained weights
    if dataset=='VOC_10_10':
        pretrained_path = "./damo_pretrain_outputs_w/voc_10/pretrain_voc_10/damo_pretrain_voc_w.pth"
        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location='cpu')
            model.load_state_dict(state_dict['model'])
    elif dataset=='VOC_15_5':
        pretrained_path = "./damo_pretrain_outputs_w/voc_15/pretrain_voc_15/epoch_300_bs16_ckpt.pth"
        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location='cpu')
            model.load_state_dict(state_dict['model'])        
    elif dataset=='BDD_domain' or dataset=='BDD_domain_small':
        pretrained_path = "./damo_pretrain_bdd100k.pth"
        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location='cpu')
            model.load_state_dict(state_dict['model'])
    elif dataset == 'SHIFT_domain' or dataset == 'SHIFT_domain_small' or dataset == 'SHIFT_domain_small2' or 'hanhwa' in dataset:
        pretrained_path = "./damo_pretrain_outputs_w/shift/pretrain_shift/damo_pretrain_shift_w_newnew.pth"
        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location='cpu')
            model.load_state_dict(state_dict['model'], strict=False)
    elif dataset=='MILITARY_SYNTHETIC_domain_1' or dataset=='MILITARY_SYNTHETIC_domain_2' or dataset=='MILITARY_SYNTHETIC_domain_3':
        pretrained_path = "./damo_pretrain_military_synthetic.pth"
        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location='cpu')
            model.load_state_dict(state_dict['model'])
    elif dataset == 'VisDrone_3_4':
        pretrained_path = "./damo_pretrain_outputs_w/visdrone_3/pretrain_visdrone_3/damo_pretrain_visdrone_3.pth"
        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location='cpu')
            model.load_state_dict(state_dict['model'])
    elif dataset == 'COCO_70_10':
        pretrained_path = "./damo_pretrain_outputs_w/coco_70/pretrain_coco_70/damo_pretrain_coco_70.pth"
        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location='cpu')
            model.load_state_dict(state_dict['model'])
    elif dataset == 'COCO_60_20':
        pretrained_path = "./damo_pretrain_outputs_w/coco_60/pretrain_coco_60/damo_pretrain_coco_60.pth"
        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location='cpu')
            model.load_state_dict(state_dict['model'])
    else:
        print("Pretrained model not found")

    print(model)
    
    return model

def select_optimizer(name, model, lr=0.01, cfg=None):
    
    # use optimizer from cfg
    name = cfg.pop('name', name)
    # remove lr in cfg
    cfg.pop('lr', None)
    
    # exclude classification head weight
    excluded_substrings = [
        'head.gfl_cls.0',
        'head.gfl_cls.1',
        'head.gfl_cls.2',
    ]
    
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = f"{module_name}.{param_name}" if module_name else param_name
            if not any(excl in fullname for excl in excluded_substrings):
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)
    param_groups = [{"params": g[0]},  # add g0 with weight_decay]
                    {"params": g[1], "weight_decay": 0.0},  # add g1 (BatchNorm2d weights)
                    {"params": g[2], "weight_decay": 0.0}]  # add g2 (biases)
    
    optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
    name = {x.lower(): x for x in optimizers}.get(name.lower())
    if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
        optimizer = getattr(optim, name, optim.Adam)(param_groups, lr=lr, **cfg)
    elif name == "RMSProp":
        optimizer = optim.RMSprop(param_groups, lr=lr, **cfg)
    elif name == "SGD":
        optimizer = optim.SGD(param_groups, lr=lr, **cfg)
    
    optimizer.add_param_group({'params': [p.weight for p in model.head.gfl_cls]})
    optimizer.add_param_group({'params': [p.bias for p in model.head.gfl_cls], "weight_decay": 0})

    return optimizer
    # if 'freeze_fc' not in opt_name:
    #     opt.add_param_group({'params': getattr(model, fc_name).parameters()})
    # return opt

def select_scheduler(sched_name, opt, hparam=None):
    if "exp" in sched_name:
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=hparam)
    elif sched_name == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=1, T_mult=2
        )
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            opt, milestones=[30, 60, 80, 90], gamma=0.1
        )
    elif sched_name == "const":
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    return scheduler


def boxlist_to_pred_dict(bl):
    boxes = bl.bbox.detach().to('cpu').float()
    labels = bl.extra_fields['labels'].detach().to('cpu').long()
    scores = bl.extra_fields.get('scores', None)
    if scores is None:
        scores = torch.empty((0,), dtype=torch.float32)
    else:
        scores = scores.detach().to('cpu').float()
    return {'boxes': boxes, 'scores': scores, 'labels': labels}

def boxlist_to_target_dict(bl):
    boxes = bl.bbox.detach().to('cpu').float()
    labels = bl.extra_fields['labels'].detach().to('cpu').long()
    return {'boxes': boxes, 'labels': labels}

class MeanAveragePrecisionCustomized(MeanAveragePrecision):
    def compute(self) -> dict:
        """Computes the metric."""
        return _calculate_map_with_coco_map50added(
            self._coco_backend,
            self.groundtruth_labels,
            self.groundtruth_box,
            self.groundtruth_mask,
            self.groundtruth_crowds,
            self.groundtruth_area,
            self.detection_labels,
            self.detection_box,
            self.detection_mask,
            self.detection_scores,
            self.iou_type,
            self.average,
            self.iou_thresholds,
            self.rec_thresholds,
            self.max_detection_thresholds,
            self.class_metrics,
            self.extended_summary,
        )


def _calculate_map_with_coco_map50added(
    coco_backend: CocoBackend,
    groundtruth_labels: List[Tensor],
    groundtruth_box: List[Tensor],
    groundtruth_mask: List[Tensor],
    groundtruth_crowds: List[Tensor],
    groundtruth_area: List[Tensor],
    detection_labels: List[Tensor],
    detection_box: List[Tensor],
    detection_mask: List[Tensor],
    detection_scores: List[Tensor],
    iou_type: Union[Literal["bbox", "segm"], Tuple[Literal["bbox", "segm"], ...]],
    average: Literal["macro", "micro"],
    iou_thresholds: List[float],
    rec_thresholds: List[float],
    max_detection_thresholds: List[int],
    class_metrics: bool,
    extended_summary: bool,
) -> Dict[str, Tensor]:
    coco_preds, coco_target = coco_backend._get_coco_datasets(
        groundtruth_labels,
        groundtruth_box,
        groundtruth_mask,
        groundtruth_crowds,
        groundtruth_area,
        detection_labels,
        detection_box,
        detection_mask,
        detection_scores,
        iou_type,
        average=average,
    )

    result_dict = {}
    # with contextlib.redirect_stdout(io.StringIO()):
    for i_type in iou_type:
        prefix = "" if len(iou_type) == 1 else f"{i_type}_"
        if len(iou_type) > 1:
            # the area calculation is different for bbox and segm and therefore to get the small, medium and
            # large values correct we need to dynamically change the area attribute of the annotations
            for anno in coco_preds.dataset["annotations"]:
                anno["area"] = anno[f"area_{i_type}"]

        if len(coco_preds.imgs) == 0 or len(coco_target.imgs) == 0:
            result_dict.update(
                coco_backend._coco_stats_to_tensor_dict(
                    12 * [-1.0], prefix=prefix, max_detection_thresholds=max_detection_thresholds
                )
            )
        else:
            coco_eval = coco_backend.cocoeval(coco_target, coco_preds, iouType=i_type)  # type: ignore[operator]
            coco_eval.params.iouThrs = np.array(iou_thresholds, dtype=np.float64)
            coco_eval.params.recThrs = np.array(rec_thresholds, dtype=np.float64)
            coco_eval.params.maxDets = max_detection_thresholds

            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            stats = coco_eval.stats
            result_dict.update(
                coco_backend._coco_stats_to_tensor_dict(
                    stats, prefix=prefix, max_detection_thresholds=max_detection_thresholds
                )
            )

            summary = {}
            if extended_summary:
                summary = {
                    f"{prefix}ious": apply_to_collection(
                        coco_eval.ious, np.ndarray, lambda x: torch.tensor(x, dtype=torch.float32)
                    ),
                    f"{prefix}precision": torch.tensor(coco_eval.eval["precision"]),
                    f"{prefix}recall": torch.tensor(coco_eval.eval["recall"]),
                    f"{prefix}scores": torch.tensor(coco_eval.eval["scores"]),
                }
            result_dict.update(summary)

            # if class mode is enabled, evaluate metrics per class
            if class_metrics:
                # regardless of average method, reinitialize dataset to get rid of internal state which can
                # lead to wrong results when evaluating per class
                coco_preds, coco_target = coco_backend._get_coco_datasets(
                    groundtruth_labels,
                    groundtruth_box,
                    groundtruth_mask,
                    groundtruth_crowds,
                    groundtruth_area,
                    detection_labels,
                    detection_box,
                    detection_mask,
                    detection_scores,
                    iou_type,
                    average="macro",
                )
                coco_eval = coco_backend.cocoeval(coco_target, coco_preds, iouType=i_type)  # type: ignore[operator]
                coco_eval.params.iouThrs = np.array(iou_thresholds, dtype=np.float64)
                coco_eval.params.recThrs = np.array(rec_thresholds, dtype=np.float64)
                coco_eval.params.maxDets = max_detection_thresholds

                map_per_class_list = []
                map50_per_class_list = []
                mar_per_class_list = []
                for class_id in _get_classes(
                    detection_labels=detection_labels, groundtruth_labels=groundtruth_labels
                ):
                    coco_eval.params.catIds = [class_id]
                    with contextlib.redirect_stdout(io.StringIO()):
                        coco_eval.evaluate()
                        coco_eval.accumulate()
                        coco_eval.summarize()
                        class_stats = coco_eval.stats

                    map_per_class_list.append(torch.tensor([class_stats[0]]))
                    map50_per_class_list.append(torch.tensor([class_stats[1]]))
                    mar_per_class_list.append(torch.tensor([class_stats[8]]))

                map_per_class_values = torch.tensor(map_per_class_list, dtype=torch.float32)
                map50_per_class_values = torch.tensor(map50_per_class_list, dtype=torch.float32)
                mar_per_class_values = torch.tensor(mar_per_class_list, dtype=torch.float32)
                
            else:
                map_per_class_values = torch.tensor([-1], dtype=torch.float32)
                map50_per_class_values = torch.tensor([-1], dtype=torch.float32)
                mar_per_class_values = torch.tensor([-1], dtype=torch.float32)
            prefix = "" if len(iou_type) == 1 else f"{i_type}_"
            result_dict.update(
                {
                    f"{prefix}map_per_class": map_per_class_values,
                    f"{prefix}map50_per_class":map50_per_class_values,
                    f"{prefix}mar_{max_detection_thresholds[-1]}_per_class": mar_per_class_values,
                },
            )
    result_dict.update({
        "classes": torch.tensor(
            _get_classes(detection_labels=detection_labels, groundtruth_labels=groundtruth_labels), dtype=torch.int32
        )
    })
    return result_dict