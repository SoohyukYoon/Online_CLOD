import copy
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from methods.er_baseline import ER
from methods.lwf_logit import LWF_Logit
from utils.data_loader import ImageDataset, StreamDataset, cutmix_data
import types
import pdb

logger = logging.getLogger()
# writer = SummaryWriter("tensorboard")


class LWF_Feature(LWF_Logit):
    def __init__(self, criterion, n_classes, device, **kwargs):
        super().__init__(criterion, n_classes, device, **kwargs)
        self.feature_layers = kwargs.get("feature_layers", ["neck.merge_3.convs.2", "neck.merge_4.convs.2", "neck.merge_6.convs.2"])  # neck
        logger.info(f"[LWF INIT] feature_layers: {self.feature_layers}")
        self.hooks = []
        self.features_per_layer = {}
    
    def _get_layer(self, root_model, layer_spec):
        mod = root_model
        for attr in str(layer_spec).split("."):
            if attr.isdigit():
                mod = mod[int(attr)]
            else:
                mod = getattr(mod, attr)
        return mod

    def put_hook(self):
        self.features_per_layer = {}
        for layer in self.feature_layers:
            self.features_per_layer[layer] = []
        
        self.old_features_per_layer = {}
        for layer in self.feature_layers:
            self.old_features_per_layer[layer] = []
        self.hooks = []
        for layer in self.feature_layers:
            hook = self._get_layer(self.model, layer).register_forward_hook(
                lambda m, x, y, layer=layer: self.features_per_layer[layer].append(y))
            hook2 = self._get_layer(self.old_model, layer).register_forward_hook(
                lambda m, x, y, layer=layer: self.old_features_per_layer[layer].append(y))
            self.hooks.append(hook)
            self.hooks.append(hook2)

    def model_forward_with_lwf(self, batch):
        for layer in self.feature_layers:
            self.features_per_layer[layer].clear()
            self.old_features_per_layer[layer].clear()
            
        images = batch.get("img").to(self.device, non_blocking=True)
        cls_targets = batch.get("cls")
        
        batch_size = images.shape[0]
        formatted_targets = []
        for i in range(batch_size):
            img_targets = cls_targets[i]
            valid_targets = img_targets[img_targets[:, 0] != -1]

            target_obj = types.SimpleNamespace()

            if valid_targets.numel() > 0:
                target_obj.bbox = valid_targets[:, 1:]
                cls_tensor = valid_targets[:, 0].long()
                target_obj._cls_tensor = cls_tensor
            else:
                target_obj.bbox = torch.empty(0, 4).to(self.device)
                target_obj._cls_tensor = torch.empty(0).to(self.device).long()
            
            target_obj.get_field = lambda field_name, t=target_obj: t._cls_tensor if field_name == 'labels' else None
            
            formatted_targets.append(target_obj)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            loss_item = self.model(images, targets=formatted_targets)
            loss_new = loss_item["total_loss"]

            loss_distill = 0.0
            if getattr(self, "old_model", None) is not None:
                with torch.no_grad():
                    _ = self.old_model(images, targets=formatted_targets)

                n_layers = len(self.feature_layers)
                for layer in self.feature_layers:
                    new_feats = self.features_per_layer[layer][0]
                    old_feats = self.old_features_per_layer[layer][0]

                    if new_feats.shape != old_feats.shape:
                        if (new_feats.dim() == 4 and old_feats.dim() == 4 
                            and new_feats.shape[0] == old_feats.shape[0]):
                            bridge = nn.Conv2d(new_feats.shape[1], old_feats.shape[1], 1, bias=False).to(new_feats.device)
                            with torch.no_grad():
                                bridge.weight.zero_()
                            new_feats = bridge(new_feats)
                        else:
                            raise RuntimeError(
                                f"Feature shape mismatch: new {tuple(new_feats.shape)} vs old {tuple(old_feats.shape)}"
                            )
                    loss_distill += self.distillation_loss(new_feats, old_feats) / n_layers
            else:
                logger.info("[LWF DEBUG] No old model yet")

        total_loss = loss_new + self.lambda_old * loss_distill
        
        print(f"loss_new : {loss_new}")
        print(f"loss_distill : {loss_distill}")
            
        return total_loss
    
    def distillation_loss(self, new_feats, old_feats):
        return F.mse_loss(new_feats, old_feats)

    def online_after_task(self, cur_iter):
        for hook in self.hooks:
            hook.remove()
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        logger.info("[LWF] Saved old model after task.")
        
        self.put_hook()
    
    def online_evaluate(self, sample_num, data_time):
        for hook in self.hooks:
            hook.remove()
        
        eval_dict = super().online_evaluate(sample_num, data_time)
        
        self.put_hook()  # Re-register hooks after evaluation
        
        return eval_dict