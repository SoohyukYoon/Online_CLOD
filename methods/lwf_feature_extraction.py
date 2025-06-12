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
import pdb

logger = logging.getLogger()
# writer = SummaryWriter("tensorboard")


class LWF_Feature(LWF_Logit):
    def __init__(self, criterion, n_classes, device, **kwargs):
        super().__init__(criterion, n_classes, device, **kwargs)
        self.feature_layers = kwargs.get("feature_layers", [15,18,21])  # neck RepNCSPELAN
        logger.info(f"[LWF INIT] feature_layers: {self.feature_layers}")
        self.hooks = []
        self.features_per_layer = {}
        
    def put_hook(self):
        self.features_per_layer = {}
        for layer in self.feature_layers:
            self.features_per_layer[layer] = []
        
        self.old_features_per_layer = {}
        for layer in self.feature_layers:
            self.old_features_per_layer[layer] = []
        self.hooks = []
        for layer in self.feature_layers:
            hook = self.model.model[layer].register_forward_hook(
                lambda m, x, y, layer=layer: self.features_per_layer[layer].append(y))
            hook2 = self.old_model.model[layer].register_forward_hook(
                lambda m, x, y, layer=layer: self.old_features_per_layer[layer].append(y))
            self.hooks.append(hook)
            self.hooks.append(hook2)

    def model_forward_with_lwf(self, batch):
        for layer in self.feature_layers:
            self.features_per_layer[layer].clear()
            self.old_features_per_layer[layer].clear()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            outputs = self.model(batch["img"])
            
            aux_raw, main_raw = outputs["AUX"], outputs["Main"]
            aux_preds = self.vec2box(aux_raw)
            main_preds = self.vec2box(main_raw)
            loss_new, _ = self.model.loss_fn(aux_preds, main_preds, batch['cls'])

            loss_distill = 0.0
            if self.old_model is not None:
                with torch.no_grad():
                    _,_ = self.old_model(batch["img"])
                
                for layer in self.feature_layers:
                    new_feats = self.features_per_layer[layer][0]
                    old_feats = self.old_features_per_layer[layer][0]
                    loss_distill += self.distillation_loss(new_feats, old_feats) / len(self.feature_layers)
            else:
                logger.info("[LWF DEBUG] No old model yet")

        total_loss = loss_new + self.lambda_old * loss_distill
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