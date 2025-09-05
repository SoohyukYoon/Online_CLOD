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

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        total_loss = 0.0
        if len(sample) > 0:
            self.memory.register_stream(sample)
        for i in range(iterations):
            self.model.train()
            data = self.memory.get_batch(batch_size, stream_batch_size)
            
            self.optimizer.zero_grad()

            loss, loss_item = self.model_forward_with_lwf(data)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.total_flops += (len(data[1]) * self.backward_flops)
            
            self.update_schedule()

            total_loss += loss.item()
            
        return total_loss / iterations
    
    def model_forward_with_lwf(self, batch):
        inps, targets = self.preprocess_batch(batch)
        
        for layer in self.feature_layers:
            self.features_per_layer[layer].clear()
            self.old_features_per_layer[layer].clear()
            
        with torch.cuda.amp.autocast(enabled=False):
            loss_item = self.model(inps, targets)
            loss_new = loss_item["total_loss"]

            self.total_flops += (len(targets) * self.forward_flops)
            
            loss_distill = 0.0
            if getattr(self, "old_model", None) is not None:
                with torch.no_grad():
                    _ = self.old_model(inps, targets)

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
        
        # print(f"loss_new : {loss_new}")
        # print(f"loss_distill : {loss_distill}")
            
        return total_loss, loss_item
    
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