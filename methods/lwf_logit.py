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
from utils.data_loader import ImageDataset, StreamDataset, cutmix_data
import pdb
import types

logger = logging.getLogger()
# writer = SummaryWriter("tensorboard")


class LWF_Logit(ER):
    def __init__(self, criterion, n_classes, device, **kwargs):
        super().__init__(criterion, n_classes, device, **kwargs)
        self.old_model = None
        self.lambda_old = kwargs.get("lambda_old", 0.1)
        logger.info(f"[LWF INIT] lambda_old: {self.lambda_old}")

    def model_forward_with_lwf(self, batch):
        images = batch.get("img").to(self.device, non_blocking=True)
        cls_targets = batch.get("cls")
        
        batch_size = images.shape[0]
        formatted_targets = []
        for i in range(batch_size):
            img_targets = cls_targets[i]
            valid_targets = img_targets[img_targets[:, 0] != -1]

            tgt = types.SimpleNamespace()
            if valid_targets.numel() > 0:
                tgt.bbox = valid_targets[:, 1:]
                cls_tensor = valid_targets[:, 0].long()
                tgt._cls_tensor = cls_tensor
            else:
                tgt.bbox = torch.empty(0, 4, device=self.device)
                tgt._cls_tensor = torch.empty(0, dtype=torch.long, device=self.device)

            tgt.get_field = (lambda field_name, t=tgt: t._cls_tensor
                            if field_name == 'labels' else None)
            formatted_targets.append(tgt)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            loss_item = self.model(images, targets=formatted_targets)
            loss_new = loss_item["total_loss"]

            loss_distill = images.new_tensor(0.0)

            # pdb.set_trace()
            
            if getattr(self, "old_model", None) is not None:
                with torch.no_grad():
                    old_feats_b = self.old_model.backbone(images)
                    old_feats_n = self.old_model.neck(old_feats_b)
                    # [3, 16, 10, 80, 80]
                    old_levels = self.old_model.head.forward_cls_logits_levels(
                        old_feats_n, apply_sigmoid=False, drop_bg=True
                    )

                # [3, 16, 11, 80, 80]
                new_feats_b = self.model.backbone(images)
                new_feats_n = self.model.neck(new_feats_b)
                new_levels = self.model.head.forward_cls_logits_levels(
                    new_feats_n, apply_sigmoid=False, drop_bg=True
                )

                L = len(old_levels)  # 3
                T = 3.0
                kl_sum = images.new_tensor(0.0)
                for l in range(L):
                    new_l = new_levels[l]
                    old_l = old_levels[l]
                    C_old = old_l.shape[1]
                    soft_new = F.log_softmax(new_l / T, dim=1)
                    soft_old = F.softmax(old_l / T, dim=1)
                    
                    soft_new = soft_new[:, :C_old, :, :]
                    
                    kl = F.kl_div(soft_new, soft_old, reduction="batchmean")
                    kl_sum = kl_sum + kl

                loss_distill = kl_sum / L
                
            else:
                logger.info("[LWF DEBUG] No old model yet")

            total_loss = loss_new + self.lambda_old * loss_distill
            return total_loss

    def online_step(self, sample, sample_num, n_worker):
        if sample.get('klass',None) and sample['klass'] not in self.exposed_classes:
            self.online_after_task(sample_num)
            self.add_new_class(sample['klass'])
        elif sample.get('domain',None) and sample['domain'] not in self.exposed_domains:
            self.exposed_domains.append(sample['domain'])
            self.online_after_task(sample_num)

        self.temp_batch.append(sample)
        self.num_updates += self.online_iter

        if len(self.temp_batch) == self.temp_batchsize:
            iteration = int(self.num_updates)
            if iteration != 0:
                train_loss = self.online_train(
                    self.temp_batch,
                    self.batch_size,
                    n_worker,
                    iterations=iteration,
                    stream_batch_size=self.temp_batchsize
                )
                self.report_training(sample_num, train_loss)

                for stored_sample in self.temp_batch:
                    self.update_memory(stored_sample)

                self.temp_batch = []
                self.num_updates -= iteration

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        total_loss = 0.0
        if len(sample) > 0:
            self.memory.register_stream(sample)

        for i in range(iterations):
            self.model.train()
            data = self.memory.get_batch(batch_size, stream_batch_size)
            batch = {
                "img": data[1],
                "cls": data[2],
                "reverse": data[3],
                "img_path": data[4],
            }
            batch = self.preprocess_batch(batch)
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss = self.model_forward_with_lwf(batch)
            else:
                loss = self.model_forward_with_lwf(batch)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.update_schedule()
            total_loss += loss.item()

        return total_loss / iterations

    def online_after_task(self, cur_iter):
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        logger.info("[LWF] Saved old model after task.")