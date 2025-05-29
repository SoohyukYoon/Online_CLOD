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

logger = logging.getLogger()
# writer = SummaryWriter("tensorboard")


class LWF1(ER):
    def __init__(self, criterion, n_classes, device, **kwargs):
        super().__init__(criterion, n_classes, device, **kwargs)
        self.old_model = None
        self.lambda_old = kwargs.get("lambda_old", 1.0)
        self.distill_type = kwargs.get("distill_type", "feature")
        self.feature_layers = kwargs.get("feature_layers", [18])  # neck RepNCSPELAN
        logger.info(f"[LWF INIT] lambda_old: {self.lambda_old}, distill_type: {self.distill_type}, feature_layers: {self.feature_layers}")

    def model_forward_with_lwf(self, batch):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            outputs = self.model(batch["img"])
            
            pdb.set_trace()
            
            aux_raw, main_raw = outputs["AUX"], outputs["Main"]
            aux_preds = self.vec2box(aux_raw)
            main_preds = self.vec2box(main_raw)
            loss_new, _ = self.model.loss_fn(aux_preds, main_preds, batch['cls'])

            loss_distill = 0.0
            if self.old_model is not None:
                with torch.no_grad():
                    old_feats = self._extract_features(self.old_model, batch["img"])
                new_feats = self._extract_features(self.model, batch["img"])
                loss_distill = self.distillation_loss(new_feats, old_feats)
            else:
                logger.info("[LWF DEBUG] No old model yet")

        total_loss = loss_new + self.lambda_old * loss_distill
        return total_loss
    
    def distillation_loss(self, new_feats, old_feats):
        return F.mse_loss(new_feats[0], old_feats[0])

    def _extract_features(self, model, x):
        feats = []

        def hook_fn(module, input, output):
            feats.append(output)

        handle = model.model[self.feature_layers[0]].register_forward_hook(hook_fn)
        _ = model(x)
        handle.remove()

        return feats

    def online_step(self, sample, sample_num, n_worker):
        self.temp_batchsize = self.batch_size
        if sample['klass'] not in self.exposed_classes:
            self.online_after_task(sample_num)  # 이전 모델 저장
            self.add_new_class(sample['klass'])  # 새로운 클래스 추가

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