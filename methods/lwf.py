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

logger = logging.getLogger()
# writer = SummaryWriter("tensorboard")


class LWF(ER):
    def __init__(self, criterion, n_classes, device, **kwargs):
        super().__init__(criterion, n_classes, device, **kwargs)
        self.old_model = None
        self.lambda_old = kwargs.get("lambda_old", 1.0)
        self.distill_type = kwargs.get("distill_type", "feature")
        self.feature_layers = kwargs.get("feature_layers", [2, 4, 6])  # YOLOv8s backbone layers
        logger.info(f"[LWF INIT] lambda_old: {self.lambda_old}, distill_type: {self.distill_type}, feature_layers: {self.feature_layers}")

    def forward_with_lwf(self, batch):
        loss_new, _ = self.model.model(batch)
        loss_distill = 0.0
        if self.old_model is not None:
            with torch.no_grad():
                old_feats = self._extract_features(self.old_model.model, batch["img"])
            new_feats = self._extract_features(self.model.model, batch["img"])
            loss_distill = self.distillation_loss(new_feats, old_feats)
        else:
            logger.info("[LWF DEBUG] No old model yet")

        total_loss = loss_new + self.lambda_old * loss_distill ## 정규화 미적용
        return total_loss

    ## feature-level MSE distillation loss
    def distillation_loss(self, new_feats, old_feats):
        loss = 0.0
        for i, (nf, of) in enumerate(zip(new_feats, old_feats)):
            loss += F.mse_loss(nf, of)
        return loss

    def _extract_features(self, model, x):
        feats = [] ## 레이어 출력을 저장할 리스트

        def hook_fn(module, input, output):
            feats.append(output)

        backbone = model.model

        backbone_layers = list(backbone._modules.values())
        handles = []

        for idx in self.feature_layers:
            try:
                handles.append(backbone_layers[idx].register_forward_hook(hook_fn))
            except Exception as e:
                logger.error(f"[LWF ERROR] Failed to hook layer {idx}: {e}")
                raise e

        with torch.no_grad():
            _ = model(x)

        for h in handles:
            h.remove()

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
            self.model.model.train()
            data = self.memory.get_batch(batch_size, stream_batch_size)
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    batch = self.preprocess_batch(data)
                    loss = self.forward_with_lwf(batch)
            else:
                batch = self.preprocess_batch(data)
                loss = self.forward_with_lwf(batch)

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