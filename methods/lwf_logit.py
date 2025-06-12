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


class LWF_Logit(ER):
    def __init__(self, criterion, n_classes, device, **kwargs):
        super().__init__(criterion, n_classes, device, **kwargs)
        self.old_model = None
        self.lambda_old = kwargs.get("lambda_old", 0.1)
        logger.info(f"[LWF INIT] lambda_old: {self.lambda_old}")

    def model_forward_with_lwf(self, batch):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            outputs = self.model(batch["img"])
            
            aux_raw, main_raw = outputs["AUX"], outputs["Main"]
            aux_preds = self.vec2box(aux_raw)
            main_preds = self.vec2box(main_raw)
            loss_new, _ = self.model.loss_fn(aux_preds, main_preds, batch['cls'])

            loss_distill = 0.0
            if self.old_model is not None:
                with torch.no_grad():
                    old_outputs = self.old_model(batch["img"])

                    T = 3.0
                    loss_distill = 0.0
                    for new_out, old_out in zip(outputs["Main"], old_outputs["Main"]):
                        new_logits = new_out[0] if isinstance(new_out, tuple) else new_out
                        old_logits = old_out[0] if isinstance(old_out, tuple) else old_out

                        soft_new = F.log_softmax(new_logits / T, dim=1)
                        soft_old = F.softmax(old_logits / T, dim=1)

                        # old 모델의 클래스 수에 맞게 자르기
                        soft_new = soft_new[:, :soft_old.shape[1], ...]

                        loss_distill += F.kl_div(soft_new, soft_old, reduction='batchmean')# * (T * T)
                    loss_distill /= len(outputs["Main"])
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