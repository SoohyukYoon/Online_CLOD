# When we make a new one, we should inherit the Finetune class.
import logging
import copy
import time
import datetime
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from scipy.stats import chi2, norm
#from ptflops import get_model_complexity_info
from flops_counter.ptflops import get_model_complexity_info
from methods.er_baseline import ER
from utils.data_loader import FreqDataset, MemoryPseudoDataset
from utils.train_utils import select_model, select_optimizer, select_scheduler

from collections import OrderedDict

logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class ERPseudo(ER):
    def initialize_memory_buffer(self, memory_size):
        self.memory_size = memory_size - self.temp_batchsize
        data_args = self.damo_cfg.get_data(self.damo_cfg.dataset.train_ann[0])
        self.memory = MemoryPseudoDataset(ann_file=data_args['args']['ann_file'], root=data_args['args']['root'], transforms=None,class_names=self.damo_cfg.dataset.class_names,
            dataset=self.dataset, cls_list=self.exposed_classes, device=self.device, memory_size=self.memory_size, image_size=self.img_size, aug=self.damo_cfg.train.augment)
        
        self.ema_ratio=0.995
        self.ema_model = copy.deepcopy(self.model)
    def copy_model_head(self):
        self.ema_model.head.gfl_cls[0] = copy.deepcopy(self.model.head.gfl_cls[0])
        self.ema_model.head.gfl_cls[1] = copy.deepcopy(self.model.head.gfl_cls[1])
        self.ema_model.head.gfl_cls[2] = copy.deepcopy(self.model.head.gfl_cls[2])
    def add_new_class(self, class_name):
        super().add_new_class(class_name)
        
        self.copy_model_head()
    
    @torch.no_grad()
    def update_ema_model(self, num_updates=1.0):
        
        model_params = OrderedDict(self.model.named_parameters())
        
        ema_params = OrderedDict(self.ema_model.named_parameters())
        
        assert model_params.keys() == ema_params.keys()
        # self.sdp_updates += 1
        for name, param in model_params.items():
            ema_params[name].sub_((1. - self.ema_ratio) * (ema_params[name] - param))
        self.copy_model_head()

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.ema_model.named_buffers())

        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            shadow_buffers[name].copy_(buffer)
    
    def online_step(self, sample, sample_num, n_worker):
        if sample.get('klass',None) and sample['klass'] not in self.exposed_classes:
            self.online_after_task(sample_num)
            self.add_new_class(sample['klass'])
        elif sample.get('domain',None) and sample['domain'] not in self.exposed_domains:
            self.exposed_domains.append(sample['domain'])
        
        self.temp_batch.append(sample)
        self.num_updates += self.online_iter
        if len(self.temp_batch) == self.temp_batchsize:
            iteration = int(self.num_updates)
            if iteration != 0:
                train_loss = self.online_train(self.temp_batch, self.batch_size, n_worker,
                                                      iterations=int(self.num_updates), stream_batch_size=self.temp_batchsize)
                self.report_training(sample_num, train_loss)
                
                for stored_sample in self.temp_batch:
                    self.update_memory(stored_sample)

                self.temp_batch = []
                self.num_updates -= int(self.num_updates)
    
    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        total_loss = 0.0
        if len(sample) > 0:
            self.memory.register_stream(sample)
        for i in range(iterations):
            data = self.memory.get_batch(batch_size, stream_batch_size, model=self.ema_model, score_thresh=0.7)
            self.model.train()
            self.optimizer.zero_grad()

            loss, loss_item = self.model_forward(data)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.total_flops += (len(data[1]) * self.backward_flops)
            
            self.update_schedule()
            self.update_ema_model(num_updates=1.0)

            total_loss += loss.item()
            
        return total_loss / iterations