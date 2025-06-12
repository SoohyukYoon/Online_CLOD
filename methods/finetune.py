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
from utils.data_loader import ImageDataset, StreamDataset, MemoryDataset, cutmix_data, get_statistics
from utils.train_utils import select_model, select_optimizer, select_scheduler

logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class FINETUNE(ER):
    def __init__(self, criterion, n_classes, device, **kwargs):
        super().__init__(criterion=criterion, n_classes=n_classes, device=device, **kwargs)
        self.memory = MemoryDataset(self.args, self.dataset, self.exposed_classes, device=self.device, memory_size=self.memory_size, init_buffer_size=3,mosaic_prob=kwargs['mosaic_prob'],mixup_prob=kwargs['mixup_prob'])
        
        self.memory.transform.get_more_data = self.memory.ft_get_more_data
        
        self.temp_batchsize = self.batch_size
    
    def online_step(self, sample, sample_num, n_worker):
        self.temp_batchsize = self.batch_size
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

                self.temp_batch = []
                self.num_updates -= int(self.num_updates)