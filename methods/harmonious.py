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
from utils.data_loader import HarmoniousDataset
from utils.train_utils import select_model, select_optimizer, select_scheduler
from copy import deepcopy
import os
from damo.detectors.detector import build_local_model_harmonious
from collections import OrderedDict


logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")


def is_parallel(model):
    """check if model is in parallel mode."""
    parallel_type = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)
            
class ModelEMA:

    def __init__(self, model, decay=0.9999):
        # Create EMA(FP32)
        self.model = deepcopy(model.module if is_parallel(model) else model).eval()
        
        # decay exponential ramp (to help early epochs)
        # self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        self.decay = decay
        for p in self.model.parameters():
            p.requires_grad_(False)
            
    def update(self, model, decay=None):
        if decay is None:
            decay = self.decay
        # Update EMA parameters
        with torch.no_grad():
            # msd = (
            #     model.module.state_dict() if is_parallel(model) else model.state_dict()
            # )  # model state_dict
            
            # for k, v in self.model.state_dict().items():
            #     if v.dtype.is_floating_point:
            #         v *= decay
            #         v += (1.0 - decay) * msd[k].detach()
            # breakpoint()
            model_params = OrderedDict(model.named_parameters())
            ema_params = OrderedDict(self.model.named_parameters())
            assert model_params.keys() == ema_params.keys()
            for name, param in model_params.items():
                ema_params[name].sub_((1. - decay) * (ema_params[name] - param))
            # breakpoint()
            # self.model.head.gfl_cls[0] = copy.deepcopy(model.head.gfl_cls[0])
            # self.model.head.gfl_cls[1] = copy.deepcopy(model.head.gfl_cls[1])
            # self.model.head.gfl_cls[2] = copy.deepcopy(model.head.gfl_cls[2])
                        
            model_buffers = OrderedDict(model.named_buffers())
            shadow_buffers = OrderedDict(self.model.named_buffers())

            assert model_buffers.keys() == shadow_buffers.keys()

            for name, buffer in model_buffers.items():
                shadow_buffers[name].copy_(buffer)


class Harmonious(ER):

    def __init__(self, criterion, n_classes, device, **kwargs):
        super().__init__(criterion, n_classes, device, **kwargs)
        
        
        # if self.dataset == 'SHIFT_domain' or self.dataset == 'SHIFT_domain_small' or self.dataset == 'SHIFT_domain_small2':
        #     self.model = build_local_model_harmonious(self.damo_cfg, device='cuda')
        #     pretrained_path = "./damo_pretrain_outputs_w/shift/pretrain_shift/damo_pretrain_shift_w_newnew.pth"
        #     if os.path.exists(pretrained_path):
        #         state_dict = torch.load(pretrained_path, map_location='cpu')
        #         self.model.load_state_dict(state_dict['model'])
        # elif self.dataset=='VOC_15_5':
        #     self.model = build_local_model_harmonious(self.damo_cfg, device='cuda')
        #     pretrained_path = "./damo_pretrain_outputs_w/voc_15/pretrain_voc_15/epoch_300_bs16_ckpt.pth"
        #     if os.path.exists(pretrained_path):
        #         state_dict = torch.load(pretrained_path, map_location='cpu')
        #         self.model.load_state_dict(state_dict['model'])        
        # else:
        #     self.model = select_model(self.dataset, self.damo_cfg)
            
            
        # self.optimizer = select_optimizer(self.opt_name, self.model, lr=self.lr, cfg=self.damo_cfg.train.optimizer)
                    
        
        self.score_thresh = 0.55
        self.decay_factor = 0.0
        self.ema_model = ModelEMA(self.model, self.decay_factor)
        
    
    def initialize_memory_buffer(self, memory_size):
        self.memory_size = memory_size - 8
        data_args = self.damo_cfg.get_data(self.damo_cfg.dataset.train_ann[0])
        self.memory = HarmoniousDataset(ann_file=data_args['args']['ann_file'], root=data_args['args']['root'], transforms=None,class_names=self.damo_cfg.dataset.class_names,
            dataset=self.dataset, cls_list=self.exposed_classes, device=self.device, memory_size=self.memory_size, image_size=self.img_size, aug=self.damo_cfg.train.augment)
        
        self.new_exposed_classes = ['pretrained']

    def add_new_class(self, class_name):
        super().add_new_class(class_name)
        self.new_exposed_classes.append(class_name)
        self.memory.new_exposed_classes = self.new_exposed_classes
    
    def online_step(self, sample, sample_num, n_worker):
        if sample.get('klass',None) and sample['klass'] not in self.exposed_classes:
            self.online_after_task(sample_num)
            self.add_new_class(sample['klass'])
        elif sample.get('domain',None) and sample['domain'] not in self.exposed_domains:
            self.exposed_domains.append(sample['domain'])
            self.new_exposed_classes.append(sample['domain'])
            self.memory.new_exposed_classes.append(sample['domain'])
            self.memory.cls_count.append(0)
            self.memory.cls_idx.append([])
        
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
            self.model.train()
            data = self.memory.get_batch(batch_size, stream_batch_size, model=self.ema_model.model, score_thresh=self.score_thresh)
            
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
            self.ema_model.update(self.model)

            total_loss += loss.item()
            
        return total_loss / iterations
    

    def model_forward(self, batch):
        inps, targets, score_weights = self.preprocess_batch(batch)
        
        # with torch.cuda.amp.autocast(enabled=self.use_amp):
        with torch.cuda.amp.autocast(enabled=False):
            loss_item = self.model(inps, targets)#, score_weights=score_weights)
            total_loss = loss_item["total_loss"]
            
            self.total_flops += (len(targets) * self.forward_flops)
        
        return total_loss, loss_item
    
    def preprocess_batch(self, batch):
        inps = batch[0].to(self.device)
        targets = [target.to(self.device) for target in batch[1]]
        score_weights = batch[3]
        return inps, targets, score_weights
    
    
    
    
    def update_memory(self, sample):
        self.balanced_replace_memory(sample)

    def balanced_replace_memory(self, sample):
        if len(self.memory) >= self.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            if sample.get('klass', None):
                sample_category = sample['klass']
            elif sample.get('domain', None):
                sample_category = sample['domain']
            else:
                sample_category = 'pretrained'
            
            label_frequency[self.new_exposed_classes.index(sample_category)] += 1
            cls_to_replace = np.random.choice(
                np.flatnonzero(np.array(label_frequency) == np.array(label_frequency).max()))
            idx_to_replace = np.random.choice(self.memory.cls_idx[cls_to_replace])
            self.memory.replace_sample(sample, idx_to_replace)
            
            self.memory.cls_count[cls_to_replace] -= 1
            self.memory.cls_idx[cls_to_replace].remove(idx_to_replace)
            self.memory.cls_idx[self.new_exposed_classes.index(sample_category)].append(idx_to_replace)
        else:
            self.memory.replace_sample(sample)
