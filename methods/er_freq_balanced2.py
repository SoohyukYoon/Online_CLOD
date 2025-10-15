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
from methods.baseline2 import BASELINE2
from utils.data_loader import FreqClsBalancedDataset, FreqDataset
from utils.train_utils import select_model, select_optimizer, select_scheduler
import torch.nn.functional as F

logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class ERFreqBalanced2(ER):
    def initialize_memory_buffer(self, memory_size):
        self.memory_size = memory_size - 8
        data_args = self.damo_cfg.get_data(self.damo_cfg.dataset.train_ann[0])
        self.memory = FreqClsBalancedDataset(ann_file=data_args['args']['ann_file'], root=data_args['args']['root'], transforms=None,class_names=self.damo_cfg.dataset.class_names,
            dataset=self.dataset, cls_list=self.exposed_classes, device=self.device, memory_size=self.memory_size, image_size=self.img_size, aug=self.damo_cfg.train.augment)
        
        self.new_exposed_classes = ['pretrained']
        self.lambda_ = 0.5
        self.use_hardlabel = True

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

    def model_forward(self, batch):
        inps, targets = self.preprocess_batch(batch)
        
        # with torch.cuda.amp.autocast(enabled=self.use_amp):
        with torch.cuda.amp.autocast(enabled=False):
            image_tensors = inps.tensors
            new_feats_b = self.model.backbone(image_tensors)
            new_feats_n = self.model.neck(new_feats_b)
            
            loss_item = self.model.head.forward_train(new_feats_n, labels=targets)
            loss_new = loss_item["total_loss"]

            cls_scores, _, bbox_before_softmax, bbox_preds, pos_inds, labels, label_scores, num_total_pos = self.model.head.get_head_outputs_with_label(
                new_feats_n, labels=targets, drop_bg=False
            )
            
            # focal loss with current prediction score as label
            mask = (labels != self.num_learned_class) & (labels != self.num_learned_class-1)
            label_scores[mask] = cls_scores[mask, labels[mask]].detach()
            loss_qfl_self = self.model.head.loss_cls(cls_scores, (labels, label_scores),
                                 avg_factor=num_total_pos)
            
            # box dfl self
            reg_max = self.model.head.reg_max
            weight_targets = cls_scores.detach()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            norm_factor = max(weight_targets.sum().item(), 1.0)
            
            if self.use_hardlabel:
                # 1. hard pseudo-label
                bins = torch.arange(reg_max + 1, device=bbox_preds.device, dtype=bbox_preds.dtype)
                expected = (bbox_preds * bins)  # (N, 4, K+1)
                expected = expected.sum(dim=-1).detach()     # (N, 4)
                # Build hard pseudo targets in the same shape/type as your original dfl_targets
                pseudo_dfl_targets_hard = expected.clamp(min=0, max=reg_max)  # (N, 4)
                loss_dfl_self = self.model.head.loss_dfl(
                    bbox_before_softmax.reshape(-1, reg_max + 1),
                    pseudo_dfl_targets_hard.reshape(-1),
                    weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                    avg_factor=4.0 * norm_factor,
                )
                
            else:
                # 2. soft pseudo-label
                pred_probs = F.softmax(bbox_before_softmax.detach(), dim=-1)
                soft_targets = pred_probs.view(-1, 4, reg_max + 1)
                student_logits = bbox_before_softmax.view(-1, 4, reg_max + 1) 
                
                # Optional temperature (sharpening); T=1.0 means no change
                T = 1.0
                log_q = F.log_softmax(student_logits / T, dim=-1)                          # (n_pos, 4, K+1)
                p   = F.softmax(soft_targets / T, dim=-1)                                  # (n_pos, 4, K+1)
                
                # KL divergence per side, then sum over bins
                kl_per_side = F.kl_div(log_q, p, reduction='none').sum(dim=-1)             # (n_pos, 4)

                # Weighting like your DFL call
                w = weight_targets[:, None].expand(-1, 4)                                  # (n_pos, 4)
                loss_dfl_self = (kl_per_side * w).sum() / (4.0 * norm_factor)
            # loss_dfl_self=0 (1 - self.lambda_) *
            total_loss = (1 - self.lambda_) * loss_new + self.lambda_ * (loss_qfl_self + loss_dfl_self)
            
            self.total_flops += (len(targets) * self.forward_flops)
        
        return total_loss, loss_item