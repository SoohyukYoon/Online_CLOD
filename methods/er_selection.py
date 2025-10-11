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
from utils.data_loader import SelectionMemoryDataset
from utils.train_utils import select_model, select_optimizer, select_scheduler
import torch.nn.functional as F

from torch import Tensor
from typing import Generator, List, Tuple, Union

logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class SampleSelectionBase(ER):
    def __init__(self, criterion, n_classes, device, **kwargs):
        super().__init__(criterion, n_classes, device, **kwargs)
    
    def initialize_memory_buffer(self, memory_size):
        self.memory_size = memory_size - self.temp_batchsize
        self.new_exposed_classes = ['pretrained']
        self.lambda_ = 0.5
        self.use_hardlabel = True
        
        data_args = self.damo_cfg.get_data(self.damo_cfg.dataset.train_ann[0])
        self.memory = SelectionMemoryDataset(ann_file=data_args['args']['ann_file'], root=data_args['args']['root'], transforms=None,class_names=self.damo_cfg.dataset.class_names,
            dataset=self.dataset, cls_list=self.exposed_classes, device=self.device, memory_size=self.memory_size, image_size=self.img_size, aug=self.damo_cfg.train.augment, selection_method=self.selection_method, priority_selection=self.priority_selection)
        print("begin cal_initial_info")
        buffer_initial_info = self.cal_initial_info()
        print("attained buffer_initial_info", len(buffer_initial_info))
        assert len(self.memory.buffer) == len(buffer_initial_info), "Buffer size and initial info size mismatch."
        print("end cal_initial_info")
        self.memory.update_initialinfo(buffer_initial_info)
        

    def cal_initial_info(self):
        """
        Calculate the initial information for each sample in the buffer.
        This is a placeholder function that should be implemented based on the selection method.
        """
        # buffer_initial_data2 = self.memory.get_buffer_data()
        buffer_initial_info = []
        # self.model.eval()
        # with torch.no_grad():
        for i in range(0,len(self.memory.buffer), self.batch_size):
            batch = self.memory.get_buffer_data(i, self.batch_size)
            # batch = buffer_initial_data2[i:i+self.batch_size]
            # data = collate_fn(batch)
            # print("data", len(batch), self.batch_size)
            # batch = {
            #     "img": data[0],         # images
            #     "cls": data[1],         # labels
            #     "img_id": data[2],    # image paths
            # }
            self.optimizer.zero_grad()
            inps, targets = self.preprocess_batch(batch)
            with torch.cuda.amp.autocast(enabled=False):
                inputs = inps.tensors
                infos = []
                loss = None
                if "loss" in self.selection_method:
                    for ind in range(len(targets)):
                        loss_item = self.model(inputs[ind], [targets[ind]])["total_loss"]
                        infos.append(loss_item.detach().cpu().item())
                        self.optimizer.zero_grad()

                elif "entropy" in self.selection_method:
                    for ind in range(len(targets)):
                        loss, sample_logit = self.model(inputs[ind], [targets[ind]], get_features=True)
                        sample_logit = sample_logit[-1]
                        probs = F.softmax(sample_logit, dim=0)
                        info = -torch.sum(probs * torch.log(probs + 1e-8)).item()
                        infos.append(info)
                        self.optimizer.zero_grad()
                
                elif "fisher" in self.selection_method:
                    for ind in range(len(targets)):
                        self.optimizer.zero_grad()
                        loss = self.model(inputs[ind],[targets[ind]])["total_loss"]
                        loss.backward()
                        # for n, p in self.model.neck.named_parameters():
                        #     if p.requires_grad == True:
                        #         grad = torch.autograd.grad(loss, p, retain_graph=True)[0].clone().detach().clamp(-1, 1)
                        info = 0
                        for n, p in self.model.named_parameters():
                            if "neck" in n and p.requires_grad == True:
                                grad = p.grad.detach().cpu().numpy()
                                info += (grad**2).sum()
                        infos.append(info)
                        self.optimizer.zero_grad()
                
            buffer_initial_info.extend(infos)      
        return buffer_initial_info
    
    
    
    def add_new_class(self, class_name):
        super().add_new_class(class_name)
        self.new_exposed_classes.append(class_name)
        self.memory.new_exposed_classes = self.new_exposed_classes
    
    
    def model_forward_samplewise(self, batch):
        inps, targets = self.preprocess_batch(batch)

        with torch.cuda.amp.autocast(enabled=False):
            # 모델 실행: output = {"AUX": ..., "Main": ...}
            # for input_s in batch["img"]:
            # outputs = self.model(batch["img"])
            # aux_raw = outputs["AUX"]
            # main_raw = outputs["Main"]

            # Vec2Box 변환: [B, A, C], [B, A, R], [B, A, 4]
            # aux_predicts = self.vec2box(aux_raw)
            # main_predicts = self.vec2box(main_raw)
            image_tensors = inps.tensors
        
            # 손실 계산
            infos = []
            loss = None
            
            if "loss" in self.selection_method:
                for ind in range(len(targets)):
                    loss_item = self.model(image_tensors[ind], [targets[ind]])
                    sample_loss = loss_item["total_loss"]
                    infos.append(sample_loss.detach().cpu().item())
                    
                    # infos.append(sample_loss.detach().cpu().item())
                    if loss == None:
                        loss = sample_loss
                    else:
                        loss += sample_loss
                        
                loss /= len(image_tensors)
                    
            elif "entropy" in self.selection_method:
                for ind in range(len(targets)):
                    loss_item, sample_logit = self.model(image_tensors[ind], [targets[ind]], get_features=True)
                    
                    sample_logit = sample_logit[-1]
                    sample_loss = loss_item["total_loss"]
                    probs = F.softmax(sample_logit, dim=0)
                    info = -torch.sum(probs * torch.log(probs + 1e-8)).item()
                    infos.append(info)
                    
                    if loss == None:
                        loss = sample_loss
                    else:
                        loss += sample_loss
                        
                loss /= len(image_tensors)
            
            elif "fisher" in self.selection_method:
                for ind in range(len(targets)):
                    self.optimizer.zero_grad()
                    loss_item = self.model(image_tensors[ind], [targets[ind]])
                    sample_loss = loss_item["total_loss"]
                    sample_loss.backward(retain_graph=True)
                    # for n, p in self.model.neck.named_parameters():
                    #     if p.requires_grad == True:
                    #         grad = torch.autograd.grad(sample_loss, p, retain_graph=True)[0].clone().detach().clamp(-1, 1)
                    # info = (grad**2).sum().cpu()
                    info = 0
                    for n, p in self.model.named_parameters():
                        if "neck" in n and p.requires_grad == True:
                            grad = p.grad.detach().cpu().numpy()
                            info += (grad**2).sum()
                    infos.append(info)
                    
                    if loss == None:
                        loss = sample_loss
                    else:
                        loss += sample_loss
                    
                loss /= len(image_tensors)
                
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.total_flops += self.backward_flops
            
            self.update_schedule()
                    
            
        return loss, infos
    
    
    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        total_loss = 0.0
        if len(sample) > 0:
            self.memory.register_stream(sample)
        for i in range(iterations):
            self.model.train()
            batch = self.memory.get_batch(batch_size, stream_batch_size)

            # batch = {
            #         "img": data[0],         # images
            #         "cls": data[1],         # labels
            #         "img_id": data[2],    # image paths
            #     }
            
            # print(f"[DEBUG] batch images: {batch['img_path']}")
            # print(f"[DEBUG] batch labels shape: {batch['cls'].shape}")

            self.optimizer.zero_grad()
            
            info = []
            loss, info = self.model_forward_samplewise(batch)
            # infos.extend(info)
        
            # else:
            #     # print(f"[DEBUG] individual losses: {loss_item}")
            #     if self.use_amp:
            #         self.scaler.scale(loss).backward()
            #         self.scaler.step(self.optimizer)
            #         self.scaler.update()
            #     else:
            #         loss.backward()
            #         self.optimizer.step()
                
            #     self.total_flops += (len(data[1]) * self.backward_flops)
                
            #     self.update_schedule()

            total_loss += loss.item()
            stream_info = info[:len(self.temp_batch)]
            memory_info = info[len(self.temp_batch):]
            self.memory.update_info(memory_info)
                # self.total_flops += (batch_size * (self.forward_flops + self.backward_flops))
                # print("self.total_flops", self.total_flops)
        return total_loss / iterations, stream_info
   
    
    
    
    def online_step(self, sample, sample_num, n_worker):
        if sample.get('klass',None) and sample['klass'] not in self.exposed_classes:
            self.online_after_task(sample_num)
            self.add_new_class(sample['klass'])
        elif sample.get('domain',None) and sample['domain'] not in self.exposed_domains:
            self.exposed_domains.append(sample['domain'])
            # self.new_exposed_classes.append(sample['domain'])
            # self.memory.new_exposed_classes.append(sample['domain'])
            # self.memory.cls_count.append(0)
            # self.memory.cls_idx.append([])
        
        self.temp_batch.append(sample)
        self.num_updates += self.online_iter
        if len(self.temp_batch) == self.temp_batchsize:
            iteration = int(self.num_updates)
            if iteration != 0:
                train_loss, infos = self.online_train(self.temp_batch, self.batch_size, n_worker,
                                                      iterations=int(self.num_updates), stream_batch_size=self.temp_batchsize)
                self.report_training(sample_num, train_loss)
                
                for s_i, stored_sample in enumerate(self.temp_batch):
                    self.update_memory(stored_sample, infos[s_i])

                self.temp_batch = []
                self.num_updates -= int(self.num_updates)
    
    def update_memory(self, sample, info):
        self.reservoir_memory(sample, info)

    def reservoir_memory(self, sample, info):
        self.seen += 1
        if len(self.memory) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                self.memory.replace_sample(sample, info, j)#, mode=self.mode, online_iter=self.online_iter)
        else:
            self.memory.replace_sample(sample, info)#, mode=self.mode, online_iter=self.online_iter)


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
            label_scores[mask] = cls_scores[mask, labels[mask]]
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
                expected = expected.sum(dim=-1)     # (N, 4)
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
            # loss_dfl_self=0
            total_loss = (1 - self.lambda_) * loss_new + self.lambda_ * (loss_qfl_self + loss_dfl_self)
            
            self.total_flops += (len(targets) * self.forward_flops)
        
        return total_loss, loss_item