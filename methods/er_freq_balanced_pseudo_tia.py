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
from utils.data_loader import FreqDataset, MemoryPseudoDataset, FreqClsBalancedPseudoDataset, FreqClsBalancedPseudoDomainDataset
from utils.train_utils import select_model, select_optimizer, select_scheduler

from collections import OrderedDict
from torch.autograd import Function

logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class ERFreqBalancedPseudo(ER):
    def initialize_memory_buffer(self, memory_size):
        self.memory_size = memory_size - self.temp_batchsize
        data_args = self.damo_cfg.get_data(self.damo_cfg.dataset.train_ann[0])
        self.memory = FreqClsBalancedPseudoDomainDataset(ann_file=data_args['args']['ann_file'], root=data_args['args']['root'], transforms=None,class_names=self.damo_cfg.dataset.class_names,
            dataset=self.dataset, cls_list=self.exposed_classes, device=self.device, memory_size=self.memory_size, image_size=self.img_size, aug=self.damo_cfg.train.augment)
        
        self.ema_ratio=0.995
        self.ema_model = copy.deepcopy(self.model)
        self.new_exposed_classes = ['pretrained']
        
    def copy_model_head(self):
        self.ema_model.head.gfl_cls[0] = copy.deepcopy(self.model.head.gfl_cls[0])
        self.ema_model.head.gfl_cls[1] = copy.deepcopy(self.model.head.gfl_cls[1])
        self.ema_model.head.gfl_cls[2] = copy.deepcopy(self.model.head.gfl_cls[2])
    def add_new_class(self, class_name):
        super().add_new_class(class_name)
        self.new_exposed_classes.append(class_name)
        self.memory.new_exposed_classes = self.new_exposed_classes
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
        if len(self.memory.buffer) >= self.memory_size:
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

import torch.nn.functional as F

class SupervisedContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.1, base_temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, embeddings, labels):
        anchor_count = embeddings.size(0)
        # Normalize the embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        # Compute the similarity matrix

        sim_matrix = torch.div(torch.matmul(embeddings, embeddings.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Create the positive mask
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float()
        # mask = mask.repeat(anchor_count, anchor_count)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(anchor_count).view(-1, 1).to(embeddings.device),
            0
        )
        # logits_mask = torch.ones_like(mask)
        # mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)+1e-6)

        log_prob[logits_mask==0] = log_prob[logits_mask==0]*0.0

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = torch.nan_to_num(loss, nan=0.0)
        loss = loss.mean()
        return loss
    

class Filter_conv1(nn.Module):
    def __init__(self, inputsize):
        super(Filter_conv1, self).__init__()
        
        self.hidden = nn.Linear(inputsize, inputsize//2)
        self.hidden2 = nn.Linear(inputsize//2, inputsize//4)
        self.output = nn.Linear(inputsize//4, 256)
        self.leakyrelu = nn.LeakyReLU()
        self.fc = nn.Linear(256, 2)
    # @autocast()
    def forward(self, x):
        x = self.hidden(x)
        x = self.leakyrelu(x)
        x = self.hidden2(x)
        x = self.leakyrelu(x)
        x = self.output(x)

        return x, self.fc(self.leakyrelu(x))


import torch
import torchvision.transforms as TT
import numpy as np
from numpy import random as R
import cv2

class PatchbasedAug:
    def __init__(self):
        self.gaussian = TT.GaussianBlur(11,(0.1,2.0))

    def mask_img(self,img,cln_img):
        while R.random()>0.4:
            x1 = R.randint(img.shape[1])
            x2 = R.randint(img.shape[1])
            y1 = R.randint(img.shape[2])
            y2 = R.randint(img.shape[2])
            img[:,x1:x2,y1:y2]=cln_img[:,x1:x2,y1:y2]
        return img

    def gaussian_heatmap(self,x):
        """
        It produces single gaussian at a random point
        """
        sig = torch.randint(low=1,high=150,size=(1,)).cuda()[0]
        image_size = x.shape[1:]
        center = (torch.randint(image_size[0],(1,))[0].cuda(), torch.randint(image_size[1],(1,))[0].cuda())
        x_axis = torch.linspace(0, image_size[0]-1, image_size[0]).cuda() - center[0]
        y_axis = torch.linspace(0, image_size[1]-1, image_size[1]).cuda() - center[1]
        xx, yy = torch.meshgrid(x_axis, y_axis)
        kernel = torch.exp(-0.5 * (torch.square(xx) + torch.square(yy)) / torch.square(sig))
        new_img = (x*(1-kernel) + 255*kernel).type(torch.uint8)
        return new_img

    def aug(self,x):
        for sample in x:
            img = sample['image'].cuda()
            g_b_flag = True

            # Guassian Blur
            if R.random()>0.5:
                img = self.gaussian(img)

            cln_img_zero = img.detach().clone()

            # Gamma
            if R.random()>0.5:
                cln_img = img.detach().clone()
                val = 1/(R.random()*0.8+0.2) if R.random() > 0.5 else R.random()*0.8+0.2
                img = TT.functional.adjust_gamma(img,val)
                img= self.mask_img(img,cln_img)
                g_b_flag = False
            
            # Brightness
            if R.random()>0.5 or g_b_flag:
                cln_img = img.detach().clone()
                val = R.random()*1.6+0.2
                img = TT.functional.adjust_brightness(img,val)
                img= self.mask_img(img,cln_img)

            # Contrast
            if R.random()>0.5:
                cln_img = img.detach().clone()
                val = R.random()*1.6+0.2
                img = TT.functional.adjust_contrast(img,val)
                img= self.mask_img(img,cln_img)
            img= self.mask_img(img,cln_img_zero)

            # glare
            prob = 0.5
            while R.random()>prob:
                img=self.gaussian_heatmap(img)
                prob+=0.1

            #Noise
            if R.random()>0.5:
                n = torch.clamp(torch.normal(0,R.randint(50),img.shape),min=0).cuda()
                img = n + img
                img = torch.clamp(img,max = 255).type(torch.uint8)

            sample['image'] = img.cpu()
        return x

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None