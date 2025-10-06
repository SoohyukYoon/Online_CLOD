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
from utils.data_loader import FreqDataset, MemoryPseudoDataset, FreqClsBalancedPseudoDataset, FreqClsBalancedPseudoDomainDataset
from utils.train_utils import select_model, select_optimizer, select_scheduler

from collections import OrderedDict


logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class BASELINEFreqBalancedPseudoGRAM(BASELINE2):
    def initialize_memory_buffer(self, memory_size):
        self.memory_size = memory_size - 1
        data_args = self.damo_cfg.get_data(self.damo_cfg.dataset.train_ann[0])
        self.memory = FreqClsBalancedPseudoDomainDataset(ann_file=data_args['args']['ann_file'], root=data_args['args']['root'], transforms=None,class_names=self.damo_cfg.dataset.class_names,
            dataset=self.dataset, cls_list=self.exposed_classes, device=self.device, memory_size=self.memory_size, init_buffer_size=(memory_size + 1) // 2, image_size=self.img_size, aug=self.damo_cfg.train.augment)
        
        self.ema_ratio=0.995
        self.ema_model = copy.deepcopy(self.model)
        self.new_exposed_classes = [] #'pretrained'
        
        self.MSE = MSELoss(num_classes=1)
        self.CE = CrossEntropyLoss(num_classes=2)
        self.FL = FocalLoss(num_classes=2, gamma=3.0)
        self.Align = FocalLoss(num_classes=2, gamma=3.0)
        self.supcon = SupervisedContrastiveLoss()

        self.DA_filter = Filter_conv1(32896).to(self.device)
        
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
        
        self.num_updates += self.online_iter
        if self.num_updates >= 1:
            train_loss = self.online_train([sample], self.batch_size, n_worker, iterations=int(self.num_updates), stream_batch_size=1)
            self.report_training(sample_num, train_loss)
            self.num_updates -= int(self.num_updates)
            self.update_schedule()
        
        self.update_memory(sample)
    
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
        
        src_batch_size = min(len(self.memory.buffer_source), int(batch_size/2)) # half or full
        tgt_batch_size = min(len(self.memory.buffer), int(batch_size/2 - stream_batch_size))
        for i in range(iterations):
            # data = self.memory.get_batch(batch_size, stream_batch_size, model=self.ema_model, score_thresh=0.7)
            tgt_memory_batch = self.memory.get_tgt_batch(tgt_batch_size, stream_batch_size, model=self.ema_model, score_thresh=0.5)
            
            src_memory_batch, src_mixup_memory_batch, src_obj_mixup_memory_batch = self.memory.get_src_batch(src_batch_size, tgt_batch=tgt_memory_batch[3], model=self.ema_model, score_thres=0.5) if src_batch_size > 0 else []
            
            self.model.train()
            self.optimizer.zero_grad()

            supervised_inps, supervised_targets, unsup_inps, unsup_targets, mixup_ratio = self.preprocess_batch((tgt_memory_batch, src_memory_batch, src_mixup_memory_batch, src_obj_mixup_memory_batch))

            # with torch.cuda.amp.autocast(enabled=self.use_amp):
            with torch.cuda.amp.autocast(enabled=False):
                loss_item, (base_feat1_src, base_feat2_src, base_feat3_src), (img_feat1_src, img_feat2_src, img_feat3_src) = self.model(supervised_inps, supervised_targets)
                total_loss = loss_item["total_loss"]
                self.total_flops += (len(supervised_targets) * self.forward_flops)
                
                pure_src_img_feat1, mixed_src_img1 = img_feat1_src[:src_batch_size], img_feat1_src[src_batch_size:]
                pure_src_img_feat2, mixed_src_img2 = img_feat2_src[:src_batch_size], img_feat2_src[src_batch_size:]
                pure_src_img_feat3, mixed_src_img3 = img_feat3_src[:src_batch_size], img_feat3_src[src_batch_size:]

                da_src_img_loss1 = torch.mean(pure_src_img_feat1 ** 2)
                da_src_img_loss2 = self.CE(pure_src_img_feat2, domain=0)
                da_src_img_loss3 = self.FL(pure_src_img_feat3, domain=0)
                
                da_mixed_img_loss1 = torch.mean((mixed_src_img1 - mixup_ratio.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) ** 2)
                da_mixed_img_loss2 = self.CE(mixed_src_img2, domain=[1-mixup_ratio, mixup_ratio])
                da_mixed_img_loss3 = self.FL(mixed_src_img3, domain=[1-mixup_ratio, mixup_ratio])

                da_img_loss =  (
                        da_src_img_loss1+da_mixed_img_loss1+
                        da_src_img_loss2* 0.15+da_mixed_img_loss2* 0.15+
                        da_src_img_loss3+da_mixed_img_loss3
                    )
                pseudo_loss, (base_feat1_tgt, base_feat2_tgt, base_feat3_tgt), (img_feat1_tgt, img_feat2_tgt, img_feat3_tgt) = self.model(unsup_inps, unsup_targets)
                
                # src_img_feat1, tgt_img_feat1 = img_feat1_tgt[:src_batch_size], img_feat1_tgt[src_batch_size:]
                # src_img_feat2, tgt_img_feat2 = img_feat2_tgt[:src_batch_size], img_feat2_tgt[src_batch_size:]
                # src_img_feat3, tgt_img_feat3 = img_feat3_tgt[:src_batch_size], img_feat3_tgt[src_batch_size:]
                tgt_img_feat1, tgt_img_feat2, tgt_img_feat3 = img_feat1_tgt, img_feat2_tgt, img_feat3_tgt

                da_tgt_img_loss1 = torch.mean((1 - tgt_img_feat1) ** 2)
                da_tgt_img_loss2 = self.CE(tgt_img_feat2, domain=1)
                da_tgt_img_loss3 = self.FL(tgt_img_feat3, domain=1)

                da_img_loss += (
                    da_tgt_img_loss1+
                    0.15*da_tgt_img_loss2+
                    da_tgt_img_loss3
                )
                
            loss = total_loss + da_img_loss / 3 + pseudo_loss["total_loss"] * 0.2

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.total_flops += ((src_batch_size+tgt_batch_size) * self.backward_flops)
            
            self.update_schedule()
            self.update_ema_model(num_updates=1.0)

            total_loss += loss.item()
            
        return total_loss / iterations

    
    def preprocess_batch(self, batch):
        tgt_memory_batch, src_memory_batch, src_mixup_memory_batch, src_obj_mixup_memory_batch = batch
        # supervised_inps = torch.cat((src_memory_batch[0].to(self.device), src_mixup_memory_batch[0].to(self.device)),dim=0)
        supervised_inps = src_memory_batch[0].to(self.device)
        supervised_inps.tensors = torch.cat((supervised_inps.tensors, src_mixup_memory_batch[0].tensors.to(self.device)),dim=0)
        supervised_inps.image_sizes = src_memory_batch[0].image_sizes + src_mixup_memory_batch[0].image_sizes
        supervised_inps.pad_sizes = src_memory_batch[0].pad_sizes + src_mixup_memory_batch[0].pad_sizes
        supervised_targets = [target.to(self.device) for target in (src_memory_batch[1] + src_mixup_memory_batch[1])]
        
        # unsup_inps = src_obj_mixup_memory_batch[0].to(self.device)
        # unsup_inps.tensors = torch.cat((unsup_inps.tensors, tgt_memory_batch[0].tensors.to(self.device)),dim=0)
        # unsup_inps.image_sizes = src_obj_mixup_memory_batch[0].image_sizes + tgt_memory_batch[0].image_sizes
        # unsup_inps.pad_sizes = src_obj_mixup_memory_batch[0].pad_sizes + tgt_memory_batch[0].pad_sizes
        # unsup_targets = [target.to(self.device) for target in (src_obj_mixup_memory_batch[1] + tgt_memory_batch[1])]
        unsup_inps = tgt_memory_batch[0].to(self.device)
        unsup_targets = [target.to(self.device) for target in tgt_memory_batch[1]]
        
        
        mixup_ratio = torch.tensor(src_mixup_memory_batch[4]).to(self.device)
        
        return supervised_inps, supervised_targets, unsup_inps, unsup_targets, mixup_ratio
    
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
from torch.autograd import Function
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
    
class CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=2):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        print(f'Softmax Cross Entropy Loss with num_classes={self.num_classes}')
    # @autocast()
    def forward(self, input, domain):
        B, C = input.shape
        probs = F.softmax(input, 1)
        if type(domain) == int:
            label = F.one_hot(torch.tensor(domain).repeat(B).cuda(), self.num_classes).float()
        elif type(domain) == list:
            assert len(domain) == self.num_classes
            if type(domain[0]) == int:
                label = torch.tensor(domain).repeat(B).reshape(-1, self.num_classes).cuda()
            elif type(domain[0]) == torch.Tensor:
                label = torch.stack(domain, dim=-1).cuda()

        loss = -(probs.log() * label).sum(-1)
        return loss.mean(-1)


class FocalLoss(nn.Module):
    def __init__(self, num_classes=2, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        print(f'Focal Loss with gamma={self.gamma} & num_classes={self.num_classes}')
    # @autocast()
    def forward(self, input, domain):
        B, C = input.shape
        probs = F.softmax(input, 1)
        if type(domain) == int:
            label = F.one_hot(torch.tensor(domain).repeat(B).cuda(), self.num_classes).float()
        elif type(domain) == list:
            assert len(domain) == self.num_classes
            if type(domain[0]) == int:
                label = torch.tensor(domain).repeat(B).reshape(-1, self.num_classes).cuda()
            elif type(domain[0]) == torch.Tensor:
                label = torch.stack(domain, dim=-1).cuda()

        probs = (probs * label).sum(-1)
        loss = -torch.pow(1 - probs, self.gamma) * probs.log()
        return loss.mean()

class MSELoss(nn.Module):
    def __init__(self, num_classes=1):
        super(MSELoss, self).__init__()
        self.num_classes = num_classes
        print(f'MSE Loss with num_classes={self.num_classes}')
    # @autocast()
    def forward(self, input, domain):
        B, C, H, W = input.shape
        # probs = F.softmax(input, 1)
        # probs = input
        # if type(domain) == int:
        #     label = F.one_hot(torch.tensor(domain).repeat(B).cuda(), self.num_classes).reshape(B, C, 1, 1).float()
        # elif type(domain) == list:
        #     assert len(domain) == self.num_classes
        #     label = torch.tensor(domain).repeat(B).reshape(-1, self.num_classes).reshape(B, C, 1, 1).cuda()

        # loss = (probs - label)**2
        loss = (input - domain[0])**2
        return loss.mean()

class GramMatrixExtractor:
    """Extract Gram matrices from feature maps for specific bounding box regions."""
    
    def __init__(self, input_size=640):
        self.input_size = input_size
        self.feat_sizes = {
            1: 80,  # feature map 1: 80x80
            2: 40,  # feature map 2: 40x40
            3: 20   # feature map 3: 20x20
        }
    
    def compute_gram_matrix(self, features):
        """
        Compute Gram matrix for feature maps.
        Args:
            features: tensor of shape (batch_size, channels, height, width)
        Returns:
            gram: tensor of shape (batch_size, channels, channels)
        """
        b, c, h, w = features.size()
        # Reshape to (batch_size, channels, height*width)
        features = features.view(b, c, -1)
        # Compute Gram matrix: (batch_size, channels, channels)
        gram = torch.bmm(features, features.transpose(1, 2))
        # Normalize by the number of elements
        gram = gram / (c * h * w)
        return gram
    
    def extract_bbox_features(self, feature_map, targets, feat_level):
        """
        Extract features from bounding box regions.
        Args:
            feature_map: tensor of shape (batch_size, channels, height, width)
            targets: list of target dictionaries with 'bboxes' key
            feat_level: feature level (1, 2, or 3)
        Returns:
            bbox_features: list of tensors, each for one bbox
        """
        batch_size, channels, feat_h, feat_w = feature_map.size()
        feat_size = self.feat_sizes[feat_level]
        scale = feat_size / self.input_size
        
        bbox_features = []
        bbox_labels = []  # Track which batch each bbox comes from
        
        for batch_idx in range(batch_size):
            if batch_idx < len(targets) and 'bboxes' in targets[batch_idx]:
                bboxes = targets[batch_idx]['bboxes']  # Assuming shape (N, 4) with [x1, y1, x2, y2]
                
                for bbox in bboxes:
                    # Convert bbox coordinates to feature map coordinates
                    x1, y1, x2, y2 = bbox
                    x1_feat = int(x1 * scale)
                    y1_feat = int(y1 * scale)
                    x2_feat = int(x2 * scale)
                    y2_feat = int(y2 * scale)
                    
                    # Ensure coordinates are within bounds
                    x1_feat = max(0, min(x1_feat, feat_w - 1))
                    y1_feat = max(0, min(y1_feat, feat_h - 1))
                    x2_feat = max(x1_feat + 1, min(x2_feat, feat_w))
                    y2_feat = max(y1_feat + 1, min(y2_feat, feat_h))
                    
                    # Extract region
                    region = feature_map[batch_idx:batch_idx+1, :, y1_feat:y2_feat, x1_feat:x2_feat]
                    
                    if region.numel() > 0:  # Check if region is not empty
                        bbox_features.append(region)
                        bbox_labels.append(batch_idx)
        
        return bbox_features, bbox_labels
    
    def compute_gram_matrices_for_targets(self, feature_maps, targets):
        """
        Compute Gram matrices for all target bounding boxes across feature levels.
        Args:
            feature_maps: tuple of (feat1, feat2, feat3)
            targets: list of target dictionaries
        Returns:
            gram_matrices: dict with keys 1, 2, 3 for each feature level
            labels: list of batch indices for each gram matrix
        """
        gram_matrices = {}
        all_labels = []
        
        for feat_level, feat_map in enumerate(feature_maps, 1):
            bbox_features, bbox_labels = self.extract_bbox_features(feat_map, targets, feat_level)
            
            if bbox_features:
                # Compute Gram matrix for each bbox region
                gram_list = []
                for bbox_feat in bbox_features:
                    gram = self.compute_gram_matrix(bbox_feat)
                    # Flatten the gram matrix for contrastive loss
                    gram_flat = gram.view(1, -1)  # Shape: (1, channels*channels)
                    gram_list.append(gram_flat)
                
                gram_matrices[feat_level] = torch.cat(gram_list, dim=0)  # Shape: (n_boxes, channels*channels)
                
                if feat_level == 1:  # Only store labels once
                    all_labels = bbox_labels
        
        return gram_matrices, all_labels