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

from typing import Generator, List, Tuple, Union
from torch import Tensor
import torch.nn.functional as F

logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tensor]]:
    """
    A collate function to handle batching of images and their corresponding targets.

    Args:
        batch (list of tuples): Each tuple contains:
            - image (Tensor): The image tensor.
            - labels (Tensor): The tensor of labels for the image.

    Returns:
        Tuple[Tensor, List[Tensor]]: A tuple containing:
            - A tensor of batched images.
            - A list of tensors, each corresponding to bboxes for each image in the batch.
    """
    batch_size = len(batch)
    target_sizes = [item[1].size(0) for item in batch]
    # TODO: Improve readability of these process
    # TODO: remove maxBbox or reduce loss function memory usage
    batch_targets = torch.zeros(batch_size, min(max(target_sizes), 100), 5)
    batch_targets[:, :, 0] = -1
    for idx, target_size in enumerate(target_sizes):
        batch_targets[idx, : min(target_size, 100)] = batch[idx][1][:100]

    batch_images, _, batch_reverse, batch_path = zip(*batch)
    batch_images = torch.stack(batch_images)
    batch_reverse = torch.stack(batch_reverse)

    return batch_size, batch_images, batch_targets, batch_reverse, batch_path

class OursMemoryDataset(MemoryDataset):
    def __init__(self, args, dataset, cls_list=None, device=None, data_dir=None, memory_size=None):
        self.buffer_info = []
        self.softmax_retrieval = 1
        print("initialize buffer")
        super().__init__(args, dataset, cls_list=cls_list, device=device, data_dir=data_dir, memory_size=memory_size)
    
    def replace_sample(self, sample, idx=None, images_dir=None,label_path=None, info=None):
        img, labels, image_path, ratio = self.load_data(sample['file_name'], image_dir=images_dir or self.image_dir, label_path=label_path or self.label_path)
        data = (img, labels, image_path, ratio)
        if idx is None:
            self.buffer.append(data)
            self.buffer_info.append(info)
        else:
            self.buffer[idx] = data
            self.buffer_info[idx] = info
     
    @torch.no_grad()
    def get_batch(self, batch_size, stream_batch_size=0, use_weight=None, transform=None, weight_method=None):
        assert batch_size >= stream_batch_size
        stream_batch_size = min(stream_batch_size, len(self.stream_data))
        batch_size = min(batch_size, stream_batch_size + len(self.buffer))
        memory_batch_size = batch_size - stream_batch_size

        data = []

        if stream_batch_size > 0:
            stream_indices = np.random.choice(range(len(self.stream_data)), size=stream_batch_size, replace=False)
            for i in stream_indices:
                img, bboxes, img_path, _ = self.stream_data[i]
                img, bboxes, rev_tensor = self.transform(img, bboxes)
                bboxes[:, [1, 3]] *= self.image_sizes[0]
                bboxes[:, [2, 4]] *= self.image_sizes[1]
                data.append((img, bboxes, rev_tensor, img_path))

        if memory_batch_size > 0:
            # indices = np.random.choice(range(len(self.buffer)), size=memory_batch_size, replace=False)
            # indices = np.argsort(self.buffer_info)[-memory_batch_size:][::-1]
            
                
            #     # batch_info_prob = buffer_info_tensor / self.softmax_retrieval
            #     # softmax_probs = torch.nn.Softmax(dim=0)(batch_info_prob).tolist()
            non_none_item_ind = [i for i, val in enumerate(self.buffer_info) if val is not None]
            non_none_item = [val for val in self.buffer_info if val is not None]
            
            if len(non_none_item) < memory_batch_size:
                print("not enough")
                indices = np.random.choice(range(len(self.buffer)), size=memory_batch_size, replace=False)
            else:
                print("enough")
                print(non_none_item)
                # softmax_probs = [info/sum(non_none_item) for info in non_none_item]
                # memory_indices = np.random.choice(len(non_none_item), memory_batch_size, p=softmax_probs, replace=False)
                
                memory_indices = np.argsort(non_none_item)[-memory_batch_size:][::-1]
                
                indices = [non_none_item_ind[ind] for ind in memory_indices]
                      
                for i in indices:
                    img, bboxes, img_path = self.get_data(i)
                    img, bboxes, rev_tensor = self.transform(img, bboxes)
                    bboxes[:, [1, 3]] *= self.image_sizes[0]
                    bboxes[:, [2, 4]] *= self.image_sizes[1]
                    data.append((img, bboxes, rev_tensor, img_path))

        return collate_fn(data)

class OursMin(ER):
    def __init__(self, criterion, n_classes, device, **kwargs):
        super().__init__(criterion, n_classes, device, **kwargs)
        self.memory = OursMemoryDataset(self.args, self.dataset, self.exposed_classes, device=self.device, memory_size=self.memory_size, mosaic_prob=kwargs['mosaic_prob'],mixup_prob=kwargs['mixup_prob'])
        self.selection_method = kwargs["selection_method"]
        
    def model_forward_samplewise(self, batch):
        batch = self.preprocess_batch(batch)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # 모델 실행: output = {"AUX": ..., "Main": ...}
            # for input_s in batch["img"]:
            print("batchsize", len(batch["img"]))
            outputs = self.model(batch["img"])
            aux_raw = outputs["AUX"]
            main_raw = outputs["Main"]

            # Vec2Box 변환: [B, A, C], [B, A, R], [B, A, 4]
            aux_predicts = self.vec2box(aux_raw)
            main_predicts = self.vec2box(main_raw)
            
            # C_old = aux_predicts[0].shape[-1]
            # logits = aux_predicts[0][..., :C_old]
            
            # T = 2.0
            
            # if self.select_method == "loss":
            #     info = loss
            # with torch.no_grad():
            #     outputs = self.model(batch["img"])
            #     for output in outputs:
            #         logits = output[0] if isinstance(output, tuple) else output
            #         soft = F.softmax(logits / T, dim=1)
            #         entropy = -1 * soft * torch.log(soft + 1e-10)
            #         entropy = torch.mean(torch.sum(entropy, dim=-1), dim=-1).detach().squeeze()
            

            # targets: [B, T, 5] (cls, cx, cy, w, h) → xyxy
            # targets = batch["cls"].clone()
            # x, y, w, h = targets[..., 1:].unbind(-1)
            # targets[..., 1] = x
            # targets[..., 2] = y
            # targets[..., 3] = x + w
            # targets[..., 4] = y + h

            # 손실 계산
            infos = []
            if self.selection_method == "loss":
                loss = None
                for ind in range(len(batch["img"])):
                    aux_predict = [predict[ind].unsqueeze(dim=0) for predict in aux_predicts]
                    main_predict = [predict[ind].unsqueeze(dim=0) for predict in main_predicts]
                    sample_loss, loss_item = self.model.loss_fn(aux_predict, main_predict, batch['cls'][ind].unsqueeze(dim=0))
                    infos.append(sample_loss.detach().cpu().item())
                    # loss.append(sample_loss.detach().cpu())
                    if loss == None:
                        loss = sample_loss
                    else:
                        loss += sample_loss
                        
                # loss= torch.tensor(loss).mean(dim=0)
                loss /= len(batch["img"])
      
                # infos= torch.tensor(infos).mean(dim=0)        
                    
            elif self.selection_method == "entropy":
                loss, loss_item = self.model.loss_fn(aux_predicts, main_predicts, batch['cls'])
                for ind in range(len(batch["img"])):
                    info = 0
                    for main_raw_i in main_raw:
                        sample_logit = main_raw_i[0][ind] if isinstance(main_raw_i, tuple) else main_raw_i[ind]
                        probs = F.softmax(sample_logit, dim=0)
                        info += -torch.sum(probs * torch.log(probs + 1e-8)).item()
                    infos.append(info)
            
            elif self.selection_method == "gradnorm":
                loss, loss_item = self.model.loss_fn(aux_predicts, main_predicts, batch['cls'])
                for ind in range(len(batch["img"])):
                    info = 0
                    for main_raw_i in main_raw:
                        sample_logit = main_raw_i[0][ind] if isinstance(main_raw_i, tuple) else main_raw_i[ind]
                        for n, p in self.model.base_model.model.model.layers[-1].named_parameters():
                            if p.requires_grad == True:
                                grad = torch.autograd.grad(sample_logit, p, retain_graph=True)[0].clone().detach().clamp(-1, 1)
                        info += (grad**2).sum().cpu()
                    infos.append(info)
                
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.total_flops += self.backward_flops
            
            self.update_schedule()
                    
            self.total_flops += self.forward_flops
            
        return loss, infos
    
    
    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        total_loss = 0.0
        infos = []
        if len(sample) > 0:
            self.memory.register_stream(sample)
        for i in range(iterations):
            self.model.train()
            data = self.memory.get_batch(batch_size, stream_batch_size)

            batch = {
                "img": data[1],         # images
                "cls": data[2],         # labels
                "reverse": data[3],     # reverse tensors
                "img_path": data[4],    # image paths
            }
            
            # print(f"[DEBUG] batch images: {batch['img_path']}")
            # print(f"[DEBUG] batch labels shape: {batch['cls'].shape}")

            self.optimizer.zero_grad()
            
            info = []
            loss, info = self.model_forward_samplewise(batch)
            infos.extend(info)
        
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
            stream_info = infos[:len(self.temp_batch)]
                # self.total_flops += (batch_size * (self.forward_flops + self.backward_flops))
                # print("self.total_flops", self.total_flops)
        return total_loss / iterations, stream_info
        
        
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
                train_loss, infos = self.online_train(self.temp_batch, self.batch_size, n_worker,
                                                      iterations=int(self.num_updates), stream_batch_size=self.temp_batchsize)
                self.report_training(sample_num, train_loss)
                for ind, stored_sample in enumerate(self.temp_batch):
                    self.update_memory(stored_sample, infos[ind])

                self.temp_batch = []
                self.num_updates -= int(self.num_updates)

    def update_memory(self, sample, info):
        self.reservoir_memory(sample, info)
    
    def reservoir_memory(self, sample, info):
        self.seen += 1
        if len(self.memory) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                self.memory.replace_sample(sample, j, info=info)#, mode=self.mode, online_iter=self.online_iter)
        else:
            self.memory.replace_sample(sample, info=info)#, mode=self.mode, online_iter=self.online_iter)