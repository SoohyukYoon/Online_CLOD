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
from utils.data_loader import MemoryDataset

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class SDP(ER):
    def __init__(self, criterion, n_classes, device, **kwargs):
        super().__init__(criterion=criterion, n_classes=n_classes, device=device, **kwargs)
        self.memory_size = kwargs["memory_size"] - 32*3 # yolov9-s 모델 하나당 이미지 32장과 동일
        self.memory = OurDataset(self.args, self.dataset, self.exposed_classes, device=self.device, memory_size=self.memory_size, mosaic_prob=kwargs['mosaic_prob'],mixup_prob=kwargs['mixup_prob'])
        self.sdp_mean = 10000 #kwargs['sdp_mean']
        self.sdp_varcoeff = 0.75 #kwargs['sdp_var']
        assert 0.5 - 1 / self.sdp_mean < self.sdp_varcoeff < 1 - 1 / self.sdp_mean
        self.ema_ratio_1 = (1 - np.sqrt(2 * self.sdp_varcoeff - 1 + 2 / self.sdp_mean)) / (self.sdp_mean - 1 - self.sdp_mean * self.sdp_varcoeff)
        self.ema_ratio_2 = (1 + np.sqrt(2 * self.sdp_varcoeff - 1 + 2 / self.sdp_mean)) / (
                self.sdp_mean - 1 - self.sdp_mean * self.sdp_varcoeff)
        self.cur_time = None
        self.sdp_model = copy.deepcopy(self.model)
        self.ema_model_1 = copy.deepcopy(self.model)
        self.ema_model_2 = copy.deepcopy(self.model)
        self.sdp_updates = 0
        self.num_steps = 0
        self.reweight_ratio = 0.0
        self.det_loss_ema = 0.0
        self.det_loss_decay = 0.99   
        
        self.feature_layers = kwargs.get("feature_layers", [15,18,21])
        # put hook
        self.put_hook()
        self.new_exposed_classes = ['pretrained']

    def put_hook(self):
        self.features_per_layer = {}
        for layer in self.feature_layers:
            self.features_per_layer[layer] = []
        
        self.sdp_features_per_layer = {}
        for layer in self.feature_layers:
            self.sdp_features_per_layer[layer] = []
        self.hooks = []
        for layer in self.feature_layers:
            hook = self.model.model[layer].register_forward_hook(
                lambda m, x, y, layer=layer: self.features_per_layer[layer].append(y))
            hook2 = self.sdp_model.model[layer].register_forward_hook(
                lambda m, x, y, layer=layer: self.sdp_features_per_layer[layer].append(y))
            self.hooks.append(hook)
            self.hooks.append(hook2)

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
        else:
            self.memory.replace_sample(sample)

    def copy_model_head(self):
        self.sdp_model.model[22].heads[0].class_conv[2] = copy.deepcopy(self.model.model[22].heads[0].class_conv[2])
        self.sdp_model.model[22].heads[1].class_conv[2] = copy.deepcopy(self.model.model[22].heads[1].class_conv[2])
        self.sdp_model.model[22].heads[2].class_conv[2] = copy.deepcopy(self.model.model[22].heads[2].class_conv[2])
        self.sdp_model.model[30].heads[0].class_conv[2] = copy.deepcopy(self.model.model[30].heads[0].class_conv[2])
        self.sdp_model.model[30].heads[1].class_conv[2] = copy.deepcopy(self.model.model[30].heads[1].class_conv[2])
        self.sdp_model.model[30].heads[2].class_conv[2] = copy.deepcopy(self.model.model[30].heads[2].class_conv[2])
        
        self.ema_model_1.model[22].heads[0].class_conv[2] = copy.deepcopy(self.model.model[22].heads[0].class_conv[2])
        self.ema_model_1.model[22].heads[1].class_conv[2] = copy.deepcopy(self.model.model[22].heads[1].class_conv[2])
        self.ema_model_1.model[22].heads[2].class_conv[2] = copy.deepcopy(self.model.model[22].heads[2].class_conv[2])
        self.ema_model_1.model[30].heads[0].class_conv[2] = copy.deepcopy(self.model.model[30].heads[0].class_conv[2])
        self.ema_model_1.model[30].heads[1].class_conv[2] = copy.deepcopy(self.model.model[30].heads[1].class_conv[2])
        self.ema_model_1.model[30].heads[2].class_conv[2] = copy.deepcopy(self.model.model[30].heads[2].class_conv[2])
        
        self.ema_model_2.model[22].heads[0].class_conv[2] = copy.deepcopy(self.model.model[22].heads[0].class_conv[2])
        self.ema_model_2.model[22].heads[1].class_conv[2] = copy.deepcopy(self.model.model[22].heads[1].class_conv[2])
        self.ema_model_2.model[22].heads[2].class_conv[2] = copy.deepcopy(self.model.model[22].heads[2].class_conv[2])
        self.ema_model_2.model[30].heads[0].class_conv[2] = copy.deepcopy(self.model.model[30].heads[0].class_conv[2])
        self.ema_model_2.model[30].heads[1].class_conv[2] = copy.deepcopy(self.model.model[30].heads[1].class_conv[2])
        self.ema_model_2.model[30].heads[2].class_conv[2] = copy.deepcopy(self.model.model[30].heads[2].class_conv[2])

    def add_new_class(self, class_name):
        super().add_new_class(class_name)
        
        self.copy_model_head()
        
        args_copied = copy.deepcopy(self.args)
        args_copied.task.loss.aux = 0.0
        self.sdp_model.set_loss_function(args_copied, self.vec2box, self.num_learned_class)
        
        self.new_exposed_classes.append(class_name)
        
        self.memory.new_exposed_classes = self.new_exposed_classes

    def online_step(self, sample, sample_num, n_worker):
        if sample.get('klass',None) and sample['klass'] not in self.exposed_classes:
            self.online_after_task(sample_num)
            self.add_new_class(sample['klass'])
        elif sample.get('domain',None) and sample['domain'] not in self.exposed_domains:
            self.exposed_domains.append(sample['domain'])
            self.memory.new_exposed_classes.append(sample['domain'])
            self.memory.cls_count.append(0)
            self.memory.cls_idx.append([])
            
        
        self.sample_inference([sample])
        
        self.num_updates += self.online_iter
        if self.num_updates >= 1:
            train_loss = self.online_train([sample], self.batch_size, n_worker, iterations=int(self.num_updates), stream_batch_size=1)
            self.report_training(sample_num, train_loss)
            self.num_updates -= int(self.num_updates)
            self.update_schedule()
        
        self.update_memory(sample)
    
    @torch.no_grad()
    def sample_inference(self, sample):
        self.sdp_model.eval()
        stream_data = self.memory.get_stream_data(sample)
        batch = {
            "img": stream_data[1],         # images
            "cls": stream_data[2],         # labels
            "reverse": stream_data[3],     # reverse tensors
            "img_path": stream_data[4],    # image paths
        }
        batch = self.preprocess_batch(batch)
        outputs = self.sdp_model(batch["img"])
        aux_raw = outputs["AUX"]
        main_raw = outputs["Main"]

        aux_predicts = self.vec2box(aux_raw)
        main_predicts = self.vec2box(main_raw)
        cur_det_loss, _ = self.sdp_model.loss_fn(aux_predicts, main_predicts, batch['cls'])
        self.det_loss_ema = self.det_loss_decay * self.det_loss_ema + (1-self.det_loss_decay) * cur_det_loss.detach().cpu()
        conf = 1 - cur_det_loss / (self.det_loss_ema + 1e-8)
        conf = float(torch.clamp(torch.tensor(conf), 0.0, 1.0))
        target = 0.5 * conf
        self.reweight_ratio += (target - self.reweight_ratio) * (1 - self.ema_ratio_1)
        print(f"Reweight ratio: {self.reweight_ratio:.4f}, Det loss: {cur_det_loss:.4f}, Det loss EMA: {self.det_loss_ema:.4f}, Confidence: {conf:.4f}")
        
    def update_schedule(self, reset=False):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr * (1 - self.reweight_ratio)

    @torch.no_grad()
    def update_sdp_model(self, num_updates=1.0):
        ema_inv_ratio_1 = (1 - self.ema_ratio_1) ** num_updates
        ema_inv_ratio_2 = (1 - self.ema_ratio_2) ** num_updates
        model_params = OrderedDict(self.model.named_parameters())
        ema_params = OrderedDict(self.sdp_model.named_parameters())
        ema_params_1 = OrderedDict(self.ema_model_1.named_parameters())
        ema_params_2 = OrderedDict(self.ema_model_2.named_parameters())
        assert model_params.keys() == ema_params.keys()
        assert model_params.keys() == ema_params_1.keys()
        assert model_params.keys() == ema_params_2.keys()
        self.sdp_updates += 1
        for name, param in model_params.items():
            ema_params_1[name].sub_((1. - ema_inv_ratio_1) * (ema_params_1[name] - param))
            ema_params_2[name].sub_((1. - ema_inv_ratio_2) * (ema_params_2[name] - param))
            ema_params[name].copy_(
                self.ema_ratio_2 / (self.ema_ratio_2 - self.ema_ratio_1) * ema_params_1[name] - self.ema_ratio_1 / (self.ema_ratio_2 - self.ema_ratio_1) * ema_params_2[
                    name])
            # + ((1. - self.ema_ratio_2)*self.ema_ratio_1**self.ema_updates - (1. - self.ema_ratio_1)*self.ema_ratio_2**self.ema_updates) / (self.ema_ratio_1 - self.ema_ratio_2) * param)
        self.copy_model_head()

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.sdp_model.named_buffers())

        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            shadow_buffers[name].copy_(buffer)
    
    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        total_loss = 0.0
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

            self.optimizer.zero_grad()

            loss, loss_item = self.model_forward(batch)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.total_flops += (len(data[1]) * self.backward_flops)
            
            self.update_schedule()
            self.update_sdp_model(num_updates=1.0)

            total_loss += loss.item()
            
        return total_loss / iterations

    def model_forward(self, batch):
        batch = self.preprocess_batch(batch)

        for layer in self.feature_layers:
            self.features_per_layer[layer].clear()
            self.sdp_features_per_layer[layer].clear()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # 모델 실행: output = {"AUX": ..., "Main": ...}
            outputs = self.model(batch["img"])
            aux_raw = outputs["AUX"]
            main_raw = outputs["Main"]

            # Vec2Box 변환: [B, A, C], [B, A, R], [B, A, 4]
            aux_predicts = self.vec2box(aux_raw)
            main_predicts = self.vec2box(main_raw)
            loss, loss_item = self.model.loss_fn(aux_predicts, main_predicts, batch['cls'])
            
            distill_loss = 0
            
            with torch.no_grad():
                _, _ = self.sdp_model(batch['img'])
            
            for layer in self.feature_layers:
                feature = self.features_per_layer[layer][0]
                sdp_feature = self.sdp_features_per_layer[layer][0]
                distill_loss += ((feature - sdp_feature.detach())**2).mean(dim=-1).mean(dim=-1).sum(dim=1) / len(self.feature_layers)
            sample_weight = self.reweight_ratio
            grad = self.get_grad(loss, 
                                 [p for p in self.model.model[22].heads[0].class_conv[2].parameters()]
                                 +[p for p in self.model.model[22].heads[1].class_conv[2].parameters()]
                                 +[p for p in self.model.model[22].heads[2].class_conv[2].parameters()])
            
            beta = torch.sqrt((grad.detach() ** 2).mean() / (distill_loss.detach() * 4 + 1e-8)).mean()
            print(f"Beta: {beta:.4f}, Grad norm: {torch.norm(grad):.4f}, Distill loss: {distill_loss.mean():.4f}")
            loss = (1 - sample_weight) * loss + beta * sample_weight * distill_loss.mean()
            
            self.total_flops += (len(batch["img"]) * self.forward_flops)

            return loss, loss_item

    def online_evaluate(self, sample_num, data_time):
        for hook in self.hooks:
            hook.remove()
        
        eval_dict = super().online_evaluate(sample_num, data_time)
        
        self.put_hook()  # Re-register hooks after evaluation
        
        return eval_dict
    
    @torch.no_grad()
    def get_grad(self, loss: torch.Tensor, param_list):
        # autograd will return *None* when the param is unused
        grads = torch.autograd.grad(
            loss,
            param_list,
            retain_graph=True,
            allow_unused=True,
        )

        flat = []
        for g, p in zip(grads, param_list):
            flat.append(
                (torch.zeros_like(p) if g is None else g)
                .contiguous()
                .view(-1)
            )
        grad_vec = torch.cat(flat)
        torch.cuda.empty_cache()
        return grad_vec

# customized MemoryDataset
import os
from utils.data_loader import MemoryDataset, get_pretrained_statistics, get_statistics, mean, AugmentationComposer
from yolo.tools.data_augmentation import *
import glob
class OurDataset(MemoryDataset):
    def __init__(self, args, dataset, cls_list=None, device=None, data_dir=None, memory_size=None, 
                 init_buffer_size=None, mosaic_prob=1.0, mixup_prob=0.0):
        self.args = args
        self.image_sizes = args.image_size  # [640, 640]
        self.memory_size = memory_size

        self.buffer = []
        self.stream_data = []
        self.logits = []

        self.dataset = dataset
        self.device = device
        self.data_dir = data_dir

        self.counts = []
        self.class_usage_cnt = []
        self.tasks = []
        
        # FIXME: fix for object detection class counting
        self.cls_list = cls_list if cls_list else []
        self.cls_used_times = []
        self.cls_dict = {}
        self.cls_count = [0]
        self.cls_idx = [[]]
        
        self.new_exposed_classes = ['pretrained']
        self.cls_train_cnt = np.array([])
        self.score = []
        self.others_loss_decrease = np.array([])
        self.previous_idx = np.array([], dtype=int)
        self.usage_cnt = []
        self.sample_weight = []
        self.data = {}
        
        self.build_initial_buffer(init_buffer_size)

        n_classes, image_dir, label_path = get_statistics(dataset=self.dataset)
        self.image_dir = image_dir
        self.label_path = label_path

        self.augment = True

        transforms = {
            "Mosaic": mosaic_prob,
            "MixUp": mixup_prob,
            "HorizontalFlip": 0.5,
            "RandomCrop": 1,
            "RemoveOutliers": 1e-8,
        }
        transforms = [eval(k)(v) for k, v in transforms.items()]
        self.base_size = mean(self.image_sizes)
        self.transform = AugmentationComposer(transforms, self.image_sizes, self.base_size)
        self.transform.get_more_data = self.get_more_data
        
        self.padResize = AugmentationComposer([], self.image_sizes, self.base_size)
        
        # Load all metadata once during initialization
        self.metadata_cache = {}
        self._load_all_metadata()
        
    
    def build_initial_buffer(self, buffer_size=None):
        n_classes, images_dir, label_path = get_pretrained_statistics(self.dataset)
        self.image_dir = images_dir
        self.label_path = label_path
        self.metadata_cache = {}
        self._load_all_metadata()
        if self.dataset == 'VOC_10_10':
            image_files = glob.glob(os.path.join(images_dir, "train2012", "*.jpg")) \
                        + glob.glob(os.path.join(images_dir, "train2007", "*.jpg")) \
                        + glob.glob(os.path.join(images_dir, "val2012", "*.jpg")) \
                        + glob.glob(os.path.join(images_dir, "val2007", "*.jpg"))
        else:
            image_files = glob.glob(os.path.join(images_dir, "train","*.jpg"))

        indices = np.random.choice(range(len(image_files)), size=buffer_size or self.memory_size, replace=False)

        for idx in indices:
            image_path = image_files[idx]
            split_name = image_path.split('/')[-2]
            base_name = image_path.split('/')[-1]
            self.replace_sample({'file_name': split_name + '/' + base_name, 'label': None}, images_dir=images_dir,label_path=label_path)
    
    def add_new_class(self, cls_list, sample=None):
        self.cls_list = cls_list
        self.cls_count.append(0)
        self.cls_idx.append([])
    
    def replace_sample(self, sample, idx=None, images_dir=None, label_path=None):
        img, labels, image_path, ratio = self.load_data(sample['file_name'], image_dir=images_dir or self.image_dir, label_path=label_path or self.label_path, cls_type = sample.get('klass', None))
        data = (img, labels, image_path, ratio)
        if sample.get('klass', None):
            self.cls_count[self.new_exposed_classes.index(sample['klass'])] += 1
            sample_category = sample['klass']
        elif sample.get('domain', None):
            self.cls_count[self.new_exposed_classes.index(sample['domain'])] += 1
            sample_category = sample['domain']
        else:
            self.cls_count[self.new_exposed_classes.index('pretrained')] += 1
            sample_category = 'pretrained'
        # self.cls_count[self.new_exposed_classes.index(sample.get('klass', 'pretrained'))] += 1
        if idx is None:
            self.cls_idx[self.new_exposed_classes.index(sample_category)].append(len(self.buffer))
        self.buffer.append(data)