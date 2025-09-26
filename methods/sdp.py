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
from methods.baseline2_balanced import BASELINE2Balanced
from utils.data_loader import MemoryDataset, ClassBalancedDataset, FreqClsBalancedDataset

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class SDP(BASELINE2Balanced):
    def __init__(self, criterion, n_classes, device, **kwargs):
        super().__init__(criterion=criterion, n_classes=n_classes, device=device, **kwargs)
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
        self.det_loss_decay = 0.5

    def initialize_memory_buffer(self, memory_size):
        self.memory_size = memory_size - 32*3
        data_args = self.damo_cfg.get_data(self.damo_cfg.dataset.train_ann[0])
        self.memory = ClassBalancedDataset(ann_file=data_args['args']['ann_file'], root=data_args['args']['root'], transforms=None,class_names=self.damo_cfg.dataset.class_names,
            dataset=self.dataset, cls_list=self.exposed_classes, device=self.device, memory_size=self.memory_size, image_size=self.img_size, aug=self.damo_cfg.train.augment)
        
        self.new_exposed_classes = ['pretrained']

    def copy_model_head(self):
        self.sdp_model.head.gfl_cls[0] = copy.deepcopy(self.model.head.gfl_cls[0])
        self.sdp_model.head.gfl_cls[1] = copy.deepcopy(self.model.head.gfl_cls[1])
        self.sdp_model.head.gfl_cls[2] = copy.deepcopy(self.model.head.gfl_cls[2])
        
        self.ema_model_1.head.gfl_cls[0] = copy.deepcopy(self.model.head.gfl_cls[0])
        self.ema_model_1.head.gfl_cls[1] = copy.deepcopy(self.model.head.gfl_cls[1])
        self.ema_model_1.head.gfl_cls[2] = copy.deepcopy(self.model.head.gfl_cls[2])
        
        self.ema_model_2.head.gfl_cls[0] = copy.deepcopy(self.model.head.gfl_cls[0])
        self.ema_model_2.head.gfl_cls[1] = copy.deepcopy(self.model.head.gfl_cls[1])
        self.ema_model_2.head.gfl_cls[2] = copy.deepcopy(self.model.head.gfl_cls[2])
        
    def add_new_class(self, class_name):
        super().add_new_class(class_name)
        
        self.copy_model_head()

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
        # return

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
            self.update_sdp_model(num_updates=1.0)

            total_loss += loss.item()
            
        return total_loss / iterations

    def model_forward(self, batch):
        batch = self.preprocess_batch(batch)

        with torch.cuda.amp.autocast(enabled=False):
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
            print(f"Distill loss: {distill_loss.mean():.4f}")
            loss = (1 - sample_weight) * loss + (beta) * sample_weight * distill_loss.mean()
            
            # sample_weight = 0.1
            # beta = 1
            
            # loss = loss + (beta) * sample_weight * distill_loss.mean()
            
            self.total_flops += (len(batch["img"]) * self.forward_flops)

            return loss, loss_item

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
