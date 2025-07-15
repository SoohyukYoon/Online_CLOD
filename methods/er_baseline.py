# When we make a new one, we should inherit the Finetune class.
import logging
import copy
import time
import datetime
import pickle
import numpy as np
import pandas as pd
import os
import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from scipy.stats import chi2, norm
#from ptflops import get_model_complexity_info
from flops_counter.ptflops import get_model_complexity_info
# from utils.cka import linear_CKA
from utils.data_loader import ImageDataset, MemoryDataset, get_exposed_classes
from utils.train_utils import select_model, select_optimizer, select_scheduler, MeanAveragePrecisionCustomized
#### object detection
import random

from utils.block_utils import get_blockwise_flops, MODEL_BLOCK_DICT

from yolo.tools.data_loader import create_dataloader
from yolo.utils.bounding_box_utils import create_converter, to_metrics_format
from yolo.utils.model_utils import PostProcess
from hydra import compose, initialize
from yolo.config.config import Config

from tqdm import tqdm

logger = logging.getLogger()


class ER:
    def __init__(self, criterion, n_classes, device, **kwargs):
        # 기존 파라미터 저장
        self.n_classes = n_classes
        self.device = device
        self.seen = 0
        self.note = kwargs["note"]
        self.rnd_seed = kwargs["rnd_seed"]
        self.dataset = kwargs["dataset"]
        self.mode = kwargs["mode"]
        self.lr = kwargs["lr"]
        self.batch_size = kwargs["batchsize"]
        self.temp_batchsize = kwargs.get("temp_batchsize") or (self.batch_size // 2)
        self.memory_size = kwargs["memory_size"] - self.temp_batchsize
        self.topk = kwargs["topk"]
        self.sigma = kwargs["sigma"]
        self.repeat = kwargs["repeat"]
        self.weight_option = kwargs["weight_option"]
        self.weight_method = kwargs["weight_method"]
        self.model_name = kwargs["model_name"]
        self.opt_name = kwargs["opt_name"]
        self.sched_name = 'const' if kwargs["sched_name"] == "default" else kwargs["sched_name"]
        self.data_dir = kwargs["data_dir"]
        self.online_iter = kwargs["online_iter"]
        self.gpu_transform = kwargs["gpu_transform"]
        self.use_kornia = kwargs["use_kornia"]
        self.use_amp = kwargs["use_amp"]
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.exposed_classes = get_exposed_classes(self.dataset)
        self.num_learned_class = len(self.exposed_classes)
        self.num_learning_class = self.num_learned_class + 1
        
        if 'VOC' in self.dataset:
            data_name = 'voc'
        elif 'BDD' in self.dataset:
            data_name = 'bdd100k'
        elif 'SHIFT' in self.dataset:
            data_name = 'shift'
        elif 'MILITARY_SYNTHETIC' in self.dataset:
            data_name = 'military_synthetic'
        with initialize(config_path="../yolo/config", version_base=None):
            self.args: Config = compose(config_name="config", overrides=["model=v9-s",f"dataset={data_name}"])
        self.exposed_domains = [f'{data_name}_source']
        self.model = select_model(self.args,self.dataset)

        self.model.model.args = self.args.model
        self.stride = max(int(self.model.model.stride.max() if hasattr(self.model.model, "stride") else 32), 32)
        self.optimizer = select_optimizer(self.opt_name, self.model, lr=self.lr)
        self.scheduler = None  # optional

        self.vec2box = create_converter(
            self.args.model.name, self.model, self.args.model.anchor, self.args.image_size, self.device
        )
        self.model.set_loss_function(self.args, self.vec2box, self.num_learned_class)
        self.memory = MemoryDataset(self.args, self.dataset, self.exposed_classes, device=self.device, memory_size=self.memory_size, mosaic_prob=kwargs['mosaic_prob'],mixup_prob=kwargs['mixup_prob'])
        self.temp_batch = []
        self.num_updates = 0
        self.train_count = 0

        self.gt_label = None
        self.test_records = []
        self.n_model_cls = []
        self.forgetting = []
        self.knowledge_gain = []
        self.total_knowledge = []
        self.retained_knowledge = []
        self.forgetting_time = []

        self.f_calculated = False
        self.total_flops = 0.0
        self.f_period = kwargs['f_period']
        self.f_next_time = 0
        self.start_time = time.time()

        self.writer = SummaryWriter(f'tensorboard/{self.dataset}/{self.note}/seed_{self.rnd_seed}')
        self.save_path = f'results/{self.dataset}/{self.note}/seed_{self.rnd_seed}'
        
        # test arguments
        self.metric = MeanAveragePrecisionCustomized(iou_type="bbox", box_format="xyxy",class_metrics=True)#, backend="faster_coco_eval")
        self.metric.warn_on_many_detections = False
        
        self.val_loader = create_dataloader(self.args.task.validation.data, self.args.dataset, self.args.task.validation.task)
        self.post_process = PostProcess(self.vec2box, self.args.task.validation.nms)
        
        self.model = self.model.to(device)
        
        
        self.block_names = MODEL_BLOCK_DICT[self.model_name]
        self.num_blocks = len(self.block_names) - 1
        self.get_flops_parameter()
        
    def get_total_flops(self):
        return self.total_flops
        
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


    def save_std_pickle(self):
        '''
        class_std, sample_std = self.memory.get_std()
        self.class_std_list.append(class_std)
        self.sample_std_list.append(sample_std)
        '''
        
        cls_file_name = self.mode + '_final_cls_std.pickle'
        sample_file_name = self.mode + '_sample_std.pickle'
        
        with open(cls_file_name, 'wb') as handle:
            pickle.dump(self.class_std_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        '''
        with open(sample_file_name, 'wb') as handle:
            pickle.dump(self.sample_std_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        '''

    def add_new_class(self, class_name):
        self.exposed_classes.append(class_name)
        print('self.exposed_classes', self.exposed_classes)
        self.num_learned_class = len(self.exposed_classes)
        # classification head for yolov9-s
        prev_weights = [
            copy.deepcopy(self.model.model[22].heads[0].class_conv[2].weight),
            copy.deepcopy(self.model.model[22].heads[1].class_conv[2].weight),
            copy.deepcopy(self.model.model[22].heads[2].class_conv[2].weight),
            copy.deepcopy(self.model.model[30].heads[0].class_conv[2].weight),
            copy.deepcopy(self.model.model[30].heads[1].class_conv[2].weight),
            copy.deepcopy(self.model.model[30].heads[2].class_conv[2].weight),
        ]
        prev_biases = [
            copy.deepcopy(self.model.model[22].heads[0].class_conv[2].bias),
            copy.deepcopy(self.model.model[22].heads[1].class_conv[2].bias),
            copy.deepcopy(self.model.model[22].heads[2].class_conv[2].bias),
            copy.deepcopy(self.model.model[30].heads[0].class_conv[2].bias),
            copy.deepcopy(self.model.model[30].heads[1].class_conv[2].bias),
            copy.deepcopy(self.model.model[30].heads[2].class_conv[2].bias),
        ]
        self.model.model[22].heads[0].class_conv[2] = nn.Conv2d(self.model.model[22].heads[0].class_conv[2].in_channels, self.num_learned_class, self.model.model[22].heads[0].class_conv[2].kernel_size, self.model.model[22].heads[0].class_conv[2].stride).to(self.device)
        self.model.model[22].heads[1].class_conv[2] = nn.Conv2d(self.model.model[22].heads[1].class_conv[2].in_channels, self.num_learned_class, self.model.model[22].heads[1].class_conv[2].kernel_size, self.model.model[22].heads[1].class_conv[2].stride).to(self.device)
        self.model.model[22].heads[2].class_conv[2] = nn.Conv2d(self.model.model[22].heads[2].class_conv[2].in_channels, self.num_learned_class, self.model.model[22].heads[2].class_conv[2].kernel_size, self.model.model[22].heads[2].class_conv[2].stride).to(self.device)
        
        self.model.model[30].heads[0].class_conv[2] = nn.Conv2d(self.model.model[30].heads[0].class_conv[2].in_channels, self.num_learned_class, self.model.model[30].heads[0].class_conv[2].kernel_size, self.model.model[30].heads[0].class_conv[2].stride).to(self.device)
        self.model.model[30].heads[1].class_conv[2] = nn.Conv2d(self.model.model[30].heads[1].class_conv[2].in_channels, self.num_learned_class, self.model.model[30].heads[1].class_conv[2].kernel_size, self.model.model[30].heads[1].class_conv[2].stride).to(self.device)
        self.model.model[30].heads[2].class_conv[2] = nn.Conv2d(self.model.model[30].heads[2].class_conv[2].in_channels, self.num_learned_class, self.model.model[30].heads[2].class_conv[2].kernel_size, self.model.model[30].heads[2].class_conv[2].stride).to(self.device)
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.model[22].heads[0].class_conv[2].weight[:self.num_learned_class - 1] = prev_weights[0]
                self.model.model[22].heads[0].class_conv[2].bias[:self.num_learned_class - 1] = prev_biases[0]
                self.model.model[22].heads[1].class_conv[2].weight[:self.num_learned_class - 1] = prev_weights[1]
                self.model.model[22].heads[1].class_conv[2].bias[:self.num_learned_class - 1] = prev_biases[1]
                self.model.model[22].heads[2].class_conv[2].weight[:self.num_learned_class - 1] = prev_weights[2]
                self.model.model[22].heads[2].class_conv[2].bias[:self.num_learned_class - 1] = prev_biases[2]
                
                self.model.model[30].heads[0].class_conv[2].weight[:self.num_learned_class - 1] = prev_weights[3]
                self.model.model[30].heads[0].class_conv[2].bias[:self.num_learned_class - 1] = prev_biases[3]
                self.model.model[30].heads[1].class_conv[2].weight[:self.num_learned_class - 1] = prev_weights[4]
                self.model.model[30].heads[1].class_conv[2].bias[:self.num_learned_class - 1] = prev_biases[4]
                self.model.model[30].heads[2].class_conv[2].weight[:self.num_learned_class - 1] = prev_weights[5]
                self.model.model[30].heads[2].class_conv[2].bias[:self.num_learned_class - 1] = prev_biases[5]
                
                self.model.model[22].heads[0].class_conv[2].weight[self.num_learned_class - 1] = prev_weights[0].mean(dim=0)
                self.model.model[22].heads[0].class_conv[2].bias[self.num_learned_class - 1] = prev_biases[0].mean(dim=0)
                self.model.model[22].heads[1].class_conv[2].weight[self.num_learned_class - 1] = prev_weights[1].mean(dim=0)
                self.model.model[22].heads[1].class_conv[2].bias[self.num_learned_class - 1] = prev_biases[1].mean(dim=0)
                self.model.model[22].heads[2].class_conv[2].weight[self.num_learned_class - 1] = prev_weights[2].mean(dim=0)
                self.model.model[22].heads[2].class_conv[2].bias[self.num_learned_class - 1] = prev_biases[2].mean(dim=0)
                self.model.model[30].heads[0].class_conv[2].weight[self.num_learned_class - 1] = prev_weights[3].mean(dim=0)
                self.model.model[30].heads[0].class_conv[2].bias[self.num_learned_class - 1] = prev_biases[3].mean(dim=0)
                self.model.model[30].heads[1].class_conv[2].weight[self.num_learned_class - 1] = prev_weights[4].mean(dim=0)
                self.model.model[30].heads[1].class_conv[2].bias[self.num_learned_class - 1] = prev_biases[4].mean(dim=0)
                self.model.model[30].heads[2].class_conv[2].weight[self.num_learned_class - 1] = prev_weights[5].mean(dim=0)
                self.model.model[30].heads[2].class_conv[2].bias[self.num_learned_class - 1] = prev_biases[5].mean(dim=0)
                
        
        for param in self.optimizer.param_groups[3]['params']:
            if param in self.optimizer.state.keys():
                del self.optimizer.state[param]
        for param in self.optimizer.param_groups[4]['params']:
            if param in self.optimizer.state.keys():
                del self.optimizer.state[param]
        
        del self.optimizer.param_groups[4], self.optimizer.param_groups[3]
        self.optimizer.add_param_group({'params': [
                                        self.model.model[22].heads[0].class_conv[2].weight,
                                        self.model.model[22].heads[1].class_conv[2].weight,
                                        self.model.model[22].heads[2].class_conv[2].weight,
                                        self.model.model[30].heads[0].class_conv[2].weight,
                                        self.model.model[30].heads[1].class_conv[2].weight,
                                        self.model.model[30].heads[2].class_conv[2].weight,
                                        ], "momentum": 0.937,})
        self.optimizer.add_param_group({'params': [
                                        self.model.model[22].heads[0].class_conv[2].bias,
                                        self.model.model[22].heads[1].class_conv[2].bias,
                                        self.model.model[22].heads[2].class_conv[2].bias,
                                        self.model.model[30].heads[0].class_conv[2].bias,
                                        self.model.model[30].heads[1].class_conv[2].bias,
                                        self.model.model[30].heads[2].class_conv[2].bias,
                                        ], "momentum": 0.937, "weight_decay": 0})
        self.memory.add_new_class(cls_list=self.exposed_classes)
        # if 'reset' in self.sched_name:
        #     self.update_schedule(reset=True)
        self.model.set_loss_function(self.args, self.vec2box, self.num_learned_class)

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
            
            # print(f"[DEBUG] batch images: {batch['img_path']}")
            # print(f"[DEBUG] batch labels shape: {batch['cls'].shape}")

            self.optimizer.zero_grad()

            loss, loss_item = self.model_forward(batch)
            # print(f"[DEBUG] individual losses: {loss_item}")

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.total_flops += (len(data[1]) * self.backward_flops)
            
            self.update_schedule()

            total_loss += loss.item()
            
            # self.total_flops += (batch_size * (self.forward_flops + self.backward_flops))
            # print("self.total_flops", self.total_flops)
        return total_loss / iterations
    
    def model_forward(self, batch):
        batch = self.preprocess_batch(batch)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # 모델 실행: output = {"AUX": ..., "Main": ...}
            outputs = self.model(batch["img"])
            aux_raw = outputs["AUX"]
            main_raw = outputs["Main"]

            # Vec2Box 변환: [B, A, C], [B, A, R], [B, A, 4]
            aux_predicts = self.vec2box(aux_raw)
            main_predicts = self.vec2box(main_raw)

            # targets: [B, T, 5] (cls, cx, cy, w, h) → xyxy
            # targets = batch["cls"].clone()
            # x, y, w, h = targets[..., 1:].unbind(-1)
            # targets[..., 1] = x
            # targets[..., 2] = y
            # targets[..., 3] = x + w
            # targets[..., 4] = y + h

            # 손실 계산
            # loss, loss_items = self.model.loss_fn(aux_predicts, main_predicts, targets)
            loss, loss_item = self.model.loss_fn(aux_predicts, main_predicts, batch['cls'])
            
            self.total_flops += (len(batch["img"]) * self.forward_flops)
        return loss, loss_item


    
    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)#.float()# / 255
        batch["cls"] = batch["cls"].to(self.device)

        # if self.args.get('multi_scale', False):
        #     imgs = batch["img"]
        #     imgsz = self.args['image_size'][0]
        #     sz = (random.randrange(int(imgsz * 0.5), int(imgsz * 1.5 + self.stride)) // self.stride) * self.stride
        #     sf = sz / max(imgs.shape[2:])
        #     if sf != 1:
        #         ns = [math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]]
        #         imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
        #     batch["img"] = imgs
        return batch



    def report_training(self, sample_num, train_loss, train_initial_cka=None, train_group1_cka=None, train_group2_cka=None, train_group3_cka=None, train_group4_cka=None):
        self.writer.add_scalar(f"train/loss", train_loss, sample_num)

        if train_initial_cka is not None:
            '''
            self.writer.add_scalar(f"train/initial_cka", train_initial_cka, sample_num)
            self.writer.add_scalar(f"train/group1_cka", train_group1_cka, sample_num)
            self.writer.add_scalar(f"train/group2_cka", train_group2_cka, sample_num)
            self.writer.add_scalar(f"train/group3_cka", train_group3_cka, sample_num)
            self.writer.add_scalar(f"train/group4_cka", train_group4_cka, sample_num)
            '''
            cka_dict = {}
            cka_dict["initial_cka"] = train_initial_cka
            cka_dict["train_group1_cka"] = train_group1_cka
            cka_dict["train_group2_cka"] = train_group2_cka
            cka_dict["train_group3_cka"] = train_group3_cka
            cka_dict["train_group4_cka"] = train_group4_cka

            self.writer.add_scalars(f"train/cka", cka_dict, sample_num)
            #self.writer.add_scalar(f"train/fc_cka", train_fc_cka, sample_num)

        logger.info(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | "
            f"TFLOPs {self.total_flops/1000:.2f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))} | "
        )

    def report_test(self, sample_num, avg_acc, classwise_acc):
        # self.writer.add_scalar(f"test/loss", avg_loss, sample_num)
        self.writer.add_scalar(f"test/acc", avg_acc, sample_num)
        # self.writer.add_scalar(f"test/classwise_acc", classwise_acc, sample_num)
        logger.info(
            f"Test | Sample # {sample_num} | test_acc {avg_acc:.4f} | \n"
            f"classwise mAP50 [{'|'.join([str(round(cls_acc,3)) for cls_acc in classwise_acc])}] | \n"
            f"TFLOPs {self.total_flops/1000:.2f} | "
        )

    def update_memory(self, sample):
        self.reservoir_memory(sample)

    def update_schedule(self, reset=False):
        pass

    def evaluate(self):
        for i, batch in enumerate(tqdm(self.val_loader)):
            batch_size, images, targets, rev_tensor, img_paths = batch
            H, W = images.shape[2:]
            images, targets = images.to(self.device), targets.to(self.device)
            predicts = self.post_process(self.model(images), image_size=[W, H])
            mAP = self.metric(
                [to_metrics_format(predict) for predict in predicts], [to_metrics_format(target) for target in targets]
            )
        
        epoch_metrics = self.metric.compute()
        del epoch_metrics["classes"]
        eval_dict = {
            "avg_mAP50": sum(epoch_metrics['map50_per_class'].tolist()[:self.num_learned_class])/self.num_learned_class,
            "classwise_mAP50": epoch_metrics['map50_per_class'].tolist()[:self.num_learned_class]
        }
        self.metric.reset()
        
        return eval_dict

    def online_evaluate(self, sample_num, data_time):
        torch.cuda.empty_cache()
        self.model.eval()
        print("evaluate")
        
        if self.dataset=='BDD_domain':
            eval_dict = {"avg_mAP50":0, "classwise_mAP50":[]}
            for data_name in self.exposed_domains:#['bdd100k_source', 'bdd100k_cloudy', 'bdd100k_rainy', 'bdd100k_dawndusk', 'bdd100k_night']:
                with initialize(config_path="../yolo/config", version_base=None):
                    self.args: Config = compose(config_name="config", overrides=["model=v9-s",f"dataset={data_name}"])            
                self.val_loader = create_dataloader(self.args.task.validation.data, self.args.dataset, self.args.task.validation.task)
                eval_dict_sub = self.evaluate()
                clean = [v for v in eval_dict_sub['classwise_mAP50'] if v != -1]
                average = sum(clean) / len(clean)
                eval_dict['avg_mAP50'] += average/len(self.exposed_domains) #/5
                eval_dict["classwise_mAP50"].append(average)
            with initialize(config_path="../yolo/config", version_base=None):
                self.args: Config = compose(config_name="config", overrides=["model=v9-s",f"dataset=bdd100k"])
        elif self.dataset=='SHIFT_domain':
            eval_dict = {"avg_mAP50":0, "classwise_mAP50":[]}
            for data_name in self.exposed_domains:#['shift_source', 'shift_overcast', 'shift_cloudy', 'shift_rainy', 'shift_foggy', 'shift_dawndusk', 'shift_night']:
                with initialize(config_path="../yolo/config", version_base=None):
                    self.args: Config = compose(config_name="config", overrides=["model=v9-s",f"dataset={data_name}"])            
                self.val_loader = create_dataloader(self.args.task.validation.data, self.args.dataset, self.args.task.validation.task)
                eval_dict_sub = self.evaluate()
                clean = [v for v in eval_dict_sub['classwise_mAP50'] if v != -1]
                average = sum(clean) / len(clean)
                eval_dict['avg_mAP50'] += average/len(self.exposed_domains)
                eval_dict["classwise_mAP50"].append(average)
            with initialize(config_path="../yolo/config", version_base=None):
                self.args: Config = compose(config_name="config", overrides=["model=v9-s",f"dataset=shift"])
        elif self.dataset=='MILITARY_SYNTHETIC_domain_1' or self.dataset=='MILITARY_SYNTHETIC_domain_2' or self.dataset=='MILITARY_SYNTHETIC_domain_3':
            eval_dict = {"avg_mAP50":0, "classwise_mAP50":[]}
            for data_name in ['military_synthetic_domain_source', 'military_synthetic_domain_night', 'military_synthetic_domain_winter', 'military_synthetic_domain_infrared']:
                with initialize(config_path="../yolo/config", version_base=None):
                    self.args: Config = compose(config_name="config", overrides=["model=v9-s",f"dataset={data_name}"])            
                self.val_loader = create_dataloader(self.args.task.validation.data, self.args.dataset, self.args.task.validation.task)
                eval_dict_sub = self.evaluate()
                clean = [v for v in eval_dict_sub['classwise_mAP50'] if v != -1]
                average = sum(clean) / len(clean)
                eval_dict['avg_mAP50'] += average/4 #len(self.exposed_domains)
                eval_dict["classwise_mAP50"].append(average)
            with initialize(config_path="../yolo/config", version_base=None):
                self.args: Config = compose(config_name="config", overrides=["model=v9-s",f"dataset=military_synthetic"])
        else:
            eval_dict = self.evaluate()

        self.report_test(sample_num, eval_dict["avg_mAP50"], eval_dict["classwise_mAP50"])
        
        return eval_dict

    def get_forgetting(self, sample_num, test_list, cls_dict, batch_size, n_worker):
        test_df = pd.DataFrame(test_list)
        test_dataset = ImageDataset(
            test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=list(cls_dict.keys()),
            data_dir=self.data_dir
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=n_worker,
        )

        preds = []
        gts = []
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                logit = self.model(x)
                pred = torch.argmax(logit, dim=-1)
                preds.append(pred.detach().cpu().numpy())
                gts.append(y.detach().cpu().numpy())
        preds = np.concatenate(preds)
        if self.gt_label is None:
            gts = np.concatenate(gts)
            self.gt_label = gts
        self.test_records.append(preds)
        self.n_model_cls.append(copy.deepcopy(self.num_learned_class))
        if len(self.test_records) > 1:
            forgetting, knowledge_gain, total_knowledge, retained_knowledge = self.calculate_online_forgetting(self.n_classes, self.gt_label, self.test_records[-2], self.test_records[-1], self.n_model_cls[-2], self.n_model_cls[-1])
            self.forgetting.append(forgetting)
            self.knowledge_gain.append(knowledge_gain)
            self.total_knowledge.append(total_knowledge)
            self.retained_knowledge.append(retained_knowledge)
            self.forgetting_time.append(sample_num)
            logger.info(f'Forgetting {forgetting} | Knowledge Gain {knowledge_gain} | Total Knowledge {total_knowledge} | Retained Knowledge {retained_knowledge}')
            np.save(self.save_path + '_forgetting.npy', self.forgetting)
            np.save(self.save_path + '_knowledge_gain.npy', self.knowledge_gain)
            np.save(self.save_path + '_total_knowledge.npy', self.total_knowledge)
            np.save(self.save_path + '_retained_knowledge.npy', self.retained_knowledge)
            np.save(self.save_path + '_forgetting_time.npy', self.forgetting_time)
        else:
            print("else")

    def online_before_task(self, cur_iter):
        # Task-Free
        pass

    def online_after_task(self, cur_iter):
        # Task-Free
        pass

    def reservoir_memory(self, sample):
        self.seen += 1
        if len(self.memory) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                self.memory.replace_sample(sample, j)#, mode=self.mode, online_iter=self.online_iter)
        else:
            self.memory.replace_sample(sample)#, mode=self.mode, online_iter=self.online_iter)

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = None #select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

    def _interpret_pred(self, y, pred):
        # xlable is batch
        ret_num_data = torch.zeros(self.n_classes)
        ret_corrects = torch.zeros(self.n_classes)

        xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

        correct_xlabel = y.masked_select(y == pred)
        correct_cls, correct_cnt = correct_xlabel.unique(return_counts=True)
        for cls_idx, cnt in zip(correct_cls, correct_cnt):
            ret_corrects[cls_idx] = cnt

        return ret_num_data, ret_corrects

    def calculate_online_acc(self, cls_acc, data_time, cls_dict, cls_addition):
        mean = (np.arange(self.n_classes*self.repeat)/self.n_classes).reshape(-1, self.n_classes)
        cls_weight = np.exp(-0.5*((data_time-mean)/(self.sigma/100))**2)/(self.sigma/100*np.sqrt(2*np.pi))
        cls_addition = np.array(cls_addition).astype(np.int)
        for i in range(self.n_classes):
            cls_weight[:cls_addition[i], i] = 0
        cls_weight = cls_weight.sum(axis=0)
        cls_order = [cls_dict[cls] for cls in self.exposed_classes]
        for i in range(self.n_classes):
            if i not in cls_order:
                cls_order.append(i)
        cls_weight = cls_weight[cls_order]/np.sum(cls_weight)
        online_acc = np.sum(np.array(cls_acc)*cls_weight)
        return online_acc

    def calculate_online_forgetting(self, n_classes, y_gt, y_t1, y_t2, n_cls_t1, n_cls_t2, significance=0.99):
        cnt = {}
        total_cnt = len(y_gt)
        uniform_cnt = len(y_gt)
        cnt_gt = np.zeros(n_classes)
        cnt_y1 = np.zeros(n_cls_t1)
        cnt_y2 = np.zeros(n_cls_t2)
        num_relevant = 0
        for i, gt in enumerate(y_gt):
            y1, y2 = y_t1[i], y_t2[i]
            cnt_gt[gt] += 1
            cnt_y1[y1] += 1
            cnt_y2[y2] += 1
            if (gt, y1, y2) in cnt.keys():
                cnt[(gt, y1, y2)] += 1
            else:
                cnt[(gt, y1, y2)] = 1
        cnt_list = list(sorted(cnt.items(), key=lambda item: item[1], reverse=True))
        for i, item in enumerate(cnt_list):
            chi2_value = total_cnt
            for j, item_ in enumerate(cnt_list[i + 1:]):
                expect = total_cnt / (n_classes * n_cls_t1 * n_cls_t2 - i)
                chi2_value += (item_[1] - expect) ** 2 / expect - expect
            if chi2.cdf(chi2_value, n_classes ** 3 - 2 - i) < significance:
                break
            uniform_cnt -= item[1]
            num_relevant += 1
        probs = uniform_cnt * np.ones([n_classes, n_cls_t1, n_cls_t2]) / ((n_classes * n_cls_t1 * n_cls_t2 - num_relevant) * total_cnt)
        for j in range(num_relevant):
            gt, y1, y2 = cnt_list[j][0]
            probs[gt][y1][y2] = cnt_list[j][1] / total_cnt
        forgetting = np.sum(probs*np.log(np.sum(probs, axis=(0, 1), keepdims=True) * probs / (np.sum(probs, axis=0, keepdims=True)+1e-10) / (np.sum(probs, axis=1, keepdims=True)+1e-10)+1e-10))/np.log(self.n_classes)
        knowledge_gain = np.sum(probs*np.log(np.sum(probs, axis=(0, 2), keepdims=True) * probs / (np.sum(probs, axis=0, keepdims=True)+1e-10) / (np.sum(probs, axis=2, keepdims=True)+1e-10)+1e-10))/np.log(self.n_classes)
        prob_gt_y2 = probs.sum(axis=1)
        total_knowledge = np.sum(prob_gt_y2*np.log(prob_gt_y2/(np.sum(prob_gt_y2, axis=0, keepdims=True)+1e-10)/(np.sum(prob_gt_y2, axis=1, keepdims=True)+1e-10)+1e-10))/np.log(self.n_classes)
        retained_knowledge = total_knowledge - knowledge_gain

        return forgetting, knowledge_gain, total_knowledge, retained_knowledge

    def n_samples(self, n_samples):
        self.total_samples = n_samples
        
    
    def get_flops_parameter(self, method=None):
        inp_size = self.args.image_size
        
        self.flops_dict = get_model_complexity_info(self.model, (3, inp_size[0], inp_size[1]),
                                                                    as_strings=False,
                                                                    print_per_layer_stat=False, verbose=True,
                                                                    original_opt=self.optimizer,
                                                                    opt_name=self.opt_name, lr=self.lr)
        forward_flops, backward_flops, G_forward_flops, G_backward_flops, F_forward_flops, F_backward_flops  = get_blockwise_flops(self.flops_dict, self.model_name, method)
        self.forward_flops = sum(forward_flops)
        self.backward_flops = sum(backward_flops)
        self.blockwise_forward_flops = forward_flops
        self.blockwise_backward_flops = backward_flops
        self.total_model_flops = self.forward_flops + self.backward_flops
        
        self.G_forward_flops, self.G_backward_flops = sum(G_forward_flops), sum(G_backward_flops)
        self.F_forward_flops, self.F_backward_flops = sum(F_forward_flops), sum(F_backward_flops)
        self.G_blockwise_forward_flops, self.G_blockwise_backward_flops = G_forward_flops, G_backward_flops
        self.F_blockwise_forward_flops, self.F_blockwise_backward_flops = F_forward_flops, F_backward_flops

    def save(self, sample_num, save_path=None):
        if save_path is None:
            return
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_path = f'model_{self.dataset}_{self.mode}_mem{self.memory_size}_lr{self.lr}_online_iter{self.online_iter}_seed{self.rnd_seed}.pth'
        torch.save(self.model.state_dict(), os.path.join(save_path, model_path))
        # with open(os.path.join(save_path, f'args.pkl'), 'wb') as f:
        #     pickle.dump(self.args, f)
        # self.save_std_pickle()
        
        logger.info(f"Sample {sample_num} | Model saved at {save_path}/{model_path}")