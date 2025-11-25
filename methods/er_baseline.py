import logging
import copy
import time
import datetime
import numpy as np
import os
import torch
import torch.nn as nn
from utils.data_loader import MemoryDataset, get_exposed_classes
from utils.train_utils import select_model, select_optimizer, select_scheduler, boxlist_to_pred_dict, boxlist_to_target_dict
#### object detection
import random

from utils.block_utils import get_blockwise_flops, MODEL_BLOCK_DICT

from damo.dataset.build import build_dataset, build_dataloader
# from damo.utils.boxes import postprocess
from damo.config.base import parse_config

from tqdm import tqdm

from contextlib import redirect_stdout
from calflops import calculate_flops
from utils.flops_utils import blockwise_from_log_file

from damo.apis.detector_inference import compute_on_dataset
from damo.dataset.datasets.evaluation import evaluate
from damo.utils.timer import Timer

logger = logging.getLogger()

class ER:
    def __init__(self, n_classes, device, **kwargs):
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
        self.model_name = kwargs["model_name"]
        self.opt_name = kwargs["opt_name"]
        self.sched_name = 'const' if kwargs["sched_name"] == "default" else kwargs["sched_name"]
        self.data_dir = kwargs["data_dir"]
        self.online_iter = kwargs["online_iter"]
        self.use_amp = kwargs["use_amp"]
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.exposed_classes = get_exposed_classes(self.dataset)
        self.num_learned_class = len(self.exposed_classes)
        self.num_learning_class = self.num_learned_class + 1
        
        
        if 'VOC_10_10' in self.dataset:
            data_name = 'voc'
            config_file = 'configs/damoyolo_tinynasL25_S_VOC_10_10.py'
        elif 'VOC_15_5' in self.dataset:
            data_name = 'voc'
            config_file = 'configs/damoyolo_tinynasL25_S_VOC_15_5.py'
        elif 'BDD' in self.dataset:
            data_name = 'bdd100k'
            config_file = 'configs/damoyolo_tinynasL25_S_BDD100K.py'
        elif 'SHIFT' in self.dataset:
            data_name = 'shift'
            config_file = 'configs/damoyolo_tinynasL25_S_SHIFT.py'
        elif 'MILITARY_SYNTHETIC' in self.dataset:
            data_name = 'military_synthetic'
            config_file = 'configs/damoyolo_tinynasL25_S_MILITARY_SYNTHETIC.py'
        elif 'VisDrone_3_4' in self.dataset:
            data_name = 'visdrone'
            config_file = 'configs/damoyolo_tinynasL25_S_VisDrone_3_4.py'
        elif 'COCO_70_10' in self.dataset:
            data_name = 'coco'
            config_file = 'configs/damoyolo_tinynasL25_S_COCO_70_10.py'
        elif 'COCO_60_20' in self.dataset:
            data_name = 'coco'
            config_file = 'configs/damoyolo_tinynasL25_S_COCO_60_20.py'
        self.damo_cfg = parse_config(config_file)
        
        self.exposed_domains = [f'{data_name}_source']
        self.model = select_model(self.dataset, self.damo_cfg)

        self.stride = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)
        self.optimizer = select_optimizer(self.opt_name, self.model, lr=self.lr, cfg=self.damo_cfg.train.optimizer)
        self.scheduler = None  # optional

        val_dataset_names = self.damo_cfg.dataset.val_ann
        val_datasets = build_dataset(self.damo_cfg, val_dataset_names, is_train=False)
        
        val_augment_config = self.damo_cfg.test.augment
        val_dataloaders = build_dataloader(
            datasets=val_datasets,
            augment=val_augment_config,
            batch_size=self.damo_cfg.test.batch_size,
            is_train=False
        )
        self.val_loader = val_dataloaders[0]
        
        self.img_size = [640, 640]
        
        self.selection_method = kwargs["selection_method"]
        self.priority_selection = kwargs.get("priority_selection", None)
        self.initialize_memory_buffer(kwargs["memory_size"])
        
        self.temp_batch = []
        self.num_updates = 0
        self.train_count = 0

        self.total_flops = 0.0
        self.start_time = time.time()

        self.save_path = f'results/{self.dataset}/{self.note}/seed_{self.rnd_seed}'
        
        # test arguments
        # self.metric = MeanAveragePrecisionCustomized(iou_type="bbox", box_format="xyxy",class_metrics=True)#, backend="faster_coco_eval")
        # self.metric.warn_on_many_detections = False
        
        self.model = self.model.to(self.device)
        
        self.block_names = MODEL_BLOCK_DICT[self.model_name]
        self.num_blocks = len(self.block_names) - 1
        self.get_flops_parameter()
     
    def initialize_memory_buffer(self, memory_size):
        self.memory_size = memory_size - self.temp_batchsize
        data_args = self.damo_cfg.get_data(self.damo_cfg.dataset.train_ann[0])
        self.memory = MemoryDataset(ann_file=data_args['args']['ann_file'], root=data_args['args']['root'], transforms=None,class_names=self.damo_cfg.dataset.class_names,
            dataset=self.dataset, cls_list=self.exposed_classes, device=self.device, memory_size=self.memory_size, image_size=self.img_size, aug=self.damo_cfg.train.augment)
     
    def get_total_flops(self):
        return self.total_flops
        
    def online_step(self, sample, sample_num, n_worker):
        if sample.get('klass',None) and sample['klass'] not in self.exposed_classes:
            self.online_after_task(sample_num)
            self.add_new_class(sample['klass'])
        elif sample.get('domain',None) and sample['domain'] not in self.exposed_domains:
            self.online_after_task(sample_num)
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

    def add_new_class(self, class_name):
        self.exposed_classes.append(class_name)
        print('self.exposed_classes', self.exposed_classes)
        self.num_learned_class = len(self.exposed_classes)
        
        self.model.head.num_classes = self.num_learned_class
        self.model.head.cls_out_channels = self.num_learned_class
        
        prev_weights = [
            copy.deepcopy(self.model.head.gfl_cls[0].weight),
            copy.deepcopy(self.model.head.gfl_cls[1].weight),
            copy.deepcopy(self.model.head.gfl_cls[2].weight),
        ]
        prev_biases = [
            copy.deepcopy(self.model.head.gfl_cls[0].bias),
            copy.deepcopy(self.model.head.gfl_cls[1].bias),
            copy.deepcopy(self.model.head.gfl_cls[2].bias),
        ]
    
        self.model.head.gfl_cls[0] = nn.Conv2d(self.model.head.gfl_cls[0].in_channels, self.num_learned_class, self.model.head.gfl_cls[0].kernel_size, self.model.head.gfl_cls[0].stride, self.model.head.gfl_cls[0].padding, bias=True).to(self.device)
        self.model.head.gfl_cls[1] = nn.Conv2d(self.model.head.gfl_cls[1].in_channels, self.num_learned_class, self.model.head.gfl_cls[1].kernel_size, self.model.head.gfl_cls[1].stride, self.model.head.gfl_cls[1].padding, bias=True).to(self.device)
        self.model.head.gfl_cls[2] = nn.Conv2d(self.model.head.gfl_cls[2].in_channels, self.num_learned_class, self.model.head.gfl_cls[2].kernel_size, self.model.head.gfl_cls[2].stride, self.model.head.gfl_cls[2].padding, bias=True).to(self.device)
    
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.head.gfl_cls[0].weight[:self.num_learned_class - 1] = prev_weights[0]
                self.model.head.gfl_cls[0].bias[:self.num_learned_class - 1] = prev_biases[0]
                self.model.head.gfl_cls[1].weight[:self.num_learned_class - 1] = prev_weights[1]
                self.model.head.gfl_cls[1].bias[:self.num_learned_class - 1] = prev_biases[1]
                self.model.head.gfl_cls[2].weight[:self.num_learned_class - 1] = prev_weights[2]
                self.model.head.gfl_cls[2].bias[:self.num_learned_class - 1] = prev_biases[2]
                
                self.model.head.gfl_cls[0].weight[self.num_learned_class - 1] = prev_weights[0].mean(dim=0)
                self.model.head.gfl_cls[0].bias[self.num_learned_class - 1] = prev_biases[0].mean(dim=0)
                self.model.head.gfl_cls[1].weight[self.num_learned_class - 1] = prev_weights[1].mean(dim=0)
                self.model.head.gfl_cls[1].bias[self.num_learned_class - 1] = prev_biases[1].mean(dim=0)
                self.model.head.gfl_cls[2].weight[self.num_learned_class - 1] = prev_weights[2].mean(dim=0)
                self.model.head.gfl_cls[2].bias[self.num_learned_class - 1] = prev_biases[2].mean(dim=0)
                
        
        for param in self.optimizer.param_groups[3]['params']:
            if param in self.optimizer.state.keys():
                del self.optimizer.state[param]
        for param in self.optimizer.param_groups[4]['params']:
            if param in self.optimizer.state.keys():
                del self.optimizer.state[param]
        
        del self.optimizer.param_groups[4], self.optimizer.param_groups[3]
        self.optimizer.add_param_group({'params': [
                                        self.model.head.gfl_cls[0].weight,
                                        self.model.head.gfl_cls[1].weight,
                                        self.model.head.gfl_cls[2].weight,
                                        ], })
        self.optimizer.add_param_group({'params': [
                                        self.model.head.gfl_cls[0].bias,
                                        self.model.head.gfl_cls[1].bias,
                                        self.model.head.gfl_cls[2].bias,
                                        ], "weight_decay": 0})
        self.memory.add_new_class(cls_list=self.exposed_classes)
        
        print("Successfully added new class and updated model/optimizer.")
        
        # if 'reset' in self.sched_name:
        #     self.update_schedule(reset=True)
        # self.model.set_loss_function(self.args, self.vec2box, self.num_learned_class)

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

            total_loss += loss.item()
            
        return total_loss / iterations
    
    def model_forward(self, batch):
        inps, targets = self.preprocess_batch(batch)
        
        # with torch.cuda.amp.autocast(enabled=self.use_amp):
        with torch.cuda.amp.autocast(enabled=False):
            loss_item = self.model(inps, targets)
            total_loss = loss_item["total_loss"]
            
            self.total_flops += (len(targets) * self.forward_flops)
        
        return total_loss, loss_item
    
    def preprocess_batch(self, batch):
        inps = batch[0].to(self.device)
        targets = [target.to(self.device) for target in batch[1]]
        return inps, targets

    def report_training(self, sample_num, train_loss):

        logger.info(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | "
            f"TFLOPs {self.total_flops/1e12:.2f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))} | "
        )

    def report_test(self, sample_num, avg_acc, classwise_acc):
        # self.writer.add_scalar(f"test/loss", avg_loss, sample_num)
        # self.writer.add_scalar(f"test/acc", avg_acc, sample_num)
        # self.writer.add_scalar(f"test/classwise_acc", classwise_acc, sample_num)
        logger.info(
            f"Test | Sample # {sample_num} | test_acc {avg_acc:.4f} | \n"
            f"classwise mAP50 [{'|'.join([str(round(cls_acc,3)) for cls_acc in classwise_acc])}] | \n"
            f"TFLOPs {self.total_flops/1e12:.2f} | "
        )

    def update_memory(self, sample):
        self.reservoir_memory(sample)

    def update_schedule(self, reset=False):
        pass   

    def evaluate(self):
        # alternative eval
        inference_timer = Timer()
        predictions = compute_on_dataset(self.model, self.val_loader, self.device, inference_timer)
        
        extra_args = dict(
            box_only=False,
            iou_types=('bbox', ),
            expected_results=(),
            expected_results_sigma_tol=4,
        )
        result = evaluate(self.val_loader.dataset, predictions, None, **extra_args)
        
        coco_eval = result[2]
        prec = coco_eval.eval['precision']  # [T, R, K, A, M]
        iou_thrs = coco_eval.params.iouThrs
        cat_ids = coco_eval.params.catIds
        area_idx = 0   # all
        maxdet_idx = -1  # use last (usually 100)

        t = np.where(np.isclose(iou_thrs, 0.5))[0][0]

        ap50_per_class = []
        for k, catId in enumerate(cat_ids):
            
            s = prec[t, :, k, area_idx, maxdet_idx]
            s = s[s > -1] 
            ap50_per_class.append(np.mean(s) if s.size else float("nan"))

        eval_dict = {
            "avg_mAP50": sum(ap50_per_class[:self.num_learned_class])/self.num_learned_class,
            "classwise_mAP50": ap50_per_class[:self.num_learned_class]
        }
        
        # preds_lists = []
        # targs_lists = []
        # for i, batch in enumerate(tqdm(self.val_loader)):
        #     images_obj, targets, _ = batch
            
        #     images_obj = images_obj.to(self.device)
        #     with torch.no_grad():
        #         predicts = self.model(images_obj)
            
        #     preds_list = [boxlist_to_pred_dict(p) for p in predicts]
        #     targs_list = [boxlist_to_target_dict(t) for t in targets]
        #     preds_lists.extend(preds_list)
        #     targs_lists.extend(targs_list)
        # self.metric(preds_lists, targs_lists)
        
        # # pdb.set_trace()
        # epoch_metrics = self.metric.compute()
        # del epoch_metrics["classes"]
        # eval_dict = {
        #     "avg_mAP50": sum(epoch_metrics['map50_per_class'].tolist()[:self.num_learned_class])/self.num_learned_class,
        #     "classwise_mAP50": epoch_metrics['map50_per_class'].tolist()[:self.num_learned_class]
        # }
        # self.metric.reset()
        return eval_dict

    def online_evaluate(self, sample_num, data_time):
        torch.cuda.empty_cache()
        self.model.eval()
        print("evaluate")

        if self.dataset=='BDD_domain' or self.dataset=='BDD_domain_small':
            eval_dict = {"avg_mAP50":0, "classwise_mAP50":[]}
            for data_name in ['bdd100k_source','bdd100k_cloudy', 'bdd100k_rainy', 'bdd100k_dawndusk', 'bdd100k_night']: #self.exposed_domains:
                datasets = build_dataset(
                    cfg=self.damo_cfg,
                    ann_files=[data_name],
                    is_train=False
                )
                dataloaders = build_dataloader(
                    datasets=datasets,
                    augment=self.damo_cfg.test.augment,
                    batch_size=self.damo_cfg.train.batch_size,
                    is_train=False,
                    num_workers=self.damo_cfg.train.get('num_workers', 8)
                )
                self.val_loader = dataloaders[0]

                eval_dict_sub = self.evaluate()
                clean = [v for v in eval_dict_sub['classwise_mAP50'] if v != -1]
                average = sum(clean) / len(clean) if clean else 0.0
                eval_dict['avg_mAP50'] += average / 5 #len(self.exposed_domains)
                eval_dict["classwise_mAP50"].append(average)

        elif self.dataset=='SHIFT_domain' or self.dataset=='SHIFT_domain_small':
            eval_dict = {"avg_mAP50":0, "classwise_mAP50":[]}
            for data_name in ['shift_source', 'shift_overcast', 'shift_cloudy', 'shift_rainy', 'shift_foggy', 'shift_dawndusk', 'shift_night']: #self.exposed_domains:
                datasets = build_dataset(self.damo_cfg, [data_name + '_val'], is_train=False)
                dataloaders = build_dataloader(
                    datasets,
                    self.damo_cfg.test.augment,
                    batch_size=self.damo_cfg.train.batch_size,
                    is_train=False,
                    num_workers=self.damo_cfg.train.get('num_workers', 8)
                )
                self.val_loader = dataloaders[0]

                eval_dict_sub = self.evaluate()
                clean = [v for v in eval_dict_sub['classwise_mAP50'] if v != -1]
                average = sum(clean) / len(clean) if clean else 0.0
                eval_dict['avg_mAP50'] += average / 7 #len(self.exposed_domains)
                eval_dict["classwise_mAP50"].append(average)
        elif self.dataset=='SHIFT_domain_small2':
            eval_dict = {"avg_mAP50":0, "classwise_mAP50":[]}
            for data_name in ['shift_source', 'shift_dawndusk', 'shift_night','shift_foggy']: #self.exposed_domains:
                datasets = build_dataset(self.damo_cfg, [data_name + '_val'], is_train=False)
                dataloaders = build_dataloader(
                    datasets,
                    self.damo_cfg.test.augment,
                    batch_size=self.damo_cfg.train.batch_size,
                    is_train=False,
                    num_workers=self.damo_cfg.train.get('num_workers', 8)
                )
                self.val_loader = dataloaders[0]

                eval_dict_sub = self.evaluate()
                clean = [v for v in eval_dict_sub['classwise_mAP50'] if v != -1]
                average = sum(clean) / len(clean) if clean else 0.0
                eval_dict['avg_mAP50'] += average / 4 #len(self.exposed_domains)
                eval_dict["classwise_mAP50"].append(average)

        elif self.dataset=='SHIFT_hanhwa_scenario1':
            eval_dict = {"avg_mAP50":0, "classwise_mAP50":[]}
            for data_name in ['shift_source','shift_daytime_clear1', 'shift_dawndusk_clear1', 'shift_night_clear', 'shift_dawndusk_clear2', 'shift_daytime_clear2']:
                datasets = build_dataset(self.damo_cfg, [data_name + '_val'], is_train=False)
                dataloaders = build_dataloader(
                    datasets,
                    self.damo_cfg.test.augment,
                    batch_size=self.damo_cfg.train.batch_size,
                    is_train=False,
                    num_workers=self.damo_cfg.train.get('num_workers', 8)
                )
                self.val_loader = dataloaders[0]

                eval_dict_sub = self.evaluate()
                clean = [v for v in eval_dict_sub['classwise_mAP50'] if v != -1]
                average = sum(clean) / len(clean) if clean else 0.0
                eval_dict['avg_mAP50'] += average / 6 #len(self.exposed_domains)
                eval_dict["classwise_mAP50"].append(average)
        
        elif self.dataset=='SHIFT_hanhwa_scenario2':
            eval_dict = {"avg_mAP50":0, "classwise_mAP50":[]}
            for data_name in ['shift_source', 'shift_daytime_clear1', 'shift_dawndusk_cloudy', 'shift_night_foggy', 'shift_dawndusk_overcast', 'shift_daytime_clear2']:
                datasets = build_dataset(self.damo_cfg, [data_name + '_val'], is_train=False)
                dataloaders = build_dataloader(
                    datasets,
                    self.damo_cfg.test.augment,
                    batch_size=self.damo_cfg.train.batch_size,
                    is_train=False,
                    num_workers=self.damo_cfg.train.get('num_workers', 8)
                )
                self.val_loader = dataloaders[0]

                eval_dict_sub = self.evaluate()
                clean = [v for v in eval_dict_sub['classwise_mAP50'] if v != -1]
                average = sum(clean) / len(clean) if clean else 0.0
                eval_dict['avg_mAP50'] += average / 6 #len(self.exposed_domains)
                eval_dict["classwise_mAP50"].append(average)
        
        elif self.dataset=='SHIFT_hanhwa_scenario3':
            eval_dict = {"avg_mAP50":0, "classwise_mAP50":[]}
            for data_name in ['shift_source', 'shift_daytime_cloudy', 'shift_dawndusk_rainy', 'shift_night_foggy', 'shift_dawndusk_overcast', 'shift_daytime_clear1']:
                datasets = build_dataset(self.damo_cfg, [data_name + '_val'], is_train=False)
                dataloaders = build_dataloader(
                    datasets,
                    self.damo_cfg.test.augment,
                    batch_size=self.damo_cfg.train.batch_size,
                    is_train=False,
                    num_workers=self.damo_cfg.train.get('num_workers', 8)
                )
                self.val_loader = dataloaders[0]

                eval_dict_sub = self.evaluate()
                clean = [v for v in eval_dict_sub['classwise_mAP50'] if v != -1]
                average = sum(clean) / len(clean) if clean else 0.0
                eval_dict['avg_mAP50'] += average / 6 #len(self.exposed_domains)
                eval_dict["classwise_mAP50"].append(average)
        
        elif self.dataset=='SHIFT_hanhwa_scenario4':
            eval_dict = {"avg_mAP50":0, "classwise_mAP50":[]}
            for data_name in ['shift_source', 'shift_daytime_foggy', 'shift_dawndusk_clear', 'shift_night_rainy', 'shift_dawndusk_cloudy', 'shift_daytime_overcast']:
                datasets = build_dataset(self.damo_cfg, [data_name + '_val'], is_train=False)
                dataloaders = build_dataloader(
                    datasets,
                    self.damo_cfg.test.augment,
                    batch_size=self.damo_cfg.train.batch_size,
                    is_train=False,
                    num_workers=self.damo_cfg.train.get('num_workers', 8)
                )
                self.val_loader = dataloaders[0]

                eval_dict_sub = self.evaluate()
                clean = [v for v in eval_dict_sub['classwise_mAP50'] if v != -1]
                average = sum(clean) / len(clean) if clean else 0.0
                eval_dict['avg_mAP50'] += average / 6 #len(self.exposed_domains)
                eval_dict["classwise_mAP50"].append(average)
        
        elif self.dataset=='SHIFT_hanhwa_scenario5':
            eval_dict = {"avg_mAP50":0, "classwise_mAP50":[]}
            for data_name in ['shift_source', 'shift_daytime_rainy', 'shift_dawndusk_foggy', 'shift_night_clear', 'shift_dawndusk_cloudy', 'shift_daytime_rainy']:
                datasets = build_dataset(self.damo_cfg, [data_name + '_val'], is_train=False)
                dataloaders = build_dataloader(
                    datasets,
                    self.damo_cfg.test.augment,
                    batch_size=self.damo_cfg.train.batch_size,
                    is_train=False,
                    num_workers=self.damo_cfg.train.get('num_workers', 8)
                )
                self.val_loader = dataloaders[0]

                eval_dict_sub = self.evaluate()
                clean = [v for v in eval_dict_sub['classwise_mAP50'] if v != -1]
                average = sum(clean) / len(clean) if clean else 0.0
                eval_dict['avg_mAP50'] += average / 6 #len(self.exposed_domains)
                eval_dict["classwise_mAP50"].append(average)
        
        elif 'MILITARY_SYNTHETIC_domain' in self.dataset:
            eval_dict = {"avg_mAP50":0, "classwise_mAP50":[]}
            domain_list = ['military_synthetic_domain_source', 'military_synthetic_domain_night', 'military_synthetic_domain_winter', 'military_synthetic_domain_infrared']
            for data_name in domain_list:
                datasets = build_dataset(self.damo_cfg, [data_name], is_train=False)
                dataloaders = build_dataloader(
                    datasets,
                    self.damo_cfg.test.augment,
                    batch_size=self.damo_cfg.train.batch_size,
                    is_train=False,
                    num_workers=self.damo_cfg.train.get('num_workers', 8)
                )
                self.val_loader = dataloaders[0]

                eval_dict_sub = self.evaluate()
                clean = [v for v in eval_dict_sub['classwise_mAP50'] if v != -1]
                average = sum(clean) / len(clean) if clean else 0.0
                eval_dict['avg_mAP50'] += average / len(domain_list)
                eval_dict["classwise_mAP50"].append(average)
                
        else:
            eval_dict = self.evaluate()

        self.report_test(sample_num, eval_dict["avg_mAP50"], eval_dict["classwise_mAP50"])
        
        return eval_dict

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


    def n_samples(self, n_samples):
        self.total_samples = n_samples
        
    
    def get_flops_parameter(self, method=None):
        input_shape = (1, 3, self.img_size[0], self.img_size[1])
        
        model_for_info = copy.deepcopy(self.model)
        flops_summary_path = "flops_summary.txt"
        with open(flops_summary_path, "w", encoding="utf-8") as f:
            with redirect_stdout(f):
                flops, macs, _ = calculate_flops(
                    model=model_for_info,
                    input_shape=input_shape,
                    output_as_string=False,
                    output_precision=4,
                    print_detailed=True,
                )
        params = sum(p.numel() for p in model_for_info.parameters())
        del model_for_info
        forward_flops = blockwise_from_log_file(flops_summary_path, self.block_names, unit='')
        backward_flops = [flop * 2 for flop in forward_flops]  # Assuming backward pass has double the flops of forward pass
        
        self.forward_flops = sum(forward_flops)
        self.backward_flops = sum(backward_flops)
        self.blockwise_forward_flops = forward_flops
        self.blockwise_backward_flops = backward_flops
        self.total_model_flops = self.forward_flops + self.backward_flops
        
        print(f"Model FLOPs: {self.total_model_flops/1e9:.2f} GFLOPs | Model forward FLOPs: {self.forward_flops/1e9:.2f} GFLOPs | Model backward FLOPs: {self.backward_flops/1e9:.2f} GFLOPs")



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
        
        logger.info(f"Sample {sample_num} | Modelc saved at {save_path}/{model_path}")