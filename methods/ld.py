import logging
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.er_baseline import ER
import torch.nn.functional as F
from utils.data_loader import ImageDataset, StreamDataset, MemoryDataset, cutmix_data, get_statistics
from utils.train_utils import select_model, select_optimizer, select_scheduler, MeanAveragePrecisionCustomized
import types
import pdb

logger = logging.getLogger(__name__)

class LD(ER):
    def __init__(self, criterion, n_classes, device, **kwargs):
        super().__init__(criterion, n_classes, device, **kwargs)

        self.alpha = kwargs.get("alpha", 1.0)
        self.frozen_point = kwargs.get("frozen_point", 3)
        self.teacher_model = None

        logger.info(f"[LD INIT] α: {self.alpha}, frozen_point: {self.frozen_point}")
        
        # self.backward_flops = self.blockwise_backward_flops[-1]


    def online_step(self, sample, sample_num, n_worker):
        # self.temp_batchsize = self.batch_size # LD는 전체 배치를 사용
    
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

                self.temp_batch = []
                self.num_updates -= int(self.num_updates)

    def online_after_task(self, cur_iter):
        print("[DEBUG] Task finished. Setting up teacher and freezing backbone for LD.")
        
        # 1. 현재 모델을 Teacher로 복사하고 동결
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        logger.info("Teacher model created and frozen.")

        # 2. Student 모델의 Backbone 동결
        frozen_cnt = 0
        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "block_list"):
            for i, m in enumerate(self.model.backbone.block_list):
                if i < self.frozen_point:
                    for p in m.parameters():
                        if p.requires_grad:
                            p.requires_grad = False
                            frozen_cnt += 1

        # self.optimizer = select_optimizer(self.opt_name, self.model, lr=self.lr)

    def model_forward(self, batch):
        batch = self.preprocess_batch(batch)

        if self.teacher_model is None:
            return super().model_forward(batch)

        images = batch.get("img").to(self.device, non_blocking=True)
        cls_targets = batch.get("cls")
        
        batch_size = images.shape[0]
        formatted_targets = []
        for i in range(batch_size):
            img_targets = cls_targets[i]
            valid_targets = img_targets[img_targets[:, 0] != -1]

            target_obj = types.SimpleNamespace()

            if valid_targets.numel() > 0:
                target_obj.bbox = valid_targets[:, 1:]
                cls_tensor = valid_targets[:, 0].long()
                target_obj._cls_tensor = cls_tensor
            else:
                target_obj.bbox = torch.empty(0, 4).to(self.device)
                target_obj._cls_tensor = torch.empty(0).to(self.device).long()
            
            target_obj.get_field = lambda field_name, t=target_obj: t._cls_tensor if field_name == 'labels' else None
            
            formatted_targets.append(target_obj)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            loss_item = self.model(images, targets=formatted_targets)
            loss = loss_item["total_loss"]
            
            dist_loss = 0.0
            
            with torch.no_grad():
                tea_feats = self.teacher_model.backbone(images)
                tea_neck  = self.teacher_model.neck(tea_feats)
                teacher_cls_list = [self.teacher_model.head.gfl_cls[i](tea_neck[i]) for i in range(len(self.teacher_model.head.gfl_cls))]
                teacher_reg_list = [self.teacher_model.head.gfl_reg[i](tea_neck[i]) for i in range(len(self.teacher_model.head.gfl_reg))]

            stu_feats = self.model.backbone(images)
            stu_neck  = self.model.neck(stu_feats)
            student_cls_list = [self.model.head.gfl_cls[i](stu_neck[i]) for i in range(len(self.model.head.gfl_cls))]
            student_reg_list = [self.model.head.gfl_reg[i](stu_neck[i]) for i in range(len(self.model.head.gfl_reg))]

            teacher_cls_logits = torch.cat([t.permute(0,2,3,1).reshape(-1, t.shape[1]) for t in teacher_cls_list], dim=0)
            student_cls_logits = torch.cat([t.permute(0,2,3,1).reshape(-1, t.shape[1]) for t in student_cls_list], dim=0)
                
            C_old = teacher_cls_logits.size(-1)
            student_cls_logits = student_cls_logits[..., :C_old]

            dist_loss = F.mse_loss(student_cls_logits, teacher_cls_logits)
            
            print(f"loss: {loss}, dist_loss: {dist_loss}")
            
            # 3. 최종 손실 = 모델 손실 + alpha * 증류 손실
            total_loss = loss + self.alpha * dist_loss
        
            # self.total_flops += (len(batch["img"]) * self.forward_flops * 2)
            
        return total_loss, loss_item

    def update_memory(self, sample):
        self.seen += 1
        pass