import logging
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.er_baseline import ER
import torch.nn.functional as F
from utils.data_loader import ImageDataset, StreamDataset, MemoryDataset, cutmix_data, get_statistics
from utils.train_utils import select_model, select_optimizer, select_scheduler, MeanAveragePrecisionCustomized
import pdb

logger = logging.getLogger(__name__)

class LD(ER):
    def __init__(self, criterion, n_classes, device, **kwargs):
        super().__init__(criterion, n_classes, device, **kwargs)

        self.alpha = kwargs.get("alpha", 0.1)
        self.frozen_point = kwargs.get("frozen_point", 22)
        self.teacher_model = None

        logger.info(f"[LD INIT] α: {self.alpha}, frozen_point: {self.frozen_point}")


    def online_step(self, sample, sample_num, n_worker):
        self.temp_batchsize = self.batch_size # LD는 전체 배치를 사용
    
        if sample['klass'] not in self.exposed_classes:
            self.online_after_task(sample_num)
            self.add_new_class(sample['klass'])

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
        frozen_count = 0
        total_count = 0
        for i, (name, param) in enumerate(self.model.named_parameters()):
            total_count += 1
            # '.model.' 다음의 숫자 인덱스를 기준으로 동결
            try:
                layer_index = int(name.split('.')[1])
                if layer_index < self.frozen_point:
                    param.requires_grad = False
                    frozen_count += 1
            except (IndexError, ValueError):
                 logger.warning(f"Could not parse layer index from {name}, not freezing.")
                 param.requires_grad = True

        logger.info(f"Student model backbone frozen: {frozen_count}/{total_count} parameters frozen (up to layer {self.frozen_point}).")
        
        # 3. Optimizer 재설정
        logger.info("Resetting optimizer for trainable parameters.")
        self.optimizer = select_optimizer(self.opt_name, self.model, lr=self.lr)

    def model_forward(self, batch):
        batch = self.preprocess_batch(batch)

        if self.teacher_model is None:
            return super().model_forward(batch)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            student_outputs = self.model(batch["img"])
            student_aux_raw = student_outputs["AUX"]
            student_main_raw = student_outputs["Main"]
            student_aux_predicts = self.vec2box(student_aux_raw)
            student_main_predicts = self.vec2box(student_main_raw)

            with torch.no_grad():
                teacher_outputs = self.teacher_model(batch["img"])
                teacher_aux_raw = teacher_outputs["AUX"]
                teacher_main_raw = teacher_outputs["Main"]
                teacher_aux_predicts = self.vec2box(teacher_aux_raw)
                teacher_main_predicts = self.vec2box(teacher_main_raw)

            # 1. 기본 모델 손실 계산
            loss, loss_item = self.model.loss_fn(student_aux_predicts, student_main_predicts, batch['cls'])
            
            # 2. 증류 손실 계산 (Student와 Teacher 출력 사용 - 이전 클래스에 대해서만)
            dist_loss = 0.0
            num_old_cls = self.num_learned_class - 1 

            if num_old_cls > 0:
                # Vec2Box 출력에서 클래스 예측 부분 추출 [B, A, C]
                student_aux_cls = student_aux_predicts[0]
                student_main_cls = student_main_predicts[0]
                teacher_aux_cls = teacher_aux_predicts[0]
                teacher_main_cls = teacher_main_predicts[0]

                # 이전 클래스에 해당하는 부분만 슬라이싱
                student_aux_old = student_aux_cls[..., :num_old_cls]
                student_main_old = student_main_cls[..., :num_old_cls]
                teacher_aux_old = teacher_aux_cls[..., :num_old_cls]
                teacher_main_old = teacher_main_cls[..., :num_old_cls]

                # MSE 손실 계산
                dist_loss += F.mse_loss(student_aux_old, teacher_aux_old)
                dist_loss += F.mse_loss(student_main_old, teacher_main_old)
            
            print(f"loss: {loss}, dist_loss: {dist_loss}")
            
            # 3. 최종 손실 = 모델 손실 + alpha * 증류 손실
            total_loss = loss + self.alpha * dist_loss
            
        return total_loss, loss_item

    def update_memory(self, sample):
        self.seen += 1
        pass