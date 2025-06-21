import torch
import torch.nn.functional as F
import copy
import logging
from methods.er_baseline import ER

from tqdm import tqdm
from yolo.utils.bounding_box_utils import create_converter, to_metrics_format

import pdb

logger = logging.getLogger()

class ERD(ER):
    def __init__(self, criterion, n_classes, device, **kwargs):
        super().__init__(criterion, n_classes, device, **kwargs)
        self.old_model = None
        self.lambda_cls = kwargs.get("lambda_cls", 1.0)
        self.lambda_reg = kwargs.get("lambda_reg", 1.0)
        self.alpha_cls = kwargs.get("alpha_cls", 2.0)
        self.alpha_reg = kwargs.get("alpha_reg", 2.0)
        logger.info(f"[ERD INIT] λ_cls: {self.lambda_cls}, λ_reg: {self.lambda_reg}, α_cls: {self.alpha_cls}, α_reg: {self.alpha_reg}")

    def model_forward_with_erd(self, current_batch):
        current_batch = self.preprocess_batch(current_batch)
        outputs = self.model(current_batch["img"])
        aux_preds = self.vec2box(outputs["AUX"])
        main_preds = self.vec2box(outputs["Main"])

        loss_model, _ = self.model.loss_fn(aux_preds, main_preds, current_batch['cls'])
        loss_cls_distill, loss_reg_distill = 0.0, 0.0

        if self.old_model is not None:
            with torch.no_grad():
                old_outputs = self.old_model(current_batch["img"])
                old_aux_preds = self.vec2box(old_outputs["AUX"])
                old_main_preds = self.vec2box(old_outputs["Main"])

            C_old = old_aux_preds[0].shape[-1]
            teacher_cls_logits = old_aux_preds[0][..., :C_old]
            student_cls_logits = aux_preds[0][..., :C_old]

            mask_cls, old_cls_sel, new_cls_sel = self._elastic_response_selection(
                teacher_cls_logits, student_cls_logits, alpha=self.alpha_cls
            )
            loss_cls_distill = self.distill_cls_loss(old_cls_sel, new_cls_sel, mask_cls)

            mask_reg, old_reg_sel, new_reg_sel = self._elastic_response_selection(
                old_aux_preds[1], aux_preds[1], alpha=self.alpha_reg
            )
            loss_reg_distill = self.distill_bbox_loss(old_reg_sel, new_reg_sel, mask_reg)

        logger.info(f"loss_model: {loss_model.item():.4f}, cls_loss: {loss_cls_distill:.4f}, reg_loss: {loss_reg_distill:.4f}")
        total_loss = loss_model + self.lambda_cls * loss_cls_distill + self.lambda_reg * loss_reg_distill
        return total_loss

    def _elastic_response_selection(self, teacher_tensor, student_tensor, alpha=2.0):
        with torch.no_grad():
            teacher_tensor = teacher_tensor[..., :student_tensor.shape[-1]]
            maxval, _ = teacher_tensor.max(dim=-1)
            mean = maxval.mean()
            std = maxval.std()
            threshold = mean + alpha * std
            mask = maxval > threshold

        return mask, teacher_tensor[mask], student_tensor[mask]

    def distill_cls_loss(self, teacher_logits, student_logits, mask):
        if teacher_logits.numel() == 0:
            return torch.tensor(0.0, device=teacher_logits.device)
        t = 2.0
        teacher_probs = F.softmax(teacher_logits / t, dim=-1)
        student_probs = F.softmax(student_logits / t, dim=-1)
        return F.mse_loss(student_probs, teacher_probs)

    def distill_bbox_loss(self, teacher_dist, student_dist, mask):
        if teacher_dist.numel() == 0:
            return torch.tensor(0.0, device=teacher_dist.device)
        teacher_soft = F.log_softmax(teacher_dist, dim=-1)
        student_soft = F.log_softmax(student_dist, dim=-1)
        return F.kl_div(student_soft, teacher_soft, reduction='batchmean', log_target=True)

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        total_loss = 0.0
        if len(sample) > 0:
            self.memory.register_stream(sample)

        for _ in range(iterations):
            self.model.train()
            data = self.memory.get_batch(batch_size, stream_batch_size)
            self.optimizer.zero_grad()

            batch = {
                "img": data[1],
                "cls": data[2],
                "reverse": data[3],
                "img_path": data[4],
            }

            loss = self.model_forward_with_erd(batch)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.update_schedule()
            total_loss += loss.item()

        return total_loss / iterations

    def online_step(self, sample, sample_num, n_worker):
        # self.temp_batchsize = self.batch_size
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
                                               iterations=iteration, stream_batch_size=self.temp_batchsize)
                self.report_training(sample_num, train_loss)
                for stored_sample in self.temp_batch:
                    self.update_memory(stored_sample)
                self.temp_batch = []
                self.num_updates -= iteration

    def online_after_task(self, cur_iter):
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad = False
        logger.info("[ERD] Old model saved for distillation.")
