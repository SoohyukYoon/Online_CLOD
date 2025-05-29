import copy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.er_baseline import ER

logger = logging.getLogger()

class ERD(ER):
    def __init__(self, criterion, n_classes, device, **kwargs):
        super().__init__(criterion, n_classes, device, **kwargs)
        self.old_model = None
        self.lambda_cls = kwargs.get("lambda_cls", 1.0)
        self.lambda_reg = kwargs.get("lambda_reg", 1.0)
        self.alpha_cls = kwargs.get("alpha_cls", kwargs.get("alpha", 0.5))
        self.alpha_reg = kwargs.get("alpha_reg", kwargs.get("alpha", 0.5))
        logger.info(f"[ERD INIT] λ_cls: {self.lambda_cls}, λ_reg: {self.lambda_reg}, α_cls: {self.alpha_cls}, α_reg: {self.alpha_reg}")

    def model_forward_with_erd(self, current_batch, replay_batch_for_cls, replay_batch_for_reg):
        current_batch_processed = self.preprocess_batch(current_batch)
        outputs = self.model(current_batch_processed["img"])
        aux_preds = self.vec2box(outputs["AUX"])
        main_preds = self.vec2box(outputs["Main"])
        loss_model, _ = self.model.loss_fn(aux_preds, main_preds, current_batch_processed['cls'])

        cls_distill_loss = torch.tensor(0.0, device=self.device)
        if replay_batch_for_cls is not None and len(replay_batch_for_cls.get("img", [])) > 0:
            cls_distill_loss = self.distill_cls_loss(replay_batch_for_cls)

        reg_distill_loss = torch.tensor(0.0, device=self.device)
        if replay_batch_for_reg is not None and len(replay_batch_for_reg.get("img", [])) > 0:
            reg_distill_loss = self.distill_bbox_loss(replay_batch_for_reg)

        logger.info(f"loss_model: {loss_model.item():.4f}, cls_loss: {cls_distill_loss.item():.4f}, reg_loss: {reg_distill_loss.item():.4f}")

        return loss_model + self.lambda_cls * cls_distill_loss + self.lambda_reg * reg_distill_loss

    def _elastic_response_selection(self, responses, alpha, response_type="cls"):
        if responses.numel() == 0:
            return torch.empty(0, dtype=torch.bool, device=responses.device), torch.empty(0, device=responses.device), torch.tensor(float('nan'))
        mean_val = responses.mean()
        std_val = max(responses.std(), torch.tensor(1e-6, device=responses.device))
        threshold = mean_val + alpha * std_val
        mask = responses > threshold
        if mask.sum() == 0:
            return mask, torch.empty_like(responses), threshold
        return mask, responses[mask], threshold

    def distill_cls_loss(self, batch):
        if self.old_model is None:
            return torch.tensor(0.0, device=self.device)
        imgs = batch["img"].to(self.device)
        with torch.no_grad():
            preds_cls, _, _ = self.vec2box(self.old_model(imgs)["Main"])
            t_logits = preds_cls
            t_confidences, _ = torch.softmax(t_logits, dim=-1).max(dim=-1)
            cls_mask_2d, _, _ = self._elastic_response_selection(t_confidences, self.alpha_cls, "cls")
            if cls_mask_2d.sum() == 0:
                return torch.tensor(0.0, device=self.device)
            t_logits_selected = t_logits[cls_mask_2d]
        s_logits = self.vec2box(self.model(imgs)["Main"])[0]
        s_logits_selected = s_logits[cls_mask_2d]
        return F.mse_loss(s_logits_selected, t_logits_selected.detach())

    def distill_bbox_loss(self, batch):
        if self.old_model is None:
            return torch.tensor(0.0, device=self.device)
        imgs = batch["img"].to(self.device)
        with torch.no_grad():
            _, _, preds_box = self.vec2box(self.old_model(imgs)["Main"])
            t_boxes_raw = preds_box
            if t_boxes_raw.ndim != 3:
                return torch.tensor(0.0, device=self.device)
            t_box_sharpness = t_boxes_raw.mean(dim=-1)
            reg_mask_2d, _, _ = self._elastic_response_selection(t_box_sharpness, self.alpha_reg, "reg")
            if reg_mask_2d.sum() == 0:
                return torch.tensor(0.0, device=self.device)
            t_boxes_selected = t_boxes_raw[reg_mask_2d]
        s_boxes_raw = self.vec2box(self.model(imgs)["Main"])[2]
        s_boxes_selected = s_boxes_raw[reg_mask_2d]
        return F.mse_loss(s_boxes_selected, t_boxes_selected.detach()) if s_boxes_selected.shape == t_boxes_selected.shape else torch.tensor(0.0, device=self.device)

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        total_loss_avg = 0.0
        if sample: self.memory.register_stream(sample)
        if not hasattr(self, 'memory') or self.memory.is_empty(): return 0.0
        for _ in range(iterations):
            self.model.train()
            data = self.memory.get_batch(batch_size, stream_batch_size)
            current_task_data = {"img": data[1][:stream_batch_size], "cls": data[2][:stream_batch_size]}
            replay_task_data = {"img": data[1][stream_batch_size:], "cls": data[2][stream_batch_size:]} if batch_size > stream_batch_size else None
            self.optimizer.zero_grad()
            loss = self.model_forward_with_erd(current_task_data, replay_task_data, replay_task_data)
            (self.scaler.scale(loss).backward(), self.scaler.step(self.optimizer), self.scaler.update()) if self.use_amp else (loss.backward(), self.optimizer.step())
            self.update_schedule()
            total_loss_avg += loss.item()
        return total_loss_avg / iterations if iterations > 0 else 0.0

    def online_step(self, sample, sample_num, n_worker):
        self.temp_batchsize = self.batch_size
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])
            self.online_after_task(sample_num)
        self.temp_batch.append(sample)
        self.num_updates += self.online_iter
        if len(self.temp_batch) == self.temp_batchsize:
            iteration = int(self.num_updates)
            if iteration:
                stream_batch_size = max(1, self.batch_size // 4)
                train_loss = self.online_train(self.temp_batch, self.batch_size, n_worker, iteration, stream_batch_size)
                self.report_training(sample_num, train_loss)
                for s in self.temp_batch: self.update_memory(s)
                self.temp_batch = []
                self.num_updates -= iteration

    def online_after_task(self, cur_iter):
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad = False
        logger.info("[ERD] Old model saved for distillation.")
