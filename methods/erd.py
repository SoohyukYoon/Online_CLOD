import copy
import logging
import torch
import torch.nn.functional as F
from torch.nn import KLDivLoss
from methods.er_baseline import ER
from utils.data_loader import StreamDataset
import torch.nn as nn

logger = logging.getLogger()

class ERD(ER):
    def __init__(self, criterion, n_classes, device, **kwargs):
        super().__init__(criterion, n_classes, device, **kwargs)
        self.old_model = None
        self.lambda_cls = kwargs.get("lambda_cls", 1.0)
        self.lambda_reg = kwargs.get("lambda_reg", 1.0)
        self.alpha = kwargs.get("alpha", 2.0)
        self.transform = kwargs.get("transform")
        self.cls_list = kwargs.get("cls_list")
        self.data_dir = kwargs.get("data_dir")
        self.dataset = kwargs.get("dataset")

        logger.info(f"[ERD INIT] λ_cls: {self.lambda_cls}, λ_reg: {self.lambda_reg}, α: {self.alpha}")

    def forward_with_erd(self, Sm, Qm, Se, Qe):
        Sm = self.preprocess_batch(Sm)
        loss_model, _ = self.model.model(Sm) ## detection loss

        cls_loss = self.distill_cls_loss(Qm)
        reg_loss = self.distill_bbox_loss(Qe)

        total_loss = loss_model + self.lambda_cls * cls_loss + self.lambda_reg * reg_loss
        return total_loss

    def distill_cls_loss(self, batch):
        if self.old_model is None:
            return torch.tensor(0.0, device=self.device)

        imgs = batch["img"].to(self.device)

        with torch.no_grad():
            _, teacher_output = self.old_model.model(imgs)
        _, student_output = self.model.model(imgs)

        assert "cls" in teacher_output, "Missing 'cls' key in teacher output"
        assert "cls" in student_output, "Missing 'cls' key in student output"

        logits_teacher = teacher_output["cls"]
        logits_student = student_output["cls"]

        confidence = torch.softmax(logits_teacher, dim=-1)
        conf_mean = confidence.mean()
        conf_std = confidence.std()
        threshold = conf_mean + self.alpha * conf_std
        mask = confidence > threshold ## ERS : threshold 이상인 샘플만 선택

        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        loss = F.mse_loss(logits_student[mask], logits_teacher[mask])
        return loss

    def distill_bbox_loss(self, batch):
        if self.old_model is None:
            return torch.tensor(0.0, device=self.device)

        imgs = batch["img"].to(self.device)

        with torch.no_grad():
            _, teacher_output = self.old_model.model(imgs)
        _, student_output = self.model.model(imgs)

        reg_key = "box" if "box" in teacher_output else "reg"
        assert reg_key in teacher_output, f"Missing '{reg_key}' in teacher output"
        assert reg_key in student_output, f"Missing '{reg_key}' in student output"

        reg_teacher = teacher_output[reg_key]
        reg_student = student_output[reg_key]

        top1_t = torch.max(reg_teacher, dim=-1).values
        top1_mean = top1_t.mean()
        top1_std = top1_t.std()
        threshold = top1_mean + self.alpha * top1_std
        mask = top1_t > threshold ## ERS : threshold 이상인 샘플만 선택

        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        loss_fn = nn.KLDivLoss(reduction='batchmean', log_target=True)
        log_s = torch.log_softmax(reg_student[mask], dim=-1)
        log_t = torch.log_softmax(reg_teacher[mask], dim=-1)

        loss = loss_fn(log_s, log_t)
        return loss

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        total_loss = 0.0
        if len(sample) > 0:
            self.memory.register_stream(sample)

        for i in range(iterations):
            self.model.model.train()
            data = self.memory.get_batch(batch_size, stream_batch_size)
            self.optimizer.zero_grad()

            Sm, Qm, Se, Qe = self.get_episode_batches(data)
            loss = self.forward_with_erd(Sm, Qm, Se, Qe)

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

    def online_after_task(self, cur_iter):
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        logger.info("[ERD] Old model saved for distillation.")

    def get_episode_batches(self, raw_samples):
        batch = self.memory.get_batch(batch_size=self.batch_size, stream_batch_size=self.temp_batchsize)

        memory_support = batch ## Sm
        memory_query = batch ## Qm
        episode_support = raw_samples ## Se
        episode_query = raw_samples ## Qe

        return memory_support, memory_query, episode_support, episode_query


# 채은님 ERD

# import copy
# import logging
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from methods.er_baseline import ER

# logger = logging.getLogger()

# class ERD(ER):
#     def __init__(self, criterion, n_classes, device, **kwargs):
#         super().__init__(criterion, n_classes, device, **kwargs)
#         self.old_model = None
#         self.lambda_cls = kwargs.get("lambda_cls", 1.0)
#         self.lambda_reg = kwargs.get("lambda_reg", 1.0)
#         self.alpha_cls = kwargs.get("alpha_cls", kwargs.get("alpha", 0.5))
#         self.alpha_reg = kwargs.get("alpha_reg", kwargs.get("alpha", 0.5))
#         logger.info(f"[ERD INIT] λ_cls: {self.lambda_cls}, λ_reg: {self.lambda_reg}, α_cls: {self.alpha_cls}, α_reg: {self.alpha_reg}")

#     def model_forward_with_erd(self, current_batch, replay_batch_for_cls, replay_batch_for_reg):
#         current_batch_processed = self.preprocess_batch(current_batch)
#         outputs = self.model(current_batch_processed["img"])
#         aux_preds = self.vec2box(outputs["AUX"])
#         main_preds = self.vec2box(outputs["Main"])
#         loss_model, _ = self.model.loss_fn(aux_preds, main_preds, current_batch_processed['cls'])

#         cls_distill_loss = torch.tensor(0.0, device=self.device)
#         if replay_batch_for_cls is not None and len(replay_batch_for_cls.get("img", [])) > 0:
#             cls_distill_loss = self.distill_cls_loss(replay_batch_for_cls)

#         reg_distill_loss = torch.tensor(0.0, device=self.device)
#         if replay_batch_for_reg is not None and len(replay_batch_for_reg.get("img", [])) > 0:
#             reg_distill_loss = self.distill_bbox_loss(replay_batch_for_reg)

#         logger.info(f"loss_model: {loss_model.item():.4f}, cls_loss: {cls_distill_loss.item():.4f}, reg_loss: {reg_distill_loss.item():.4f}")

#         return loss_model + self.lambda_cls * cls_distill_loss + self.lambda_reg * reg_distill_loss

#     def _elastic_response_selection(self, responses, alpha, response_type="cls"):
#         if responses.numel() == 0:
#             return torch.empty(0, dtype=torch.bool, device=responses.device), torch.empty(0, device=responses.device), torch.tensor(float('nan'))
#         mean_val = responses.mean()
#         std_val = max(responses.std(), torch.tensor(1e-6, device=responses.device))
#         threshold = mean_val + alpha * std_val
#         mask = responses > threshold
#         if mask.sum() == 0:
#             return mask, torch.empty_like(responses), threshold
#         return mask, responses[mask], threshold

#     def distill_cls_loss(self, batch):
#         if self.old_model is None:
#             return torch.tensor(0.0, device=self.device)
#         imgs = batch["img"].to(self.device)
#         with torch.no_grad():
#             preds_cls, _, _ = self.vec2box(self.old_model(imgs)["Main"])
#             t_logits = preds_cls
#             t_confidences, _ = torch.softmax(t_logits, dim=-1).max(dim=-1)
#             cls_mask_2d, _, _ = self._elastic_response_selection(t_confidences, self.alpha_cls, "cls")
#             if cls_mask_2d.sum() == 0:
#                 return torch.tensor(0.0, device=self.device)
#             t_logits_selected = t_logits[cls_mask_2d]
#         s_logits = self.vec2box(self.model(imgs)["Main"])[0]
#         s_logits_selected = s_logits[cls_mask_2d]
#         return F.mse_loss(s_logits_selected, t_logits_selected.detach())

#     def distill_bbox_loss(self, batch):
#         if self.old_model is None:
#             return torch.tensor(0.0, device=self.device)
#         imgs = batch["img"].to(self.device)
#         with torch.no_grad():
#             _, _, preds_box = self.vec2box(self.old_model(imgs)["Main"])
#             t_boxes_raw = preds_box
#             if t_boxes_raw.ndim != 3:
#                 return torch.tensor(0.0, device=self.device)
#             t_box_sharpness = t_boxes_raw.mean(dim=-1)
#             reg_mask_2d, _, _ = self._elastic_response_selection(t_box_sharpness, self.alpha_reg, "reg")
#             if reg_mask_2d.sum() == 0:
#                 return torch.tensor(0.0, device=self.device)
#             t_boxes_selected = t_boxes_raw[reg_mask_2d]
#         s_boxes_raw = self.vec2box(self.model(imgs)["Main"])[2]
#         s_boxes_selected = s_boxes_raw[reg_mask_2d]
#         return F.mse_loss(s_boxes_selected, t_boxes_selected.detach()) if s_boxes_selected.shape == t_boxes_selected.shape else torch.tensor(0.0, device=self.device)

#     def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
#         total_loss_avg = 0.0
#         if sample: self.memory.register_stream(sample)
#         if not hasattr(self, 'memory') or self.memory.is_empty(): return 0.0
#         for _ in range(iterations):
#             self.model.train()
#             data = self.memory.get_batch(batch_size, stream_batch_size)
#             current_task_data = {"img": data[1][:stream_batch_size], "cls": data[2][:stream_batch_size]}
#             replay_task_data = {"img": data[1][stream_batch_size:], "cls": data[2][stream_batch_size:]} if batch_size > stream_batch_size else None
#             self.optimizer.zero_grad()
#             loss = self.model_forward_with_erd(current_task_data, replay_task_data, replay_task_data)
#             (self.scaler.scale(loss).backward(), self.scaler.step(self.optimizer), self.scaler.update()) if self.use_amp else (loss.backward(), self.optimizer.step())
#             self.update_schedule()
#             total_loss_avg += loss.item()
#         return total_loss_avg / iterations if iterations > 0 else 0.0

#     def online_step(self, sample, sample_num, n_worker):
#         self.temp_batchsize = self.batch_size
#         if sample['klass'] not in self.exposed_classes:
#             self.add_new_class(sample['klass'])
#             self.online_after_task(sample_num)
#         self.temp_batch.append(sample)
#         self.num_updates += self.online_iter
#         if len(self.temp_batch) == self.temp_batchsize:
#             iteration = int(self.num_updates)
#             if iteration:
#                 stream_batch_size = max(1, self.batch_size // 4)
#                 train_loss = self.online_train(self.temp_batch, self.batch_size, n_worker, iteration, stream_batch_size)
#                 self.report_training(sample_num, train_loss)
#                 for s in self.temp_batch: self.update_memory(s)
#                 self.temp_batch = []
#                 self.num_updates -= iteration

#     def online_after_task(self, cur_iter):
#         self.old_model = copy.deepcopy(self.model)
#         self.old_model.eval()
#         for p in self.old_model.parameters():
#             p.requires_grad = False
#         logger.info("[ERD] Old model saved for distillation.")
