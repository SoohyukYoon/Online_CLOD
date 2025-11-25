import torch
import torch.nn.functional as F
import copy
import logging
from methods.er_baseline import ER

logger = logging.getLogger()

class ERD(ER):
    def __init__(self, n_classes, device, **kwargs):
        super().__init__(n_classes, device, **kwargs)
        self.old_model = None
        self.lambda_cls = kwargs.get("lambda_cls", 0.5)
        self.lambda_reg = kwargs.get("lambda_reg", 0.5)
        self.alpha_cls = kwargs.get("alpha_cls", 2.0)
        self.alpha_reg = kwargs.get("alpha_reg", 2.0)
        logger.info(f"[ERD INIT] λ_cls: {self.lambda_cls}, λ_reg: {self.lambda_reg}, α_cls: {self.alpha_cls}, α_reg: {self.alpha_reg}")
    
    def _get_layer(self, root_model, layer_spec):
        mod = root_model
        for attr in str(layer_spec).split("."):
            if attr.isdigit():
                mod = mod[int(attr)]
            else:
                mod = getattr(mod, attr)
        return mod
    
    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        total_loss = 0.0
        if len(sample) > 0:
            self.memory.register_stream(sample)
        for i in range(iterations):
            self.model.train()
            data = self.memory.get_batch(batch_size, stream_batch_size)
            
            self.optimizer.zero_grad()

            loss, loss_item = self.model_forward_with_erd(data)

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
        
    def model_forward_with_erd(self, batch):
        inps, targets = self.preprocess_batch(batch)

        with torch.cuda.amp.autocast(enabled=False):
            loss_item = self.model(inps, targets)
            loss_model = loss_item["total_loss"]
            
            loss_cls_distill, loss_reg_distill = 0.0, 0.0

            if self.old_model is not None:
                with torch.no_grad():
                    image_tensors = inps.tensors
                    tea_feats = self.old_model.backbone(image_tensors)
                    tea_neck  = self.old_model.neck(tea_feats)
                    teacher_cls_list = [self.old_model.head.gfl_cls[i](tea_neck[i]) for i in range(len(self.old_model.head.gfl_cls))]
                    teacher_reg_list = [self.old_model.head.gfl_reg[i](tea_neck[i]) for i in range(len(self.old_model.head.gfl_reg))]

                stu_feats = self.model.backbone(image_tensors)
                stu_neck  = self.model.neck(stu_feats)
                student_cls_list = [self.model.head.gfl_cls[i](stu_neck[i]) for i in range(len(self.model.head.gfl_cls))]
                student_reg_list = [self.model.head.gfl_reg[i](stu_neck[i]) for i in range(len(self.model.head.gfl_reg))]

                teacher_cls_logits = torch.cat([t.permute(0,2,3,1).reshape(-1, t.shape[1]) for t in teacher_cls_list], dim=0)
                student_cls_logits = torch.cat([t.permute(0,2,3,1).reshape(-1, t.shape[1]) for t in student_cls_list], dim=0)
                teacher_reg_logits = torch.cat([t.permute(0,2,3,1).reshape(-1, t.shape[1]) for t in teacher_reg_list], dim=0)
                student_reg_logits = torch.cat([t.permute(0,2,3,1).reshape(-1, t.shape[1]) for t in student_reg_list], dim=0)
                
                C_old = self.old_model.head.gfl_cls[0].out_channels
                teacher_cls_logits = teacher_cls_logits[..., :C_old]
                student_cls_logits = student_cls_logits[..., :C_old]

                mask_cls, old_cls_sel, new_cls_sel = self._elastic_response_selection(
                    teacher_cls_logits, student_cls_logits, alpha=self.alpha_cls
                )
                loss_cls_distill = self.distill_cls_loss(old_cls_sel, new_cls_sel, mask_cls)

                mask_reg, old_reg_sel, new_reg_sel = self._elastic_response_selection(
                    teacher_reg_logits, student_reg_logits, alpha=self.alpha_reg
                )
                loss_reg_distill = self.distill_bbox_loss(old_reg_sel, new_reg_sel, mask_reg)

        logger.info(f"loss_model: {loss_model.item():.4f}, cls_loss: {loss_cls_distill:.4f}, reg_loss: {loss_reg_distill:.4f}")
        total_loss = loss_model + self.lambda_cls * loss_cls_distill + self.lambda_reg * loss_reg_distill
        
        return total_loss, loss_item

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
        # t = 2.0
        # teacher_probs = F.softmax(teacher_logits / t, dim=-1)
        # student_probs = F.softmax(student_logits / t, dim=-1)
        return F.mse_loss(student_logits, teacher_logits)

    def distill_bbox_loss(self, teacher_dist, student_dist, mask):
        if teacher_dist.numel() == 0:
            return torch.tensor(0.0, device=teacher_dist.device)
        teacher_soft = F.softmax(teacher_dist, dim=-1)
        student_soft = F.log_softmax(student_dist, dim=-1)
        return F.kl_div(student_soft, teacher_soft, reduction='batchmean')

    def online_step(self, sample, sample_num, n_worker):
        # self.temp_batchsize = self.batch_size
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
