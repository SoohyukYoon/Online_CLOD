import copy
import logging
import torch
import torch.nn.functional as F
from methods.er_baseline import ER
logger = logging.getLogger()

class LWF_Logit(ER):
    def __init__(self, n_classes, device, **kwargs):
        super().__init__(n_classes, device, **kwargs)
        self.old_model = None
        self.lambda_old = 0.1#kwargs.get("lambda_old", 0.1)
        logger.info(f"[LWF INIT] lambda_old: {self.lambda_old}")
        
    def model_forward(self, batch):
        inps, targets = self.preprocess_batch(batch)
        drop_bg=True

        with torch.cuda.amp.autocast(enabled=False):
            image_tensors = inps.tensors
            new_feats_b = self.model.backbone(image_tensors)
            new_feats_n = self.model.neck(new_feats_b)
            
            loss_item = self.model.head.forward_train(new_feats_n, labels=targets)
            loss_new = loss_item["total_loss"]

            _, cls_scores_without_sigmoid, bbox_before_softmax, _, pos_inds = self.model.head.get_head_outputs(
                new_feats_n, labels=targets, drop_bg=drop_bg
            )

            loss_distill = image_tensors.new_tensor(0.0)
            
            if getattr(self, "old_model", None) is not None:
                with torch.no_grad():
                    old_feats_b = self.old_model.backbone(image_tensors)
                    old_feats_n = self.old_model.neck(old_feats_b)
                    _, cls_scores_without_sigmoid_old, bbox_before_softmax_old, _, _ = self.old_model.head.get_head_outputs(
                        old_feats_n, labels=targets, drop_bg=drop_bg, pos_inds=pos_inds
                    )
                loss_distill = F.mse_loss(cls_scores_without_sigmoid[:,:cls_scores_without_sigmoid_old.shape[1]], cls_scores_without_sigmoid_old) \
                    + F.mse_loss(bbox_before_softmax, bbox_before_softmax_old)
                
            else:
                logger.info("[LWF DEBUG] No old model yet")
            
            total_loss = loss_new + self.lambda_old * loss_distill
            self.total_flops += (len(targets) * self.forward_flops * 2)
            # print(f"loss_new : {loss_new}")
            # print(f"loss_distill : {loss_distill}")
            
            return total_loss, loss_item

    def online_step(self, sample, sample_num, n_worker):
        if sample.get('klass',None) and sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])
            self.online_after_task(sample_num)
        elif sample.get('domain',None) and sample['domain'] not in self.exposed_domains:
            self.online_after_task(sample_num)
            self.exposed_domains.append(sample['domain'])
            
        self.temp_batch.append(sample)
        self.num_updates += self.online_iter

        if len(self.temp_batch) == self.temp_batchsize:
            iteration = int(self.num_updates)
            if iteration != 0:
                train_loss = self.online_train(
                    self.temp_batch,
                    self.batch_size,
                    n_worker,
                    iterations=iteration,
                    stream_batch_size=self.temp_batchsize
                )
                self.report_training(sample_num, train_loss)

                for stored_sample in self.temp_batch:
                    self.update_memory(stored_sample)

                self.temp_batch = []
                self.num_updates -= iteration

    def online_after_task(self, cur_iter):
        self.old_model = copy.deepcopy(self.model)
        self.old_model.train()
        logger.info("[LWF] Saved old model after task.")