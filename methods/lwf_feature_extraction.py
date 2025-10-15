import logging
import torch
import torch.nn.functional as F
from methods.er_baseline import ER
from methods.lwf_logit import LWF_Logit
logger = logging.getLogger()


class LWF_Feature(LWF_Logit):
    def __init__(self, n_classes, device, **kwargs):
        super().__init__(n_classes, device, **kwargs)
        
        self.lambda_old = 0.1#kwargs.get("lambda_old", 0.1)

    def model_forward(self, batch):
        inps, targets = self.preprocess_batch(batch)
        
        with torch.cuda.amp.autocast(enabled=False):
            image_tensors = inps.tensors
            new_feats_b = self.model.backbone(image_tensors)
            new_feats_n = self.model.neck(new_feats_b)
            
            loss_item = self.model.head.forward_train(new_feats_n, labels=targets)
            loss_new = loss_item["total_loss"]

            loss_distill = 0.0
            if getattr(self, "old_model", None) is not None:
                with torch.no_grad():
                    old_feats_b = self.old_model.backbone(image_tensors)
                    old_feats_n = self.old_model.neck(old_feats_b)

                for new_feat, old_feat in zip(new_feats_n, old_feats_n):
                    loss_distill += self.distillation_loss(new_feat, old_feat) / len(new_feats_n)
            else:
                logger.info("[LWF DEBUG] No old model yet")

        total_loss = loss_new + self.lambda_old * loss_distill
        self.total_flops += (len(targets) * self.forward_flops * 2)
        # print(f"loss_new : {loss_new}")
        # print(f"loss_distill : {loss_distill}")
            
        return total_loss, loss_item
    
    def distillation_loss(self, new_feats, old_feats):
        return F.mse_loss(new_feats, old_feats)