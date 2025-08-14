## 결과 확인하고 mixup 및 mosaic 시각화

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import random
import os
from copy import deepcopy
from methods.er_baseline import ER

from yolo.utils.bounding_box_utils import transform_bbox
import pdb

class ABR(ER):
    def __init__(self, criterion, n_classes, device, **kwargs):
        super().__init__(criterion, n_classes, device, **kwargs)
        self.abr_weight = kwargs.get("abr_weight", 0.5)
        self.pad_weight = kwargs.get("pad_weight", 1.0)
        self.afd_weight = kwargs.get("afd_weight", 1.0)
        self.ic_weight  = kwargs.get("ic_weight", 1.0)
        self.id_weight  = kwargs.get("id_weight", 1.0)
        self.abr_alpha  = kwargs.get("abr_alpha", 1.0)
        
        teacher_model = kwargs.get("teacher_model", None)
        if teacher_model is None:
            self.teacher = None
        else:
            self.teacher = deepcopy(teacher_model).eval().to(device)
            for p in self.teacher.parameters():
                p.requires_grad_(False)
                
        self._distill_layers = kwargs.get("distill_layers",
                      kwargs.get("feature_layers", [15, 18, 21]))

        self.student_feats: list[torch.Tensor] = []
        self.teacher_feats: list[torch.Tensor] = []
        if self.teacher is not None:
            self._register_hooks()
            
        self.boxes_index = list(range(len(self.memory.buffer)))
        self.bg_size = 0
        
    def compute_overlap(self, a, b):
        area = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
        iw = np.minimum(a[2], b[2]) - np.maximum(a[0], b[0]) + 1
        ih = np.minimum(a[3], b[3]) - np.maximum(a[1], b[1]) + 1
        iw = np.maximum(iw, 0)
        ih = np.maximum(ih, 0)
        aa = (a[2] - a[0] + 1)*(a[3] - a[1]+1)
        ba = area
        intersection = iw*ih
        if intersection/aa > 0.3 or intersection/ba > 0.3:
            return intersection/ba, True
        else:
            return intersection/ba, False

    def _sample_per_bbox_from_boxrehearsal(self, i, im_shape):
        if len(self.memory.buffer) == 0:
            raise RuntimeError("Memory buffer is empty.")
        sample = self.memory.buffer[self.boxes_index[i]]
        img_tensor, label_tensor = sample[0], sample[1]

        if isinstance(img_tensor, Image.Image):
            box_im = img_tensor.convert("RGB")
        else:
            box_im = Image.fromarray((img_tensor.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).convert("RGB")
        
        cls_name = int(label_tensor[0,0])
        bboxes = [0, 0, box_im.size[0], box_im.size[1]]
        box_o_h, box_o_w = box_im.size[1], box_im.size[0]
        gt_classes = cls_name

        im_mean_size = np.mean(im_shape)
        box_mean_size = np.mean(np.array([int(bboxes[2]), int(bboxes[3])]))
        if float(box_mean_size) >= float(im_mean_size*0.2) and float(box_mean_size) <= float(im_mean_size*0.7):
            box_scale = 1.0
        else:
            box_scale = random.uniform(float(im_mean_size*0.4), float(im_mean_size*0.6)) / float(box_mean_size)
        
        box_im = box_im.resize((int(box_scale * box_o_w), int(box_scale * box_o_h)))
        gt_boxes = [0, 0, box_im.size[0], box_im.size[1], gt_classes]
        return box_im, np.array([gt_boxes]), self.boxes_index[i]

    def _start_mixup(self, image, targets, alpha=2.0, beta=5.0):
        image = np.array(image)
        img_shape = image.shape
        if not isinstance(targets, np.ndarray):
            gts = []
            for box in targets:
                gts.append(box.tolist())
            gts = np.array(gts)
        else:
            gts = targets

        _MIXUP=True
        if gts.shape[0] == 1:
            img_w = gts[0][2]-gts[0][0]
            img_h = gts[0][3]-gts[0][1]
            if (img_shape[1]-img_w)<(img_shape[1]*0.25) and (img_shape[0]-img_h)<(img_shape[0]*0.25):
                _MIXUP=False

        if _MIXUP:
            Lambda = torch.distributions.beta.Beta(alpha, beta).sample().item()
            num_mixup = 3
            if len(self.boxes_index) < self.batch_size:
                self.boxes_index = list(range(len(self.memory.buffer)))

            mixup_count = 0
            for i in range(num_mixup):
                c_img, c_gt, b_id = self._sample_per_bbox_from_boxrehearsal(i, img_shape)
                c_img = np.asarray(c_img)
                _c_gt = c_gt.copy()
                pos_x = random.randint(0, int(img_shape[1] * 0.6))
                pos_y = random.randint(0, int(img_shape[0] * 0.4))
                new_gt = [c_gt[0][0] + pos_x, c_gt[0][1] + pos_y, c_gt[0][2] + pos_x, c_gt[0][3] + pos_y]

                restart = True
                overlap = False
                max_iter = 0
                while restart:
                    for g in gts:
                        _, overlap = self.compute_overlap(g, new_gt)
                        if max_iter >= 20:
                            restart = False
                        elif max_iter < 10 and overlap:
                            pos_x = random.randint(0, int(img_shape[1] * 0.6))
                            pos_y = random.randint(0, int(img_shape[0] * 0.4))
                            new_gt = [c_gt[0][0] + pos_x, c_gt[0][1] + pos_y, c_gt[0][2] + pos_x, c_gt[0][3] + pos_y]
                            max_iter += 1
                            restart = True
                            break
                        elif 20 > max_iter >= 10 and overlap:
                            pos_x = random.randint(int(img_shape[1] * 0.4), img_shape[1])
                            pos_y = random.randint(int(img_shape[0] * 0.6), img_shape[0])
                            new_gt = [pos_x-(c_gt[0][2]-c_gt[0][0]), pos_y-(c_gt[0][3]-c_gt[0][1]), pos_x, pos_y]
                            max_iter += 1
                            restart = True
                            break
                        else:
                            restart = False

                if max_iter < 20:
                    a,b,c,d = 0,0,0,0
                    if new_gt[3] >= img_shape[0]:
                        a = new_gt[3]-img_shape[0]; new_gt[3]=img_shape[0]
                    if new_gt[2] >= img_shape[1]:
                        b = new_gt[2]-img_shape[1]; new_gt[2]=img_shape[1]
                    if new_gt[0] < 0:
                        c = -new_gt[0]; new_gt[0]=0
                    if new_gt[1] < 0:
                        d = -new_gt[1]; new_gt[1]=0
                    img1 = Lambda*image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]]
                    c_img = (1-Lambda)*c_img
                    image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = img1 + c_img[d or None: -a or None, c or None: -b or None]
                    _c_gt[0][:-1] = new_gt
                    gts = np.insert(gts, 0, values=_c_gt, axis=0)
                    if b_id in self.boxes_index:
                        self.boxes_index.remove(b_id)
                mixup_count += 1
                if mixup_count>=2: break

        Current_image = Image.fromarray(np.uint8(image))
        Current_target = torch.tensor(gts, dtype=torch.float32)
        return Current_image, Current_target

    def _start_boxes_mosaic(self, s_imgs=[], targets=[], num_boxes=4):
        gt4 = []
        id = []
        scale = int(np.mean(s_imgs.size))
        s_w = scale
        s_h = scale
        yc = int(random.uniform(s_h*0.4, s_h*0.6))
        xc = int(random.uniform(s_w*0.4, s_w*0.6))
        if len(self.boxes_index) < self.batch_size:
            self.boxes_index = list(range(len(self.memory.buffer)))
        imgs = []
        for i in range(num_boxes):
            img, target, b_id = self._sample_per_bbox_from_boxrehearsal(i, s_imgs.size)
            imgs.append(img)
            targets.append(target)
            id.append(b_id)
        img4 = np.full((s_h, s_w, 3), 114., dtype=np.float32)
        for i, (img, target, b_id) in enumerate(zip(imgs, targets, id)):
            (w, h) = img.size
            if i%4==0:
                xc_, yc_ = xc+self.bg_size, yc-self.bg_size
                x1a,y1a,x2a,y2a = xc_, max(yc_-h,0), min(xc_+w,s_w), yc_
                x1b,y1b,x2b,y2b = 0,h-(y2a-y1a),min(w,x2a-x1a),h
            elif i%4==1:
                xc_, yc_ = xc-self.bg_size, yc+self.bg_size
                x1a,y1a,x2a,y2a = max(xc_-w,0), yc_, xc_, min(s_h,yc_+h)
                x1b,y1b,x2b,y2b = w-(x2a-x1a),0,max(xc_,w),min(y2a-y1a,h)
            elif i%4==2:
                xc_, yc_ = xc+self.bg_size, yc+self.bg_size
                x1a,y1a,x2a,y2a = xc_, yc_, min(xc_+w,s_w), min(s_h,yc_+h)
                x1b,y1b,x2b,y2b = 0,0,min(w,x2a-x1a),min(y2a-y1a,h)
            else:
                xc_, yc_ = xc-self.bg_size, yc-self.bg_size
                x1a,y1a,x2a,y2a = max(xc_-w,0), max(yc_-h,0), xc_, yc_
                x1b,y1b,x2b,y2b = w-(x2a-x1a),h-(y2a-y1a),w,h
            img4[y1a:y2a, x1a:x2a] = np.asarray(img)[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b
            gts = target
            gts[:, 0] += padw
            gts[:, 1] += padh
            gts[:, 2] += padw
            gts[:, 3] += padh
            gt4.append(gts)
            if b_id in self.boxes_index:
                self.boxes_index.remove(b_id)
        if len(gt4):
            gt4 = np.concatenate(gt4, 0)
            np.clip(gt4[:, 0], 0, s_w, out=gt4[:, 0])
            np.clip(gt4[:, 2], 0, s_w, out=gt4[:, 2])
            np.clip(gt4[:, 1], 0, s_h, out=gt4[:, 1])
            np.clip(gt4[:, 3], 0, s_h, out=gt4[:, 3])
        Current_image = Image.fromarray(np.uint8(img4))
        Current_target = torch.tensor(gt4, dtype=torch.float32)
        return Current_image, Current_target

    def transform_current_data_with_ABR(self, img=None, target=None):
        if random.randint(0, 1) == 0:
            img, tgt = self._start_mixup(img, target)
        else:
            img, tgt = self._start_boxes_mosaic(img, [], num_boxes=4)

        if tgt.ndim == 2 and tgt.shape[1] == 5:       # [x1,y1,x2,y2,cls] → [cls,x1,y1,x2,y2]
            tgt = tgt[:, [4, 0, 1, 2, 3]]
        return img, tgt

    def _register_hooks(self):
        # 먼저 이전 hook 제거
        if hasattr(self, "_hook_handles"):
            for h in self._hook_handles:
                h.remove()
        self._hook_handles = []

        modules_s = list(self.model.modules())
        modules_t = list(self.teacher.modules()) if self.teacher else []
        named_s   = dict(self.model.named_modules())
        named_t   = dict(self.teacher.named_modules()) if self.teacher else {}

        for l in self._distill_layers:
            # student
            layer_s = modules_s[l] if isinstance(l, int) else named_s[l]
            self._hook_handles.append(
                layer_s.register_forward_hook(self._hook_student)
            )
            # teacher
            if self.teacher is None:
                continue
            layer_t = modules_t[l] if isinstance(l, int) else named_t[l]
            self._hook_handles.append(
                layer_t.register_forward_hook(self._hook_teacher)
            )

    def _hook_student(self, _m, _i, o):
        self.student_feats.append(o)

    def _hook_teacher(self, _m, _i, o):
        self.teacher_feats.append(o)

    @staticmethod
    def _spatial_attention(F_map: torch.Tensor, p: int = 2) -> torch.Tensor:
        """F_map[B,C,H,W] → [B,1,H,W] (ℓ_p‑norm‑based importance)"""
        return F_map.abs().pow(p).sum(dim=1, keepdim=True)

    def _pad_loss(self):
        if not self.student_feats:
            return torch.tensor(0., device=self.device)
        losses = [F.l1_loss(self._spatial_attention(s),
                            self._spatial_attention(t))
                  for s, t in zip(self.student_feats, self.teacher_feats)]
        return torch.stack(losses).mean()
    
    def _afd_loss(self):
        if not self.student_feats:
            return torch.tensor(0., device=self.device)
        losses = []
        for s, t in zip(self.student_feats, self.teacher_feats):
            with torch.no_grad():
                a_t = self._spatial_attention(t)
            losses.append(((s - t).pow(2) * a_t).mean())
        return torch.stack(losses).mean()

    def _inclusive_cls_loss(self, s_logits, gt_labels):
        B, N, _ = s_logits.shape
        total, cnt = 0., 0
        for b in range(B):
            mask = gt_labels[b] >= 0
            if mask.any():
                total += F.cross_entropy(s_logits[b][mask], gt_labels[b][mask], reduction="sum")
                cnt += mask.sum().item()
        return torch.tensor(0., device=self.device) if cnt == 0 else total / cnt

    def _inclusive_kd_loss(self, s_logits, t_logits, T=1.0):
        return F.kl_div(F.log_softmax(s_logits / T, dim=-1),
                        F.softmax(t_logits / T, dim=-1),
                        reduction="batchmean") * (T ** 2)

    def model_forward_with_abr(self, batch):
        self.student_feats.clear()
        self.teacher_feats.clear()

        batch = self.preprocess_batch(batch)
        imgs, labels = batch["img"], batch["cls"]

        if self.teacher is not None:
            with torch.no_grad():
                t_out = self.teacher(imgs)
        else:
            t_out = {}

        s_out = self.model(imgs)

        aux_preds  = self.vec2box(s_out["AUX"])
        main_preds = self.vec2box(s_out["Main"])

        num_known = self.model.num_classes
        invalid   = (labels[..., 0] < 0) | (labels[..., 0] >= num_known)
        labels[..., 0] = torch.where(invalid,
                                     torch.tensor(0, device=labels.device),
                                     labels[..., 0])
        det_loss, _ = self.model.loss_fn(aux_preds, main_preds, labels)

        matcher = self.model.loss_fn.loss.matcher
        with torch.no_grad():
            aligned = matcher(labels, (main_preds[0].detach(), main_preds[2].detach()))[0]
        gt_boxes   = aligned[..., -4:]
        pred_boxes = main_preds[2]
        conf       = torch.ones_like(pred_boxes[..., :1])
        ref_loss   = F.smooth_l1_loss(pred_boxes, gt_boxes, reduction="none")
        ref_loss   = (torch.exp(-self.abr_alpha * (1 - conf)) * ref_loss).mean()

        pad_loss = self._pad_loss() if self.teacher else torch.tensor(0., device=self.device)
        afd_loss = self._afd_loss() if self.teacher else torch.tensor(0., device=self.device)

        if self.teacher and "cls_logits" in s_out and "cls_logits" in t_out:
            s_logit = s_out["cls_logits"]
            t_logit = t_out["cls_logits"]
            anchor_labels = labels[..., 0].long()
            ic_loss = self._inclusive_cls_loss(s_logit, anchor_labels)
            id_loss = self._inclusive_kd_loss(s_logit, t_logit)
        else:
            ic_loss = torch.tensor(0., device=self.device)
            id_loss = torch.tensor(0., device=self.device)

        print(f"ref_loss:", ref_loss)
        print(f"pad_loss:", pad_loss)
        print(f"afd_loss:", afd_loss)
        print(f"ic_loss:", ic_loss)
        print(f"id_loss:", id_loss)
        
        total_loss = (
            det_loss
            + self.abr_weight * ref_loss
            + self.pad_weight * pad_loss
            + self.afd_weight * afd_loss
            + self.ic_weight  * ic_loss
            + self.id_weight  * id_loss
        )
        return total_loss

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        total_loss = 0.0
        if len(sample) > 0:
            self.memory.register_stream(sample)

        for _ in range(iterations):
            self.model.train()
            data = self.memory.get_batch(batch_size, stream_batch_size)
            self.optimizer.zero_grad()

            new_imgs, new_labels = [], []
            for i in range(len(data[1])):
                aug_img, aug_lbl = self.transform_current_data_with_ABR(
                    Image.fromarray((data[1][i].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)),
                    data[2][i].cpu().numpy()
                )
                new_imgs.append(torch.from_numpy(np.array(aug_img)).permute(2,0,1).float() / 255.0)
                new_labels.append(torch.tensor(aug_lbl, dtype=torch.float32))
            
            aug_imgs = torch.stack(new_imgs).to(self.device)

            max_objects = max(lbl.shape[0] for lbl in new_labels)
            padded_labels = []
            for lbl in new_labels:
                if lbl.shape[0] < max_objects:
                    pad = torch.zeros((max_objects - lbl.shape[0], lbl.shape[1]), dtype=lbl.dtype)
                    pad[:, 0] = -1  # 무효 라벨로 패딩
                    lbl = torch.cat([lbl, pad], dim=0)
                # 현재 모델 클래스 범위 초과 시 -1로 (무효 라벨)
                num_known = self.model.num_classes
                lbl[:, 0] = torch.where(
                    (lbl[:, 0] >= 0) & (lbl[:, 0] < num_known),
                    lbl[:, 0],
                    torch.tensor(-1, device=lbl.device)
                )
                padded_labels.append(lbl.unsqueeze(0))
            aug_labels = torch.cat(padded_labels, dim=0).to(self.device)  # [B, T, 5]

            batch = {
                "img": aug_imgs,
                "cls": aug_labels,
                "reverse": data[3],
                "img_path": data[4],
            }

            loss = self.model_forward_with_abr(batch)
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
        if sample.get('klass', None) and sample['klass'] not in self.exposed_classes:
            self.online_after_task(sample_num)
            self.add_new_class(sample['klass'])
        elif sample.get('domain', None) and sample['domain'] not in self.exposed_domains:
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
                    
    def update_memory(self, sample):
        super().update_memory(sample)
        self.boxes_index = list(range(len(self.memory.buffer)))
        
    def online_after_task(self, cur_iter):
        if self.teacher is None:
            self.teacher = deepcopy(self.model).eval().to(self.device)
            for p in self.teacher.parameters():
                p.requires_grad_(False)
        else:
            src_sd = self.model.state_dict()
            dst_sd = self.teacher.state_dict()
            compatible = {
                k: v for k, v in src_sd.items()
                if k in dst_sd and v.shape == dst_sd[k].shape
            }
            missing = [k for k in dst_sd if k not in compatible]
            if missing:
                print(f"[ABR] teacher 업데이트: {len(compatible)}개 복사, "
                      f"{len(missing)}개(shape mismatch) 유지")
            self.teacher.load_state_dict(compatible, strict=False)

        self.teacher.eval()

        self._register_hooks()
        self.boxes_index = list(range(len(self.memory.buffer)))
        torch.cuda.empty_cache()