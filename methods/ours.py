from methods.er_baseline import ER
import torch
from torch import nn
import numpy as np
from utils import autograd_hacks
from operator import attrgetter

class Ours(ER):
    def __init__(self, criterion, n_classes, device, **kwargs):
        super().__init__(criterion=criterion, n_classes=n_classes, device=device, **kwargs)
        
        self.memory = OurDataset(self.args, self.dataset, self.exposed_classes, device=self.device, memory_size=self.memory_size)
        
        # Information based freezing
        self.unfreeze_rate = 0.5
        self.fisher_ema_ratio = 0.001
        self.fisher = [0.0 for _ in range(self.num_blocks)]
        self.last_grad_mean = 0.0
        # autograd_hacks.add_hooks(self.model)  # install once – cheap
        
        self.cumulative_fisher = []
        self.freeze_idx = []
        self.frozen = False

        self.cls_weight_decay = kwargs["cls_weight_decay"]
    
    # def add_new_class(self, class_name):
        # super().add_new_class(class_name)
        
        # autograd_hacks.remove_hooks(self.model)
        # autograd_hacks.add_hooks(self.model)
    
    def _layer_type(self, layer: nn.Module) -> str:
        return layer.__class__.__name__

    def prev_check(self, idx):
        result = True
        for i in range(idx):
            if i not in self.freeze_idx:
                result = False
                break
        return result

    def unfreeze_layers(self):
        self.frozen = False
        for name, param in self.model.named_parameters():
            param.requires_grad = True

    def freeze_layers(self):
        if len(self.freeze_idx) > 0:
            self.frozen = True
        for i in self.freeze_idx:
            self.freeze_layer(i)

    def freeze_layer(self, block_index):
        # blcok(i)가 들어간 layer 모두 freeze
        block_name = self.block_names[block_index]
        # print("freeze", group_name)
        for subblock_name in block_name:
            for name, param in self.model.named_parameters():
                if subblock_name in name:
                    param.requires_grad = False
    
    def online_step(self, sample, sample_num, n_worker):
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])
        self.update_memory(sample)
        self.num_updates += self.online_iter
        
        if self.num_updates >= 1:
            train_loss = self.online_train([], self.batch_size, n_worker, iterations=int(self.num_updates), stream_batch_size=0)
            self.report_training(sample_num, train_loss)
            self.num_updates -= int(self.num_updates)
            self.update_schedule()

    # -----------------------------------------------------------
    # def _apply_freeze_to_optimizer(self):
    #     # rebuild param_groups keeping hyper-params
    #     new_pgs = []
    #     for pg in self.optimizer.param_groups:
    #         params = [p for p in pg['params'] if p.requires_grad]
    #         if params:
    #             pg['params'] = params
    #             new_pgs.append(pg)
    #     self.optimizer.param_groups = new_pgs

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        total_loss = 0.0
        if len(sample) > 0:
            self.memory.register_stream(sample)
        for i in range(iterations):
            self.model.train()
            data = self.memory.get_batch(batch_size, stream_batch_size)

            batch = {
                "img": data[1],         # images
                "cls": data[2],         # labels
                "reverse": data[3],     # reverse tensors
                "img_path": data[4],    # image paths
            }
            
            # print(f"[DEBUG] batch images: {batch['img_path']}")
            # print(f"[DEBUG] batch labels shape: {batch['cls'].shape}")

            self.optimizer.zero_grad()

            loss, loss_item = self.model_forward(batch)
            
            if self.train_count > 2:
                if self.unfreeze_rate < 1.0:
                    self.get_freeze_idx(loss)
                if np.random.rand() > self.unfreeze_rate:
                    self.freeze_layers()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                # with torch.cuda.amp.autocast(self.use_amp):
                    # autograd_hacks.compute_grad1(self.model)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                # autograd_hacks.compute_grad1(self.model)
                self.optimizer.step()
            
            # autograd_hacks.compute_grad1(self.model)
            
            if not self.frozen:
                self.calculate_fisher()
            
            # autograd_hacks.clear_backprops(self.model)
            
            self.update_schedule()

            total_loss += loss.item()
            
            if len(self.freeze_idx) == 0:    
                # forward와 backward가 full로 일어날 때
                self.total_flops += (len(data[1]) * (self.forward_flops + self.backward_flops))
            else:
                self.total_flops += (len(data[1]) * (self.forward_flops + self.get_backward_flops()))
            
            self.unfreeze_layers()
            self.freeze_idx = []
            self.train_count += 1
            
            
        return total_loss / iterations

    def get_backward_flops(self):
        backward_flops = self.backward_flops
        if self.frozen:
            for i in self.freeze_idx:
                backward_flops -= self.blockwise_backward_flops[i]
        return backward_flops

    @torch.no_grad()
    def calculate_fisher(self):
        block_fisher = [0.0 for _ in range(self.num_blocks)]
        for i, block_name in enumerate(self.block_names[:-1]):
            for subblock_name in block_name:
                get_attr = attrgetter(subblock_name)
                block = get_attr(self.model)
                block_grad = []
                block_input = []
                for n, p in block.named_parameters():
                    if p.requires_grad is True and p.grad is not None:
                        if not p.grad.isnan().any():
                            block_fisher[i] += (p.grad.clone().detach().clamp(-1, 1) ** 2).sum().item()
                            if self.unfreeze_rate < 1:
                                self.total_flops += len(p.grad.clone().detach().flatten())*2 / 10e9
        
        for i in range(self.num_blocks):
            if i not in self.freeze_idx or not self.frozen:
                self.fisher[i] += self.fisher_ema_ratio * (block_fisher[i] - self.fisher[i])
        # print(group_fisher)
        self.total_fisher = sum(self.fisher)
        self.cumulative_fisher = [sum(self.fisher[0:i+1]) for i in range(self.num_blocks)]
        # print(self.fisher, self.layerwise_backward_flops)

    def get_flops_parameter(self):
        super().get_flops_parameter()
        self.cumulative_backward_flops = [sum(self.blockwise_backward_flops[0:i+1]) for i in range(self.num_blocks)]
        print("self.cumulative_backward_flops")
        print(self.cumulative_backward_flops)

    @torch.no_grad()
    def get_freeze_idx(self, loss):
        ## FIXME: set param list
        grad = self.get_grad(loss, [p for p in self.model.model[22].parameters()] + [p for p in self.model.model[30].parameters()])
        last_grad = (grad ** 2 ).sum().item()
        if self.unfreeze_rate < 1:
            self.total_flops += len(grad.clone().detach().flatten())/10e9
        batch_freeze_score = last_grad/(self.last_grad_mean+1e-10)
        self.last_grad_mean += self.fisher_ema_ratio * (last_grad - self.last_grad_mean)
        freeze_score = []
        freeze_score.append(1)
        # if 'noinitial' in self.note:
        freeze_score.append(1)
        total_model_flops = self.total_model_flops - self.blockwise_forward_flops[0] - self.blockwise_backward_flops[0]
        cumulative_backward_flops = [sum(self.blockwise_backward_flops[1:i+1]) for i in range(1, self.num_blocks)]
        total_fisher = sum(self.fisher[1:])
        cumulative_fisher = [sum(self.fisher[1:i+1]) for i in range(1, self.num_blocks)]
        
        for i in range(self.num_blocks-1):
            freeze_score.append(total_model_flops / (total_model_flops - cumulative_backward_flops[i]) * (
                        total_fisher - cumulative_fisher[i]) / (total_fisher + 1e-10))
        max_score = max(freeze_score)
        modified_score = []
        modified_score.append(batch_freeze_score)
        modified_score.append(batch_freeze_score)
        for i in range(self.num_blocks -1):
            modified_score.append(batch_freeze_score*(total_fisher - cumulative_fisher[i])/(total_fisher + 1e-10) + cumulative_backward_flops[i]/total_model_flops * max_score)
        optimal_freeze = np.argmax(modified_score)

        
        print(f'I/C: {freeze_score} \n BI/C: {modified_score} \n Grad_Magnitude: {batch_freeze_score}')
        print(f'Iter: {self.train_count} Freeze: {optimal_freeze}')
        self.freeze_idx = list(range(self.num_blocks))[0:optimal_freeze]

    @torch.no_grad()
    def get_grad(self, loss: torch.Tensor, param_list):
        # autograd will return *None* when the param is unused
        grads = torch.autograd.grad(
            loss,
            param_list,
            retain_graph=True,
            allow_unused=True,
        )

        flat = []
        for g, p in zip(grads, param_list):
            flat.append(
                (torch.zeros_like(p) if g is None else g)
                .contiguous()
                .view(-1)
            )
        grad_vec = torch.cat(flat)
        torch.cuda.empty_cache()
        return grad_vec
    

# customized MemoryDataset
import os
from utils.data_loader import MemoryDataset, collate_fn, get_pretrained_statistics
import glob
from PIL import Image
import cv2
class OurDataset(MemoryDataset):
    def __init__(self, args, dataset, cls_list=None, device=None, data_dir=None, memory_size=None):
        super().__init__(args, dataset, cls_list, device, data_dir, memory_size)
        self.alpha = 1.0                  # smoothing constant in 1/(usage+α)
        self.betaa = 1.0
    
    def build_initial_buffer(self):
        n_classes, images_dir, label_path = get_pretrained_statistics(self.dataset)
        if self.dataset == 'VOC_10_10':
            image_files = glob.glob(os.path.join(images_dir, "train2012", "*.jpg")) \
                        + glob.glob(os.path.join(images_dir, "train2007", "*.jpg")) \
                        + glob.glob(os.path.join(images_dir, "val2012", "*.jpg")) \
                        + glob.glob(os.path.join(images_dir, "val2007", "*.jpg"))
        else:
            image_files = glob.glob(os.path.join(images_dir, "train", "*.jpg"))

        indices = np.random.choice(range(len(image_files)), size=self.memory_size, replace=False)

        for idx in indices:
            image_path = image_files[idx]
            split_name = image_path.split('/')[-2]
            base_name = image_path.split('/')[-1]
            self.replace_sample({'file_name': split_name + '/' + base_name, 'label': None}, images_dir=images_dir,label_path=label_path)

        for idx in indices:
            image_path = image_files[idx]
            split_name = image_path.split('/')[-2]
            base_name = image_path.split('/')[-1]
            self.replace_sample(
                {
                    'file_name': split_name + '/' + base_name,
                    'label': None,
                    'usage': 0,
                    'classes': [],   # we fill it in replace_sample() after we know labels
                },
                images_dir=images_dir, label_path=label_path
            )
    
    def replace_sample(self, sample, idx=None, images_dir=None, label_path=None):
        img, labels, image_path, ratio = self.load_data(
            sample['file_name'],
            cls_label=sample['label'],
            image_dir=images_dir or self.image_dir,
            label_path=label_path or self.label_path
        )

        ### BEGIN USAGE
        entry = {
            "img": img,
            "labels": labels,
            "img_path": image_path,
            "ratio": ratio,
            "usage": sample.get("usage", 0),
            "classes": torch.unique(labels[:, 0]).tolist() if len(labels) else []
        }
        ### END USAGE

        if idx is None:
            self.buffer.append(entry)
        else:
            self.buffer[idx] = entry
    
    def get_data(self, idx):
        entry = self.buffer[idx]
        img, labels, img_path = entry["img"], entry["labels"], entry["img_path"]
        valid_mask = labels[:, 0] != -1

        if isinstance(img, Image.Image):
            img = np.array(img)

        h0, w0 = img.shape[:2]
        r = self.image_sizes[0] / max(h0, w0)

        if r != 1:
            new_w = int(w0 * r)
            new_h = int(h0 * r)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
        pad_w = self.image_sizes[0] - new_w
        pad_h = self.image_sizes[1] - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114))

        img = Image.fromarray(img)

        return img, labels[valid_mask], img_path

    @torch.no_grad()
    def get_batch(self, batch_size, stream_batch_size=0, use_weight=None,
                  transform=None, weight_method=None):
        assert batch_size >= stream_batch_size
        stream_batch_size = min(stream_batch_size, len(self.stream_data))
        batch_size = min(batch_size, stream_batch_size + len(self.buffer))
        memory_batch_size = batch_size - stream_batch_size

        data = []
        if stream_batch_size > 0:
            stream_indices = np.random.choice(range(len(self.stream_data)), size=stream_batch_size, replace=False)
            for i in stream_indices:
                img, bboxes, img_path, _ = self.stream_data[i]
                img, bboxes, rev_tensor = self.transform(img, bboxes)
                bboxes[:, [1, 3]] *= self.image_sizes[0]
                bboxes[:, [2, 4]] *= self.image_sizes[1]
                data.append((img, bboxes, rev_tensor, img_path))

        # ───── Memory part ──────────────────────────────────────────────
        if memory_batch_size > 0 and len(self.buffer):

            ### HYBRID WEIGHT BEGIN
            if weight_method == "cls_usage":          # new option
                # 1 / (usage+α)  ×  1 / (mean cls_trained + β)
                alpha = getattr(self, "alpha", 1.0)
                beta  = getattr(self, "beta", 1.0)       # you may set self.beta in __init__
                weights = []
                for entry in self.buffer:
                    u = entry["usage"]
                    # gather per-image class-trained counts
                    if entry["classes"]:
                        t = [self.cls_train_cnt[self.cls_dict[c]]    # safe: cls_dict maps real IDs
                            for c in entry["classes"]               # (skip if not yet in dict)
                            if c in self.cls_dict and
                                self.cls_dict[c] < len(self.cls_train_cnt)]
                        mean_t = np.mean(t) if t else 0.0
                    else:
                        mean_t = 0.0        # no GT boxes → neutral
                    weights.append(1.0 / (u + alpha) * 1.0 / (mean_t + beta))
                w = np.asarray(weights, dtype=np.float64)
                w /= w.sum()
            else:
                # old: purely usage-based
                w = np.array([1.0 / (e["usage"] + self.alpha) for e in self.buffer],
                            dtype=np.float64)
                w /= w.sum()
            ### HYBRID WEIGHT END

            indices = np.random.choice(
                len(self.buffer),
                size=memory_batch_size,
                replace=len(self.buffer) < memory_batch_size,
                p=w,
            )

            for i in indices:
                # update usage counter *and* class-train counts
                self.buffer[i]["usage"] += 1
                for cls in self.buffer[i]["classes"]:
                    idx_cls = self.cls_dict.get(cls, None)
                    if idx_cls is not None and idx_cls < len(self.cls_train_cnt):
                        self.cls_train_cnt[idx_cls] += 1

                img, bboxes, img_path = self.get_data(i)
                img, bboxes, rev_tensor = self.transform(img, bboxes)
                bboxes[:, [1, 3]] *= self.image_sizes[0]
                bboxes[:, [2, 4]] *= self.image_sizes[1]
                data.append((img, bboxes, rev_tensor, img_path))

        return collate_fn(data)