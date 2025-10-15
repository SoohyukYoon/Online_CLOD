import logging
import copy
import numpy as np
import torch
from methods.er_baseline import ER
from utils.data_loader import SelectionClsBalancedDataset
import torch.nn.functional as F


logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")

class SampleSelection(ER):
    def __init__(self, n_classes, device, **kwargs):
        super().__init__(n_classes, device, **kwargs)
    
    def initialize_memory_buffer(self, memory_size):
        self.memory_size = memory_size - self.temp_batchsize
        self.new_exposed_classes = ['pretrained']
        self.lambda_ = 0.5
        self.use_hardlabel = True
        
        data_args = self.damo_cfg.get_data(self.damo_cfg.dataset.train_ann[0])
        self.memory = SelectionClsBalancedDataset(ann_file=data_args['args']['ann_file'], root=data_args['args']['root'], transforms=None,class_names=self.damo_cfg.dataset.class_names,
            dataset=self.dataset, cls_list=self.exposed_classes, device=self.device, memory_size=self.memory_size, image_size=self.img_size, aug=self.damo_cfg.train.augment, selection_method=self.selection_method, priority_selection=self.priority_selection)

        buffer_initial_info = self.cal_initial_info()
        assert len(self.memory.buffer) == len(buffer_initial_info), "Buffer size and initial info size mismatch."
        self.memory.update_initialinfo(buffer_initial_info)
        

    def cal_initial_info(self):
        """
        Calculate the initial information for each sample in the buffer.
        This is a placeholder function that should be implemented based on the selection method.
        """
        # buffer_initial_data2 = self.memory.get_buffer_data()
        buffer_initial_info = []
        # self.model.eval()
        # with torch.no_grad():
        for i in range(0,len(self.memory.buffer), self.batch_size):
            batch = self.memory.get_buffer_data(i, self.batch_size)
            # batch = buffer_initial_data2[i:i+self.batch_size]
            # data = collate_fn(batch)
            # print("data", len(batch), self.batch_size)
            # batch = {
            #     "img": data[0],         # images
            #     "cls": data[1],         # labels
            #     "img_id": data[2],    # image paths
            # }
            self.optimizer.zero_grad()
            inps, targets = self.preprocess_batch(batch)
            with torch.cuda.amp.autocast(enabled=False):
                inputs = inps.tensors
                infos = []
                loss = None
                if "loss" in self.selection_method:
                    for ind in range(len(targets)):
                        loss_item = self.model(inputs[ind], [targets[ind]])["total_loss"]
                        infos.append(loss_item.detach().cpu().item())
                        self.optimizer.zero_grad()

                elif "entropy" in self.selection_method:
                    for ind in range(len(targets)):
                        loss, sample_logit = self.model(inputs[ind], [targets[ind]], get_features=True)
                        sample_logit = sample_logit[-1]
                        probs = F.softmax(sample_logit, dim=0)
                        info = -torch.sum(probs * torch.log(probs + 1e-8)).item()
                        infos.append(info)
                        self.optimizer.zero_grad()
                
                elif "fisher" in self.selection_method:
                    for ind in range(len(targets)):
                        loss = self.model(inputs[ind],[targets[ind]])["total_loss"]
                        for n, p in self.model.neck.named_parameters():
                            if p.requires_grad == True:
                                grad = torch.autograd.grad(loss, p, retain_graph=True)[0].clone().detach().clamp(-1, 1)
                        info = (grad**2).sum().cpu()
                        infos.append(info)
                        self.optimizer.zero_grad()
                
            buffer_initial_info.extend(infos)          
        return buffer_initial_info
    
    
    
    def add_new_class(self, class_name):
        super().add_new_class(class_name)
        self.new_exposed_classes.append(class_name)
        self.memory.new_exposed_classes = self.new_exposed_classes
    
    
    def model_forward_samplewise(self, batch):
        inps, targets = self.preprocess_batch(batch)

        with torch.cuda.amp.autocast(enabled=False):
            # 모델 실행: output = {"AUX": ..., "Main": ...}
            # for input_s in batch["img"]:
            # outputs = self.model(batch["img"])
            # aux_raw = outputs["AUX"]
            # main_raw = outputs["Main"]

            # Vec2Box 변환: [B, A, C], [B, A, R], [B, A, 4]
            # aux_predicts = self.vec2box(aux_raw)
            # main_predicts = self.vec2box(main_raw)
            image_tensors = inps.tensors
        
            # 손실 계산
            infos = []
            loss = None
            
            if "loss" in self.selection_method:
                for ind in range(len(targets)):
                    loss_item = self.model(image_tensors[ind], [targets[ind]])
                    sample_loss = loss_item["total_loss"]
                    infos.append(sample_loss.detach().cpu().item())
                    
                    # infos.append(sample_loss.detach().cpu().item())
                    if loss == None:
                        loss = sample_loss
                    else:
                        loss += sample_loss
                        
                loss /= len(image_tensors)
                    
            elif "entropy" in self.selection_method:
                for ind in range(len(targets)):
                    loss_item, sample_logit = self.model(image_tensors[ind], [targets[ind]], get_features=True)
                    sample_logit = sample_logit[-1]
                    sample_loss = loss_item["total_loss"]
                    probs = F.softmax(sample_logit, dim=0)
                    info = -torch.sum(probs * torch.log(probs + 1e-8)).item()
                    infos.append(info)
                    
                    if loss == None:
                        loss = sample_loss
                    else:
                        loss += sample_loss
                        
                loss /= len(image_tensors)
            
            elif self.selection_method == "fisher":
                for ind in range(len(targets)):
                    loss_item = self.model(image_tensors[ind], [targets[ind]])["total_loss"]
                    sample_loss = loss_item["total_loss"]
                    for n, p in self.model.neck.named_parameters():
                        if p.requires_grad == True:
                            grad = torch.autograd.grad(sample_loss, p, retain_graph=True)[0].clone().detach().clamp(-1, 1)
                    info = (grad**2).sum().cpu()
                    infos.append(info)
                    
                    if loss == None:
                        loss = sample_loss
                    else:
                        loss += sample_loss
                    
                loss /= len(image_tensors)
                
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.total_flops += self.backward_flops
            
            self.update_schedule()
                    
            
        return loss, infos
    
    
    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        total_loss = 0.0
        if len(sample) > 0:
            self.memory.register_stream(sample)
        for i in range(iterations):
            self.model.train()
            batch = self.memory.get_batch(batch_size, stream_batch_size)

            # batch = {
            #         "img": data[0],         # images
            #         "cls": data[1],         # labels
            #         "img_id": data[2],    # image paths
            #     }
            
            # print(f"[DEBUG] batch images: {batch['img_path']}")
            # print(f"[DEBUG] batch labels shape: {batch['cls'].shape}")

            self.optimizer.zero_grad()
            
            info = []
            loss, info = self.model_forward_samplewise(batch)
            # infos.extend(info)
        
            # else:
            #     # print(f"[DEBUG] individual losses: {loss_item}")
            #     if self.use_amp:
            #         self.scaler.scale(loss).backward()
            #         self.scaler.step(self.optimizer)
            #         self.scaler.update()
            #     else:
            #         loss.backward()
            #         self.optimizer.step()
                
            #     self.total_flops += (len(data[1]) * self.backward_flops)
                
            #     self.update_schedule()

            total_loss += loss.item()
            stream_info = info[:len(self.temp_batch)]
            memory_info = info[len(self.temp_batch):]
            self.memory.update_info(memory_info)
                # self.total_flops += (batch_size * (self.forward_flops + self.backward_flops))
                # print("self.total_flops", self.total_flops)
        return total_loss / iterations, stream_info
   
    
    
    
    def online_step(self, sample, sample_num, n_worker):
        if sample.get('klass',None) and sample['klass'] not in self.exposed_classes:
            self.online_after_task(sample_num)
            self.add_new_class(sample['klass'])
        elif sample.get('domain',None) and sample['domain'] not in self.exposed_domains:
            self.exposed_domains.append(sample['domain'])
            # self.new_exposed_classes.append(sample['domain'])
            # self.memory.new_exposed_classes.append(sample['domain'])
            # self.memory.cls_count.append(0)
            # self.memory.cls_idx.append([])
        
        self.temp_batch.append(sample)
        self.num_updates += self.online_iter
        if len(self.temp_batch) == self.temp_batchsize:
            iteration = int(self.num_updates)
            if iteration != 0:
                train_loss, infos = self.online_train(self.temp_batch, self.batch_size, n_worker,
                                                      iterations=int(self.num_updates), stream_batch_size=self.temp_batchsize)
                self.report_training(sample_num, train_loss)
                
                for s_i, stored_sample in enumerate(self.temp_batch):
                    self.update_memory(stored_sample, infos[s_i])

                self.temp_batch = []
                self.num_updates -= int(self.num_updates)
    
    def update_memory(self, sample, info):
        self.balanced_replace_memory(sample, info)

    def balanced_replace_memory(self, sample):
        if len(self.memory) >= self.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            if sample.get('klass', None):
                sample_category = sample['klass']
            elif sample.get('domain', None):
                sample_category = sample['domain']
            else:
                sample_category = 'pretrained'
            
            label_frequency[self.new_exposed_classes.index(sample_category)] += 1
            cls_to_replace = np.random.choice(
                np.flatnonzero(np.array(label_frequency) == np.array(label_frequency).max()))
            idx_to_replace = np.random.choice(self.memory.cls_idx[cls_to_replace])
            labels = self.memory.buffer[idx_to_replace][1]
            self.memory.replace_sample(sample, idx_to_replace)
            
            if sample.get('klass', None):
                classes = [obj['category_id'] for obj in labels]
                classes = [self.memory.contiguous_class2id[self.memory.ori_id2class[c]] 
                        for c in classes]
                classes = list(set(classes))
                for cls_ in classes:
                    self.memory.cls_count[cls_] -= 1
                    self.memory.cls_idx[cls_].remove(idx_to_replace)
                
                new_labels = self.memory.buffer[idx_to_replace][1]
                new_classes = [obj['category_id'] for obj in new_labels]
                new_classes = [self.memory.contiguous_class2id[self.memory.ori_id2class[c]] 
                        for c in new_classes]
                new_classes = list(set(new_classes))
                for cls_ in new_classes:
                    self.memory.cls_idx[cls_].append(idx_to_replace)
            else:
                self.memory.cls_count[cls_to_replace] -= 1
                self.memory.cls_idx[cls_to_replace].remove(idx_to_replace)
                self.memory.cls_idx[self.new_exposed_classes.index(sample_category)].append(idx_to_replace)
        else:
            self.memory.replace_sample(sample)
