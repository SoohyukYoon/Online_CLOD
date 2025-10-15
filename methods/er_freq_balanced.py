import logging
import copy
import numpy as np
from methods.er_baseline import ER
from utils.data_loader import FreqClsBalancedDataset

logger = logging.getLogger()

class ERFreqBalanced(ER):
    def initialize_memory_buffer(self, memory_size):
        self.memory_size = memory_size - 8
        data_args = self.damo_cfg.get_data(self.damo_cfg.dataset.train_ann[0])
        self.memory = FreqClsBalancedDataset(ann_file=data_args['args']['ann_file'], root=data_args['args']['root'], transforms=None,class_names=self.damo_cfg.dataset.class_names,
            dataset=self.dataset, cls_list=self.exposed_classes, device=self.device, memory_size=self.memory_size, image_size=self.img_size, aug=self.damo_cfg.train.augment)
        
        self.new_exposed_classes = ['pretrained']

    def add_new_class(self, class_name):
        super().add_new_class(class_name)
        self.new_exposed_classes.append(class_name)
        self.memory.new_exposed_classes = self.new_exposed_classes
    
    def online_step(self, sample, sample_num, n_worker):
        if sample.get('klass',None) and sample['klass'] not in self.exposed_classes:
            self.online_after_task(sample_num)
            self.add_new_class(sample['klass'])
        elif sample.get('domain',None) and sample['domain'] not in self.exposed_domains:
            self.exposed_domains.append(sample['domain'])
            self.new_exposed_classes.append(sample['domain'])
            self.memory.new_exposed_classes.append(sample['domain'])
            self.memory.cls_count.append(0)
            self.memory.cls_idx.append([])
        
        self.temp_batch.append(sample)
        self.num_updates += self.online_iter
        if len(self.temp_batch) == self.temp_batchsize:
            iteration = int(self.num_updates)
            if iteration != 0:
                train_loss = self.online_train(self.temp_batch, self.batch_size, n_worker,
                                                      iterations=int(self.num_updates), stream_batch_size=self.temp_batchsize)
                self.report_training(sample_num, train_loss)
                
                for stored_sample in self.temp_batch:
                    self.update_memory(stored_sample)

                self.temp_batch = []
                self.num_updates -= int(self.num_updates)
    
    def update_memory(self, sample):
        self.balanced_replace_memory(sample)

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
            labels = self.memory.buffer[idx_to_replace]['labels']
            self.memory.replace_sample(sample, idx_to_replace)
            
            if sample.get('klass', None):
                classes = [obj['category_id'] for obj in labels]
                classes = [self.memory.contiguous_class2id[self.memory.ori_id2class[c]] 
                        for c in classes]
                classes = list(set(classes))
                for cls_ in classes:
                    self.memory.cls_count[cls_] -= 1
                    self.memory.cls_idx[cls_].remove(idx_to_replace)
                
                new_labels = self.memory.buffer[idx_to_replace]['labels']
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
