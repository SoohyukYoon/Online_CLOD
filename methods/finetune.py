import logging
from methods.er_baseline import ER
from utils.data_loader import MemoryDataset

logger = logging.getLogger()

class FINETUNE(ER):
    def __init__(self, n_classes, device, **kwargs):
        super().__init__(n_classes=n_classes, device=device, **kwargs)
        
        self.temp_batchsize = self.batch_size
    
    def initialize_memory_buffer(self, memory_size):
        self.memory_size = 1#memory_size - self.temp_batchsize
        data_args = self.damo_cfg.get_data(self.damo_cfg.dataset.train_ann[0])
        self.memory = MemoryDataset(ann_file=data_args['args']['ann_file'], root=data_args['args']['root'], transforms=None,class_names=self.damo_cfg.dataset.class_names,
            dataset=self.dataset, cls_list=self.exposed_classes, device=self.device, memory_size=self.memory_size, image_size=self.img_size, aug=self.damo_cfg.train.augment)