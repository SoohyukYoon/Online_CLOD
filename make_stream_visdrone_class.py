import json
import os
import random
import numpy as np
import scipy.stats as stats
import re

SEQ_RE = re.compile(r"(uav\d{7}_\d{5}_v)")

def _extract_seq_name(image_entry):
    """
    Try to get the sequence/video name from the COCO image entry.
    1) If there is a custom 'seq_name' field, use it.
    2) Else parse from file_name using regex like '.../uav0000009_03358_v/0000001.jpg'
    """
    if "seq_name" in image_entry:
        return image_entry["seq_name"]
    fname = image_entry.get("file_name", "")
    m = SEQ_RE.search(fname.replace("\\", "/"))
    return m.group(1) if m else None

dataset = 'VisDrone2019-VID'
output_dataset = 'VisDrone2019-VID_class_3_4'
dataset_dir = f'data/{dataset}'
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'test')

repeats = [1]
sigmas = [0.1]
seeds = [1, 2, 3]
init_cls = 1

clses = list(range(3, 7))
cls_list = {3: 'van',
  4: 'truck',
  5: 'motorcycle',
  6: 'bus'
}

cls_datalist = {}
# for id in cls_list.keys():
#     cls_datalist[id] = []

labels = json.load(open(f'data/{dataset}/annotations/instances_trainval.json','r'))

images = labels['images']
annotations = labels['annotations']

whitelist = {
    3: ['uav0000248_00001_v',
        'uav0000361_02323_v',
        'uav0000323_01173_v',
        'uav0000315_00000_v',
        'uav0000268_05773_v',
        'uav0000342_04692_v',
        'uav0000239_03720_v',
        'uav0000329_04715_v',
        'uav0000339_00001_v',
        'uav0000145_00000_v',
        'uav0000305_00000_v'],
    4: ['uav0000273_00001_v',
        'uav0000266_04830_v',
        'uav0000076_00720_v',
        'uav0000279_00001_v',
        'uav0000307_00000_v',
        'uav0000264_02760_v',
        'uav0000326_01035_v',
        'uav0000289_06922_v'],
    5: ['uav0000013_00000_v',
        'uav0000013_01073_v',
        'uav0000013_01392_v',
        'uav0000020_00406_v',
        'uav0000278_00001_v',
        'uav0000182_00000_v',
        'uav0000150_02310_v',
        'uav0000363_00001_v',
        'uav0000360_00001_v',
        'uav0000357_00920_v',
        'uav0000300_00000_v',
        'uav0000308_01380_v',
        'uav0000309_00000_v'],
    6: ['uav0000222_03150_v',
        'uav0000244_01440_v',
        'uav0000263_03289_v',
        'uav0000138_00000_v',
        'uav0000295_02300_v',
        'uav0000126_00001_v',
        'uav0000143_02250_v']
}

for cls_id, wl in whitelist.items():
    for img in images:
        seq = _extract_seq_name(img)
        if seq in wl:
            # cls_datalist[cls_id].append(img['file_name'][:-4])
            if seq in cls_datalist:
                cls_datalist[seq].append({'filename':img['file_name'][:-4], 'frame_id':int(img['frame_id'])})
            else:
                cls_datalist[seq] = [{'filename':img['file_name'][:-4], 'frame_id':int(img['frame_id'])}]

# sort each cls_datalist[seq] by frame id
for seq in cls_datalist.keys():
    cls_datalist[seq] = sorted(cls_datalist[seq], key=lambda x: x['frame_id'])
    # print(cls_datalist[seq])
    cls_datalist[seq] = [x['filename'] for x in cls_datalist[seq]]

for repeat in repeats:
    for sigma in sigmas:
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            stream = []
            for cls_id, wl in whitelist.items():
                random.shuffle(wl)
                for seq in wl:
                    samples = cls_datalist[seq]
                    for sample in samples:
                        stream.append({'file_name': sample, 'klass': cls_list[cls_id], 'label':cls_id, 'time':0})
            data = {'cls_dict':cls_list, 'stream':stream, 'cls_addition':[0]*len(clses)}
            
            with open(f'collections/{output_dataset}/{output_dataset}_sigma{int(sigma*100)}_repeat{repeat}_seed{seed}.json', 'w') as fp:
                json.dump(data, fp)                
                
                
# val = []
# cls_list = os.listdir(train_dir)
# n_classes = len(cls_list)
# cls_dict = {cls:i for i, cls in enumerate(cls_list)}
# for i in range(n_classes):
#     cls_val_list = os.listdir(os.path.join(val_dir, cls_list[i]))
#     for ii, sample in enumerate(cls_val_list):
#         val.append({'file_name': os.path.join('val/', cls_val_list[ii]), 'klass': cls_list[i], 'label':i})

# with open(f'collections/{dataset}/{dataset}_val2.json', 'w') as fp:
#     json.dump(val, fp)