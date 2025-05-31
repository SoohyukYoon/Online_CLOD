# Online CLOD (Continual Learning Object Detection)

## Environment
```bash
conda create -n clod python==3.10 -y
conda activate clod
git clone https://github.com/ta3h30nk1m/Online_CLOD.git
cd Online_CLOD
pip install -r requirements.txt
```


## Download Datasets
#### VOC
```bash
python -m preprocess_dataset.download_voc
```

#### SHIFT
```bash
python -m preprocess_dataset.download_shift
```

#### BDD100K
```bash
python -m preprocess_dataset.download_bdd
```


## Preprocess Datasets
#### VOC_10
```bash
python -m preprocess_dataset.preprocess_voc10
```

#### BDD100K source
```bash
python -m preprocess_dataset.preprocess_bdd_source
```
#### BDD100K dawn/dusk
```bash
python -m preprocess_dataset.preprocess_bdd_dawndusk
```
#### BDD100K night
```bash
python -m preprocess_dataset.preprocess_bdd_night
```
#### BDD100K rainy/snowy
```bash
python -m preprocess_dataset.preprocess_bdd_rainy
```
#### BDD100K cloudy
```bash
python -m preprocess_dataset.preprocess_bdd_cloudy
```
- Subsample bdd100K validation set
```bash
python subsample_bdd_validation.py
```

## Training

#### Pretrain VOC_10
```bash
python yolo/lazy.py task=train name='pre_train' dataset=voc_10.yaml use_wandb=False
```

#### Joint-train VOC
```bash
python yolo/lazy.py task=train name='joint_train' dataset=voc.yaml use_wandb=False
```

#### Joint-train SHIFT
```bash
python yolo/lazy.py task=train name='joint_train' dataset=shift.yaml use_wandb=False
```


#### Joint-train BDD100K
```bash
python yolo/lazy.py task=train name='joint_train' dataset=bdd100k.yaml use_wandb=False
```


## Zip/Unzip Annotations
```bash
python -m preprocess_dataset.annotations_zip
python -m preprocess_dataset.annotations_unzip
```