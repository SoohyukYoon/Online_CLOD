# Online CLOD (Continual Learning Object Detection)

This repository provides tools for online continual learning in object detection tasks using various popular datasets.

## Environment Setup

Follow these instructions to set up the environment:

### Step 1: Create and Activate Conda Environment

```bash
conda create -n clod python=3.10 -y
conda activate clod
```

### Step 2: Clone Repository

```bash
git clone https://github.com/ta3h30nk1m/Online_CLOD.git
cd Online_CLOD
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Download Datasets

Run the following commands to download the necessary datasets:

### VOC Dataset

```bash
python -m preprocess_dataset.download_voc
```

### COCO Dataset

```bash
python -m preprocess_dataset.download_coco
```

### SHIFT Dataset

```bash
python -m preprocess_dataset.download_shift
```

### BDD100K Dataset

```bash
python -m preprocess_dataset.download_bdd
```

## Preprocess Datasets

### VOC

```bash
python -m preprocess_dataset.preprocess_voc10
```

### COCO

```bash
python -m preprocess_dataset.preprocess_coco40
```

### BDD100K

- **Source:** `python -m preprocess_dataset.preprocess_bdd_source`
- **Dawn/Dusk:** `python -m preprocess_dataset.preprocess_bdd_dawndusk`
- **Night:** `python -m preprocess_dataset.preprocess_bdd_night`
- **Rainy/Snowy:** `python -m preprocess_dataset.preprocess_bdd_rainy`
- **Cloudy:** `python -m preprocess_dataset.preprocess_bdd_cloudy`

**Subsample BDD100K validation set:**

```bash
python subsample_bdd_validation.py
```

### SHIFT

- **Source:** `python -m preprocess_dataset.preprocess_shift_source`
- **Dawn/Dusk:** `python -m preprocess_dataset.preprocess_shift_dawndusk`
- **Night:** `python -m preprocess_dataset.preprocess_shift_night`
- **Foggy:** `python -m preprocess_dataset.preprocess_shift_foggy`
- **Cloudy:** `python -m preprocess_dataset.preprocess_shift_cloudy`
- **Overcast:** `python -m preprocess_dataset.preprocess_shift_overcast`
- **Rainy:** `python -m preprocess_dataset.preprocess_shift_rainy`

**Subsample SHIFT validation set:**

```bash
python subsample_shift_validation.py
```

## Training

### Pretraining

- **VOC\_10:**

```bash
python yolo/lazy.py task=train name='pre_train' dataset=voc_10.yaml use_wandb=False
```

- **COCO\_40:**

```bash
python yolo/lazy.py task=train name='pre_train' dataset=coco_40.yaml use_wandb=False
```

### Joint-training

- **VOC:**

```bash
python yolo/lazy.py task=train name='joint_train' dataset=voc.yaml use_wandb=False
```

- **COCO:**

```bash
python yolo/lazy.py task=train name='joint_train' dataset=coco.yaml use_wandb=False
```

- **SHIFT:**

```bash
python yolo/lazy.py task=train name='joint_train' dataset=shift.yaml use_wandb=False
```

- **BDD100K:**

```bash
python yolo/lazy.py task=train name='joint_train' dataset=bdd100k.yaml use_wandb=False
```

## Run Experiments

To run the continual learning experiment, execute:

```bash
bash main.sh <gpu_id>
```

Example (using GPU 0):

```bash
bash main.sh 0
```

### Arguments (customize in `main.sh`)

- `NOTE`: Experiment name
- `MODE`: Method to run (available methods listed in `utils/method_manager.py`)
- `DATASET`: Dataset to use (`"VOC_10_10"`, `"BDD_domain"`, `"SHIFT_domain"`)
- `MEM_SIZE`: Replay memory size
- `ONLINE_ITER`: Number of training iterations per sample in the online stream
- `EVAL_PERIOD`: Interval for evaluation
- `LR`: Learning rate

## Utilities

- **Zip annotations:**

```bash
python -m preprocess_dataset.annotations_zip
```

- **Unzip annotations:**

```bash
python -m preprocess_dataset.annotations_unzip
```

## Notes

- Make sure to check the availability of methods in `utils/method_manager.py` before running experiments.
- Adjust hyperparameters in `main.sh` according to your experimental needs.

