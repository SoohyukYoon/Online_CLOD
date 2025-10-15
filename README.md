# Online CLOD (Continual Learning for Object Detection)

Tools and recipes for **online continual learning** in object detection across popular datasets (VOC, COCO, BDD100K, SHIFT) and custom data.

## Requirements

* Python **3.10**
* Conda (recommended)
* CUDA-enabled PyTorch environment (version depends on your CUDA/toolkit). See the official PyTorch install guide if needed.

---

## Quick Start

```bash
# 1) Create env
conda create -n clod_damo python=3.10 -y
conda activate clod_damo

# 2) Clone
git clone https://github.com/ta3h30nk1m/Online_CLOD.git -b CLOD_DAMO
cd Online_CLOD

# 3) Install
pip install -r requirements.txt

# 4) Download + preprocess a dataset (example: VOC)
python -m preprocess_dataset.download_voc
python -m preprocess_dataset.preprocess_voc10

# 5) (Optional) Pretrain on base classes / source domain
torchrun --nproc_per_node=4 -m tools.train -f configs/pretrain_voc_10.py

# 6) Run an experiment (GPU 0)
bash main.sh 0
```

---

## Environment Setup

```bash
conda create -n clod_damo python=3.10 -y
conda activate clod_damo

git clone https://github.com/ta3h30nk1m/Online_CLOD.git -b CLOD_DAMO
cd Online_CLOD

pip install -r requirements.txt
```

---

## Dataset Preparation

### Download

Run any of the following to fetch datasets.

**VOC**

```bash
python -m preprocess_dataset.download_voc
```

**COCO**

```bash
python -m preprocess_dataset.download_coco
```

**SHIFT**

```bash
python -m preprocess_dataset.download_shift
```

**BDD100K**

```bash
python -m preprocess_dataset.download_bdd
```

### Preprocess

**VOC**

```bash
python -m preprocess_dataset.preprocess_voc10
python -m preprocess_dataset.preprocess_voc15
```

**COCO**

```bash
python -m preprocess_dataset.preprocess_coco40
```

**BDD100K**

```bash
# Source / base
python -m preprocess_dataset.preprocess_bdd_source
# Subdomains
python -m preprocess_dataset.preprocess_bdd_dawndusk
python -m preprocess_dataset.preprocess_bdd_night
python -m preprocess_dataset.preprocess_bdd_rainy
python -m preprocess_dataset.preprocess_bdd_cloudy
```

**SHIFT**

```bash
# Source / base
python -m preprocess_dataset.preprocess_shift_source
# Subdomains
python -m preprocess_dataset.preprocess_shift_dawndusk
python -m preprocess_dataset.preprocess_shift_night
python -m preprocess_dataset.preprocess_shift_foggy
python -m preprocess_dataset.preprocess_shift_cloudy
python -m preprocess_dataset.preprocess_shift_overcast
python -m preprocess_dataset.preprocess_shift_rainy
```

### Subsample Validation Splits

```bash
# BDD100K
python subsample_bdd_validation.py

# SHIFT
python subsample_shift_validation.py
```

---

## Using a Custom Dataset

1. **Images**: place all images in a single folder.
2. **Annotation**: prepare a single **COCO-style** JSON file:

```json
{
  "images": [
    {"file_name": "XX.jpg", "height": 500, "width": 500, "id": 0}
  ],
  "annotations": [
    {
      "area": 19536,
      "iscrowd": 0,
      "image_id": 1,
      "bbox": [47, 239, 148, 132],
      "category_id": 12,
      "id": 1,
      "ignore": 0
    }
  ],
  "categories": [
    {"supercategory": "none", "id": 1, "name": "aeroplane"}
  ]
}
```

> **Important:** All `id` fields must be **integers**.

3. **Register paths**: add your dataset directory in `damo/config/paths_catalog.py`.
4. **Create a config**: add a config under `configs/` (or `config/` as used in your branch). Use existing datasets as reference, and be sure to set:

   * `self.miscs.output_dir`
   * `self.dataset.train_ann`
   * `self.dataset.val_ann`
   * `self.dataset.class_names`
   * `num_classes`

---

## Training

### Pretraining

Create a pretrained model on initial classes (class-incremental) or a source domain (domain-incremental). Use one of the `configs/pretrain_*.py` files with the provided launcher.

**Example (VOC 10):**

```bash
torchrun --nproc_per_node=4 -m tools.train -f configs/pretrain_voc_10.py
```

---

## Run Continual Learning Experiments

Launch experiments via the helper script (single-GPU example shown):

```bash
bash main.sh <gpu_id>
# e.g.,
bash main.sh 0
```

### Configuration via `main.sh`

The following environment variables control an experiment; edit them inside `main.sh`.

| Variable      | Meaning                                       | Examples / Notes                                                           |
| ------------- | --------------------------------------------- | -------------------------------------------------------------------------- |
| `NOTE`        | Human-readable experiment name                | `voc10to10_er_balanced_50mem`                                              |
| `MODE`        | Method to run (see `utils/method_manager.py`) | `er`, `er_frequency`, `er_balanced`, `er_freq_balanced`, `adaptive_freeze` |
| `DATASET`     | Dataset preset                                | `"VOC_10_10"`, `"BDD_domain"`, `"SHIFT_domain"`                            |
| `MEM_SIZE`    | Replay memory size                            | integer (e.g., `500`, `1000`)                                                |
| `ONLINE_ITER` | Training iterations per stream sample         | float (e.g., `0.5`, `1`)                                                   |
| `EVAL_PERIOD` | Evaluation interval                           | integer steps (e.g., `500`)                                               |
| `LR`          | Learning rate                                 | e.g., `1e-4`                                                               |

> Check `utils/method_manager.py` for the latest list of supported methods.

---

## Utilities

* **Zip annotations**

  ```bash
  python -m preprocess_dataset.annotations_zip
  ```
* **Unzip annotations**

  ```bash
  python -m preprocess_dataset.annotations_unzip
  ```

---

## Tips & Notes

* Ensure datasets are **fully downloaded** before preprocessing.
* Some commands can be long-running—consider using `tmux` or `screen` on remote servers.
* For multi-GPU training, use `torchrun --nproc_per_node=<N>`; for single-GPU, setting `CUDA_VISIBLE_DEVICES` is sufficient.
* Keep an eye on disk usage; COCO/BDD/SHIFT can consume significant space after extraction and preprocessing.
