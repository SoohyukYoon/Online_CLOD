import json
from collections import Counter, defaultdict
from pathlib import Path
import random
import os

def balanced_subsample(coco_path, out_path="instances_val_balanced.json",
                       max_images=1_000, seed=42):
    random.seed(seed)
    coco = json.loads(Path(coco_path).read_text())

    # Early-exit if already small enough
    if len(coco["images"]) <= max_images:
        Path(out_path).write_text(json.dumps(coco, indent=2))
        print("Dataset already ≤ max_images; copied unchanged.")
        return

    # ---------- build helpers ----------
    cat_ids = [c["id"] for c in coco["categories"]]

    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    img_cat_cnt = {
        img_id: Counter(a["category_id"] for a in anns)
        for img_id, anns in anns_by_img.items()
    }

    cat_queues = {cid: [] for cid in cat_ids}
    for img_id, cnt in img_cat_cnt.items():
        for cid, n in cnt.items():
            cat_queues[cid].append((n, img_id))
    for q in cat_queues.values():
        q.sort(key=lambda x: (-x[0], x[1]))

    ptr = {cid: 0 for cid in cat_ids}
    selected, cat_boxes = set(), Counter()

    # 👉 categories that can still supply *new* images
    active_cats = set(cid for cid, q in cat_queues.items() if q)

    # ---------- greedy selection ----------
    while len(selected) < max_images and active_cats:
        cid = min(active_cats, key=lambda c: cat_boxes[c])
        q, p = cat_queues[cid], ptr[cid]

        # advance past already-selected images
        while p < len(q) and q[p][1] in selected:
            p += 1
        ptr[cid] = p

        if p == len(q):                  # queue exhausted ➜ retire category
            active_cats.remove(cid)
            continue                     # pick another category
        img_id = q[p][1]

        selected.add(img_id)
        cat_boxes.update(img_cat_cnt[img_id])

    print(f"Finished with {len(selected)} images "
          f"(budget {max_images}); per-class boxes: {dict(cat_boxes)}")

    # ---------- write COCO subset ----------
    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": [img for img in coco["images"] if img["id"] in selected],
        "annotations": [ann for ann in coco["annotations"]
                        if ann["image_id"] in selected],
    }
    Path(out_path).write_text(json.dumps(new_coco, indent=2))
    print(f"Saved to {out_path}")

# ---------- usage ----------
# balanced_subsample("instances_val.json")
balanced_subsample("data/bdd100k_cloudy/annotations/instances_val.json",
                   "data/bdd100k_cloudy/annotations/instances_val_subsampled.json")

balanced_subsample("data/bdd100k_dawndusk/annotations/instances_val.json",
                   "data/bdd100k_dawndusk/annotations/instances_val_subsampled.json")

balanced_subsample("data/bdd100k_night/annotations/instances_val.json",
                   "data/bdd100k_night/annotations/instances_val_subsampled.json")

balanced_subsample("data/bdd100k_rainy/annotations/instances_val.json",
                   "data/bdd100k_rainy/annotations/instances_val_subsampled.json")

balanced_subsample("data/bdd100k_source/annotations/instances_val.json",
                   "data/bdd100k_source/annotations/instances_val_subsampled.json")

os.rename("data/bdd100k_cloudy/annotations/instances_val.json",
          "data/bdd100k_cloudy/annotations/instances_val_full.json")
os.rename("data/bdd100k_cloudy/annotations/instances_val_subsampled.json",
          "data/bdd100k_cloudy/annotations/instances_val.json")

os.rename("data/bdd100k_dawndusk/annotations/instances_val.json",
          "data/bdd100k_dawndusk/annotations/instances_val_full.json")
os.rename("data/bdd100k_dawndusk/annotations/instances_val_subsampled.json",
          "data/bdd100k_dawndusk/annotations/instances_val.json")

os.rename("data/bdd100k_night/annotations/instances_val.json",
          "data/bdd100k_night/annotations/instances_val_full.json")
os.rename("data/bdd100k_night/annotations/instances_val_subsampled.json",
          "data/bdd100k_night/annotations/instances_val.json")

os.rename("data/bdd100k_rainy/annotations/instances_val.json",
          "data/bdd100k_rainy/annotations/instances_val_full.json")
os.rename("data/bdd100k_rainy/annotations/instances_val_subsampled.json",
          "data/bdd100k_rainy/annotations/instances_val.json")

os.rename("data/bdd100k_source/annotations/instances_val.json",
          "data/bdd100k_source/annotations/instances_val_full.json")
os.rename("data/bdd100k_source/annotations/instances_val_subsampled.json",
          "data/bdd100k_source/annotations/instances_val.json")
