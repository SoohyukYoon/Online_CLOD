from pathlib import Path
import json
import os
import re
import shutil

VIDEO_WHITELIST = {
    "uav0000079_00480_v",
    "uav0000084_00000_v",
    "uav0000086_00000_v",
    "uav0000099_02109_v",
    "uav0000266_03598_v",
    "uav0000218_00001_v",
    "uav0000366_00001_v",
    "uav0000352_05980_v",
    "uav0000288_00001_v",
    "uav0000137_00458_v",
    "uav0000124_00944_v",
    "uav0000270_00001_v",
    "uav0000072_04488_v",
    "uav0000072_05448_v",
    "uav0000071_03240_v",
    "uav0000072_06432_v",
    "uav0000140_01590_v",
    "uav0000281_00460_v",
    "uav0000117_02622_v",
    "uav0000243_00001_v",
    "uav0000239_12336_v",
    "uav0000289_00001_v",
    "uav0000316_01288_v",
    "uav0000308_00000_v",
}

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


def filter_dataset(original_path, output_path, target_dataset, label_bound):

    ann_path = original_path / "annotations" / f"instances_{target_dataset}.json"
    with open(ann_path, "r", encoding="utf-8") as file:
        labels_data = json.load(file)

    # 1) Filter annotations by category_id <= label_bound (original behavior)
    kept_anns = [a for a in labels_data["annotations"] if a.get("category_id", 10**9) <= label_bound]

    # 2) Build a set of image_ids that belong to whitelisted sequences
    #    (first, map image_id -> seq_name by scanning images)
    image_id_to_seq = {}
    for img in labels_data["images"]:
        seq = _extract_seq_name(img)
        image_id_to_seq[img["id"]] = seq

    allowed_image_ids_by_video = {
        img["id"]
        for img in labels_data["images"]
        if (target_dataset == 'test') or (target_dataset == 'trainval' and (seq := image_id_to_seq.get(img["id"])) in VIDEO_WHITELIST)
    }

    # 3) Keep only annotations whose image_id is from the whitelisted videos
    kept_anns = [a for a in kept_anns if a["image_id"] in allowed_image_ids_by_video]
    labels_data["annotations"] = kept_anns

    # 4) Filter categories (original behavior)
    labels_data["categories"] = [
        c for c in labels_data["categories"] if c.get("id", 10**9) <= label_bound
    ]

    # 5) Filter images: keep only those that (a) are in whitelisted videos and
    #    (b) actually have at least one remaining annotation
    images_with_anns = {a["image_id"] for a in labels_data["annotations"]}
    labels_data["images"] = [
        img for img in labels_data["images"]
        if img["id"] in images_with_anns and img["id"] in allowed_image_ids_by_video
    ]
    print(len(labels_data["images"]))

    # 6) Copy images
    for image in labels_data["images"]:
        filename = image["file_name"]
        src = original_path / "images" / target_dataset / filename
        if not src.exists():
            raise FileNotFoundError(f"Image file {src} does not exist.")
        dst = output_path / "images" / target_dataset / filename
        os.makedirs(dst.parent, exist_ok=True)
        shutil.copy(src, dst)

    # 7) Save filtered labels
    os.makedirs(output_path / "annotations", exist_ok=True)
    out_ann = output_path / "annotations" / f"instances_{target_dataset}.json"
    with open(out_ann, "w", encoding="utf-8") as file:
        json.dump(labels_data, file, ensure_ascii=False, indent=4)
    print(f"[OK] Saved: {out_ann}")


if __name__ == "__main__":

    original_path = Path("./data/VisDrone2019-VID")
    output_path   = Path("./data/VisDrone2019-VID-3")
    target_datasets = ["trainval", "test"]
    label_bound = 3

    for target_dataset in target_datasets:
        print(f"Processing {target_dataset}")
        filter_dataset(original_path, output_path, target_dataset, label_bound)
