from PIL import Image
from torchvision.transforms import ToTensor
import os
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as TF
import json

def resize_to_yolo_input(img_tensor, size=640):
    H, W = img_tensor.shape[2:]
    if H != size or W != size:
        img_tensor = F.interpolate(img_tensor, size=(size, size), mode='bilinear', align_corners=False)
    return img_tensor

def get_gt_boxes_for_sample(file_name: str, annotation_root: str):
    split, file_base = file_name.split("/")
    json_path = os.path.join(annotation_root, f"instances_{split}.json")

    if not os.path.exists(json_path):
        print(f"[GT] Annotation JSON not found: {json_path}")
        return None, (None, None)

    with open(json_path, "r") as f:
        data = json.load(f)

    file_to_meta = {}
    for img in data['images']:
        base_name = os.path.splitext(img['file_name'])[0]
        file_to_meta[base_name] = (img['id'], img['width'], img['height'])

    if file_base not in file_to_meta:
        print(f"[GT] Image ID not found for {file_base} in {json_path}")
        return None, (None, None)

    image_id_int, width, height = file_to_meta[file_base]
    boxes = []
    for ann in data['annotations']:
        if ann['image_id'] != image_id_int or ann.get('iscrowd', 0): continue
        x, y, w, h = ann['bbox']
        x1, y1, x2, y2 = x, y, x + w, y + h
        boxes.append([ann['category_id'], x1, y1, x2, y2])

    return torch.tensor(boxes, dtype=torch.float32) if boxes else None, (width, height)

def basic_pseudo_labels(model, post_process, device, batch, conf_thresh=0.7, topk=5, current_class=None):
    model.eval()
    pseudo_labeled = []

    for i, sample in enumerate(batch):
        image_root = "/home/vision/chaeeunlee/YOLOv9/Online_CLOD/data/voc/images"
        filename = sample["file_name"]
        img_path = os.path.join(image_root, filename + ".jpg")
        if not os.path.exists(img_path):
            print(f"[PseudoLabel] Sample {i}: File not found → {img_path}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            to_tensor = ToTensor()
            img_tensor = to_tensor(img).unsqueeze(0).to(device)
            img_tensor = resize_to_yolo_input(img_tensor, size=640)
        except Exception as e:
            print(f"[PseudoLabel] Sample {i}: Failed to load image → {e}")
            continue

        H, W = img_tensor.shape[2:]

        with torch.no_grad():
            output = model(img_tensor)
            preds = post_process(output, image_size=[W, H])[0]

        if preds.shape[0] == 0:
            print(f"[PseudoLabel] Sample {i}: No predictions")
            continue

        confidences = preds[:, 5]
        preds = preds[confidences >= conf_thresh]

        if preds.shape[0] == 0:
            print(f"[PseudoLabel] Sample {i}: All predictions below threshold")
            continue

        preds = preds[torch.argsort(preds[:, 5], descending=True)]

        top_preds = preds[:min(len(preds), topk)]
        top_preds[:, 0] += 1

        if current_class is not None:
            top_preds = top_preds[top_preds[:, 0] < current_class]

        if top_preds.shape[0] == 0:
            print(f"[PseudoLabel] Sample {i}: No pseudo-labels with class < {current_class}")
            continue

        cls = top_preds[:, 0].long().unsqueeze(0).unsqueeze(-1)
        boxes = top_preds[:, 1:5].unsqueeze(0)
        target = torch.cat([cls.float(), boxes], dim=-1)

        annotation_root = "/home/vision/chaeeunlee/YOLOv9/Online_CLOD/data/voc/annotations"
        gt_tensor, original_size = get_gt_boxes_for_sample(filename, annotation_root)

        pseudo_sample = {
            "img": img_tensor.squeeze(0),
            "cls": target.to(device),
            "gt": gt_tensor,
            "original_size": original_size,
            "img_path": img_path,
            "file_name": sample.get("file_name", ""),
            "klass": sample.get("klass", None),
        }
        pseudo_labeled.append(pseudo_sample)
        print(f"[PseudoLabel] Sample {i}: {top_preds.shape[0]} pseudo-labels added")

    print(f"[PseudoLabel] Total pseudo-labeled samples: {len(pseudo_labeled)}")
    return pseudo_labeled

def visualize_pseudo_sample(pseudo_sample, save_dir="/home/vision/chaeeunlee/YOLOv9/Online_CLOD/pseudo_vis", suffix=""):
    os.makedirs(save_dir, exist_ok=True)
    img_tensor = pseudo_sample["img"].cpu()
    bbox_tensor = pseudo_sample["cls"].cpu().squeeze(0)
    gt_tensor = pseudo_sample.get("gt", None)
    orig_w, orig_h = pseudo_sample.get("original_size", (None, None))

    if gt_tensor is not None and orig_w and orig_h:
        gt_tensor = gt_tensor.clone().cpu()
        scale_x = 640 / orig_w
        scale_y = 640 / orig_h
        gt_tensor[:, 1] *= scale_x  # x1
        gt_tensor[:, 2] *= scale_y  # y1
        gt_tensor[:, 3] *= scale_x  # x2
        gt_tensor[:, 4] *= scale_y  # y2

    img = TF.to_pil_image(img_tensor)
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Draw pseudo-labels in RED
    for box in bbox_tensor:
        class_id = int(box[0].item())
        x1, y1, x2, y2 = box[1:].tolist()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                 edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"pseudo:{class_id}", color='red', fontsize=10, backgroundcolor='white')

    # Draw GT bboxes in GREEN
    if gt_tensor is not None:
        for box in gt_tensor:
            class_id = int(box[0].item())
            x1, y1, x2, y2 = box[1:].tolist()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                     edgecolor='green', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(x1, y1 + 10, f"GT:{class_id}", color='green', fontsize=10, backgroundcolor='white')

    plt.axis("off")
    filename = os.path.basename(pseudo_sample.get("file_name", "unnamed"))
    save_path = os.path.join(save_dir, f"{filename}{suffix}.jpg")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"[PseudoLabel] Visualization saved to {save_path}")