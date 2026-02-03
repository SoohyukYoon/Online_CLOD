# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import os

import torch
import cv2
import numpy as np
from PIL import Image
from loguru import logger
from tqdm import tqdm

from damo.dataset.datasets.evaluation import evaluate
from damo.utils import all_gather, get_world_size, is_main_process, synchronize
from damo.utils.timer import Timer, get_time_str
from damo.structures.boxlist_ops import boxlist_iou
from damo.utils.boxes import filter_results

# Global counter for sample_count
_sample_count = 0


def compute_on_dataset(model, data_loader, device, timer=None, tta=False,
                       sample_num=None, dataset_name=None, batch_size=None, score_threshold=None, dataset=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device('cpu')
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        # os.makedirs("TEST_TEST", exist_ok=True)
        # for i, img in enumerate(images.tensors):
        #     if i > 1: 
        #         break
        #     # Convert tensor to numpy (assuming CHW format, values 0-1 or normalized)
        #     img_np = img.cpu().permute(1, 2, 0).numpy()
        #     # Denormalize and convert to uint8
        #     if img_np.max() <= 1.0:
        #         img_np = (img_np * 255).astype(np.uint8)
        #     else:
        #         img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        #     pil_img = Image.fromarray(img_np)
        #     pil_img.save(f"TEST_TEST/{image_ids[i]}.jpg")
        with torch.no_grad():
            if timer:
                timer.tic()
                output = model(images.to(device))
            if timer:
                # torch.cuda.synchronize() # consume much time
                timer.toc()
            output = [o.to(cpu_device) if o is not None else o for o in output]
        results_dict.update(
            {img_id: result
             for img_id, result in zip(image_ids, output)})

        # Process only the first image in the batch
        first_img_id = image_ids[0]
        dataset = data_loader.dataset
        img_info = dataset.get_img_info(first_img_id)
        file_name = img_info['file_name']
        file_name_base = os.path.splitext(file_name)[0]
        formatted_name = f"{file_name_base}_{sample_num}"
        
        # Get first image data
        first_output = output[0]
        first_target = targets[0]
        
        # Load original image from dataset
        img_path = os.path.join(dataset.root, file_name)
        original_image = cv2.imread(img_path)
        if original_image is None:
            # If image not found, skip visualization
            continue
        
        # Apply NMS for visualization
        output_reduced = nms_for_visual_analysis(first_output, threshold=0.)
        
        # Check if GT has zero IoU with predictions
        gt_missing = gt_has_zero_IoU(first_target, output_reduced)
        
        # Create bbox visualization
        img_np = create_bbox(original_image, first_target, output_reduced, 
                            gt_missing=gt_missing, image_id=first_img_id,
                            dataset_name=dataset_name, batch_size=batch_size, 
                            score_threshold=score_threshold)
        
        # Store the image
        if img_np is not None:
            store_bbox(img_np, gt_missing, formatted_name, dataset_name, 
                      batch_size, score_threshold)

    return results_dict

def gt_has_zero_IoU(target, output_reduced):
    """
    If a GT-bbox has IoU==0 to every predicted bounding box return True immediately
    """
    if output_reduced is None or len(output_reduced.bbox) == 0:
        return len(target.bbox) > 0
    
    if len(target.bbox) == 0:
        return False
    
    # Compute IoU between GT and predictions
    iou_matrix = boxlist_iou(target, output_reduced)  # [N_gt, N_pred]
    
    # Check if any GT box has IoU==0 with all predictions
    max_iou_per_gt = iou_matrix.max(dim=1)[0]  # [N_gt]
    has_zero_iou = (max_iou_per_gt == 0).any().item()
    
    return has_zero_iou


def nms_for_visual_analysis(output, threshold=0.9):
    """
    Returns a new set of bboxes that have little overlap, uses the returned set for create_bbox()
    """
    if output is None or len(output.bbox) == 0:
        return output
    
    import torchvision
    from damo.structures.bounding_box import BoxList
    
    # Apply NMS with high threshold to reduce overlap
    boxes = output.bbox
    scores = output.get_field('scores')
    labels = output.get_field('labels')
    
    nms_out_index = torchvision.ops.nms(
        boxes,
        scores,
        threshold,
    )
    output_reduced = output[nms_out_index]
    
    return output_reduced


def create_bbox(image, target, output_reduced, gt_missing=False, image_id=None,
                dataset_name=None, batch_size=None, score_threshold=None):
    """
    Creates a new image that overlays GT and predicted bbox over the sample image. 
    Overlay GT bbox, and predicted class and its confidence score is right above the bbox in red
    Overlay predicted bbox, and predicted class and its confidence score is right above the bbox in blue

    Make the predicted class and confidence scores in the following format: 
    {class_index},{confidence score without the 0 or ., i.e. 0.36 -> 36 or 0.07 -> 7}, i.e. 0,3 -- 
    """
    # Image should already be loaded as BGR numpy array from cv2.imread
    if isinstance(image, np.ndarray):
        img_np = image.copy()
    else:
        # Fallback: convert if needed
        img_np = np.asarray(image)
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)
        # Ensure BGR format for cv2
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            # If RGB, convert to BGR
            if img_np[0, 0, 0] > img_np[0, 0, 2]:  # Heuristic: if R > B, likely RGB
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Get image dimensions for box scaling
    img_h, img_w = img_np.shape[:2]
    
    # Draw GT boxes in red
    if target is not None and len(target.bbox) > 0:
        gt_boxes = target.bbox.cpu().numpy()
        gt_labels = target.get_field('labels').cpu().numpy()
        target_size = target.size  # (width, height) from BoxList
        
        # Scale boxes if target size differs from image size
        scale_x = img_w / target_size[0] if target_size[0] > 0 else 1.0
        scale_y = img_h / target_size[1] if target_size[1] > 0 else 1.0
        
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box
            # Scale to image coordinates
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            # Draw GT box in red
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 0, 255), 1)
            # Draw label above box
            label_text = f"{int(label)}"
            cv2.putText(img_np, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Draw predicted boxes in blue
    if output_reduced is not None and len(output_reduced.bbox) > 0:
        pred_boxes = output_reduced.bbox.cpu().numpy()
        pred_labels = output_reduced.get_field('labels').cpu().numpy()
        pred_scores = output_reduced.get_field('scores').cpu().numpy()
        pred_size = output_reduced.size  # (width, height) from BoxList
        
        # Scale boxes if prediction size differs from image size
        scale_x = img_w / pred_size[0] if pred_size[0] > 0 else 1.0
        scale_y = img_h / pred_size[1] if pred_size[1] > 0 else 1.0
        
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            x1, y1, x2, y2 = box
            # Scale to image coordinates
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            # Draw predicted box in blue
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 1)
            
            # Format score: remove 0 and decimal point (0.36 -> 36, 0.07 -> 7)
            # Convert to integer representation without decimal
            score_int = int(score * 100)  # 0.36 -> 36, 0.07 -> 7
            score_str = str(score_int)
            
            # Format: {class_index},{score}
            label_text = f"{int(label)},{score_str}"
            cv2.putText(img_np, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return img_np

def store_bbox(img_np, gt_missing, image_id, dataset_name, batch_size, score_threshold):
    """
    Stores the bbox overlayed image in
    results/{dataset}_{temp_batchsize}_{score_threshold}_{no_GT or None}
    in which the image is named {original file name}_{sample_count} where the image type is jpg  
    """
    if img_np is None:
        return
    
    # Create directory name
    gt_suffix = "no_GT" if gt_missing else "None"
    dir_name = f"{dataset_name}_{batch_size}_{score_threshold}_{gt_suffix}_FUCK"
    output_dir = os.path.join("results", dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename: {original file name}_{sample_count}.jpg
    filename = f"{image_id}.jpg"
    output_path = os.path.join(output_dir, filename)
    
    # Save image
    cv2.imwrite(output_path, img_np)


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu,
                                               multi_gpu_infer):
    if multi_gpu_infer:
        all_predictions = all_gather(predictions_per_gpu)
    else:
        all_predictions = [predictions_per_gpu]
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger.warning(
            'Number of images that were gathered from multiple processes is'
            'not a contiguous set. Some images might be missing from the'
            'evaluation')

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
    model,
    data_loader,
    dataset_name,
    iou_types=('bbox', ),
    box_only=False,
    device='cuda',
    expected_results=(),
    expected_results_sigma_tol=4,
    output_folder=None,
    multi_gpu_infer=True,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    dataset = data_loader.dataset
    logger.info('Start evaluation on {} dataset({} images).'.format(
        dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    # Get batch_size from data_loader if available
    batch_size = getattr(data_loader, 'batch_size', None) or getattr(data_loader.batch_sampler, 'batch_size', None)
    # Get score_threshold from model config if available
    score_threshold = getattr(model.head, 'nms_conf_thre', None) if hasattr(model, 'head') else None
    predictions = compute_on_dataset(model, data_loader, device,
                                     inference_timer,
                                     dataset_name=dataset_name,
                                     batch_size=batch_size,
                                     score_threshold=score_threshold)
    # wait for all processes to complete before measuring the time
    if multi_gpu_infer:
        synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        'Total run time: {} ({} s / img per device, on {} devices)'.format(
            total_time_str, total_time * num_devices / len(dataset),
            num_devices))
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        'Model inference time: {} ({} s / img per device, on {} devices)'.
        format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        ))

    predictions = _accumulate_predictions_from_multiple_gpus(
        predictions, multi_gpu_infer)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, 'predictions.pth'))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
