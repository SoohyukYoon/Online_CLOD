from pathlib import Path
import json
import os
import shutil
import numpy as np
from PIL import Image
import cv2

def equalize_image(image_path):
    """
    Apply histogram equalization to an RGB image using YCrCb color space.
    The Y channel is equalized, then converted back to RGB.
    
    Args:
        image_path: Path to the input image (PIL Image or file path)
    
    Returns:
        PIL Image: Equalized RGB image
    """
    # Load image if it's a path, otherwise assume it's already a PIL Image
    if isinstance(image_path, (str, Path)):
        img = Image.open(image_path)
    else:
        img = image_path
    
    # Convert PIL RGB to numpy array
    img_array = np.array(img)
    
    # Convert RGB to YCrCb
    img_ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
    
    # Extract Y channel
    y_channel = img_ycrcb[:, :, 0]
    img_w, img_h = y_channel.shape[:2]
    
    # Calculate histogram
    histogram_data = {}
    for pixel_value in range(256):
        histogram_data[pixel_value] = 0
    
    for i in range(img_w):
        for j in range(img_h):
            pixel_value = int(y_channel[i, j])
            histogram_data[pixel_value] += 1
    
    # Calculate cumulative distribution function (CDF)
    cdf = {}
    cumulative = 0
    total_pixels = img_w * img_h
    for pixel_value in range(256):
        cumulative += histogram_data[pixel_value]
        cdf[pixel_value] = cumulative
    
    # Find min and max CDF values
    min_cdf = min(cdf.values())
    max_cdf = max(cdf.values())
    cdf_range = max_cdf - min_cdf
    
    # Normalize CDF to 0-255 range and create mapping
    equalized_mapping = {}
    for pixel_value in range(256):
        if cdf_range > 0:
            equalized_mapping[pixel_value] = int(255 * (cdf[pixel_value] - min_cdf) / cdf_range)
        else:
            equalized_mapping[pixel_value] = pixel_value
    
    # Apply histogram equalization to Y channel
    equalized_y = np.zeros((img_w, img_h), dtype=np.uint8)
    for i in range(img_w):
        for j in range(img_h):
            original_value = int(y_channel[i, j])
            equalized_y[i, j] = equalized_mapping[original_value]
    
    # Replace Y channel with equalized version
    img_ycrcb[:, :, 0] = equalized_y
    
    # Convert back to RGB
    img_rgb = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB)
    
    # Convert back to PIL Image
    equalized_img = Image.fromarray(img_rgb)
    
    return equalized_img

def filter_dataset(original_path, output_path, target_dataset):

    with open(original_path / 'annotations' / f"instances_{target_dataset}.json", "r") as file:
        labels_data = json.load(file)

    # filter images
    images_to_keep = []
    # only keep clear, daytime images
    for i in range(len(labels_data['images'])):
        if labels_data['images'][i]['attributes']['weather_coarse'] == 'clear' and labels_data['images'][i]['attributes']['timeofday_coarse'] == 'night':
            images_to_keep.append(labels_data['images'][i]['id'])
    images_to_keep = set(images_to_keep)
    print('image num: ', len(images_to_keep))

    filtered_images = [image for image in labels_data['images'] if image['id'] in images_to_keep]
    labels_data['images'] = filtered_images
    
    # filter annotations
    print("filtering annotations")
    filtered_annotations = [annotation for annotation in labels_data['annotations'] if annotation['image_id'] in images_to_keep]
    labels_data['annotations'] = filtered_annotations

    # copy and equalize images
    print("copying images")
    count = 0 
    for image in labels_data['images']:
        print(count)
        if count % 1000 == 0: 
            print("copied count: ", count)
        count += 1 
        filename = image['file_name']

        src_dir = original_path / 'images' / target_dataset

        matches = list(src_dir.glob(f"{filename}"))
        if not matches:
            raise FileNotFoundError(f"No file found for image_id={filename} in {src_dir}")
        
        src_file = matches[0]
        dest_dir = output_path / 'images' / target_dataset
        os.makedirs(dest_dir, exist_ok=True)
        
        # Equalize image and save
        equalized_img = equalize_image(src_file)
        dest_file = os.path.join(dest_dir, os.path.basename(src_file))
        equalized_img.save(dest_file)

    # save filtered labels
    os.makedirs(output_path / 'annotations', exist_ok=True)
    with open(output_path / 'annotations' / f"instances_{target_dataset}.json", "w") as file:
        json.dump(labels_data, file)


if __name__ == "__main__":

    original_path = Path("./data/shift")
    output_path   = Path("./data/shift_night_equalized")
    target_datasets = ['train', 'val']
    # label_bound = 10
    
    for target_dataset in target_datasets:
        print(f"Processing {target_dataset}")
        filter_dataset(original_path, output_path, target_dataset)

