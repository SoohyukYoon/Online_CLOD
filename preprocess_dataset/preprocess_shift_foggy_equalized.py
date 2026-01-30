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
    
    # Calculate histogram using numpy
    histogram, _ = np.histogram(y_channel, bins=256, range=(0, 256))
    
    # Calculate cumulative distribution function (CDF)
    cdf = np.cumsum(histogram)
    
    # Find min and max CDF values
    min_cdf = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
    max_cdf = cdf[-1]
    cdf_range = max_cdf - min_cdf
    
    # Normalize CDF to 0-255 range and create mapping
    if cdf_range > 0:
        equalized_mapping = (255 * (cdf - min_cdf) / cdf_range).astype(np.uint8)
    else:
        equalized_mapping = np.arange(256, dtype=np.uint8)
    
    # Apply histogram equalization to Y channel using vectorized operations
    equalized_y = equalized_mapping[y_channel]
    
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
        if labels_data['images'][i]['attributes']['weather_coarse'] == 'foggy' and (labels_data['images'][i]['attributes']['timeofday_coarse'] == 'daytime'):
            images_to_keep.append(labels_data['images'][i]['id'])
    images_to_keep = set(images_to_keep)
    print('image num: ', len(images_to_keep))

    filtered_images = [image for image in labels_data['images'] if image['id'] in images_to_keep]
    labels_data['images'] = filtered_images
    
    # filter annotations
    filtered_annotations = [annotation for annotation in labels_data['annotations'] if annotation['image_id'] in images_to_keep]
    labels_data['annotations'] = filtered_annotations

    # copy and equalize images
    for image in labels_data['images']:
        filename = image['file_name']

        src_file = original_path / 'images' / target_dataset / filename
        if not src_file.exists():
            raise FileNotFoundError(f"No file found for image_id={filename} in {src_file.parent}")
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
    output_path   = Path("./data/shift_foggy_equalized")
    target_datasets = ['train', 'val']
    # label_bound = 10
    
    for target_dataset in target_datasets:
        print(f"Processing {target_dataset}")
        filter_dataset(original_path, output_path, target_dataset)

