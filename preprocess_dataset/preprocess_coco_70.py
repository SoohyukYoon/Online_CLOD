from pathlib import Path
from tqdm import tqdm
import json
import os
import shutil


def filter_dataset(original_path, output_path, target_dataset, target_labels):

    with open(original_path / 'annotations' / f"instances_{target_dataset}.json", "r") as file:
        labels_data = json.load(file)
        
        
    # filter annotations
    print("Filtering annotations")
    
    for i in tqdm(range(len(labels_data['annotations'])-1, -1, -1)):
        if labels_data['annotations'][i]['category_id'] not in target_labels:
            labels_data['annotations'].pop(i)


    # filter categories
    print("Filtering categories")
    
    for i in tqdm(range(len(labels_data['categories'])-1, -1, -1)):
        if labels_data['categories'][i]['id'] not in target_labels:
            labels_data['categories'].pop(i)


    
    # filter images
    print("Filtering images based on annotations")
    
    images_to_keep = []
    for i in tqdm(range(len(labels_data['annotations']))):
        images_to_keep.append(labels_data['annotations'][i]['image_id'])
    images_to_keep = set(images_to_keep)

    indices_to_remove = []
    for i in tqdm(range(len(labels_data['images']))):
        if labels_data['images'][i]['id'] not in images_to_keep:
            indices_to_remove.append(i)
    filtered_images = [image for i, image in enumerate(labels_data['images']) if i not in indices_to_remove]
    labels_data['images'] = filtered_images



    # copy images
    print("Copying images to output directory")
    for image_id in tqdm(images_to_keep):

        image_id = str(image_id).zfill(12)

        src_dir = original_path / 'images' / target_dataset

        matches = list(src_dir.glob(f"{image_id}.*"))
        if not matches:
            raise FileNotFoundError(f"No file found for image_id={image_id} in {src_dir}")
        
        src_file = matches[0]
        dest_dir = output_path / 'images' / target_dataset
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(src_file, os.path.join(dest_dir, os.path.basename(src_file)))
        
        

    # save filtered labels
    os.makedirs(output_path / 'annotations', exist_ok=True)
    with open(output_path / 'annotations' / f"instances_{target_dataset}.json", "w") as file:
        json.dump(labels_data, file)


if __name__ == "__main__":

    original_path = Path("./data/coco")
    output_path   = Path("./data/coco_70")
    target_datasets = ['train2017', 'val2017']
    
    target_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
                     46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79#,80,81,82,84,85,86,87,88,89,90
                     ]
    
    for target_dataset in target_datasets:
        print(f"Processing {target_dataset}")
        filter_dataset(original_path, output_path, target_dataset, target_labels)