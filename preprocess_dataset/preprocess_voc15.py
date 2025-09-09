from pathlib import Path
import json
import os
import shutil

def filter_dataset(original_path, output_path, target_dataset, label_bound):

    with open(original_path / 'annotations' / f"instances_{target_dataset}.json", "r") as file:
        labels_data = json.load(file)

    # filter annotations
    indices_to_remove = []
    for i in range(len(labels_data['annotations'])):
        if labels_data['annotations'][i]['category_id'] > label_bound:
            indices_to_remove.append(i)
    filtered_annotations = [annotation for i, annotation in enumerate(labels_data['annotations']) if i not in indices_to_remove]
    labels_data['annotations'] = filtered_annotations

    # filter categories
    indices_to_remove = []
    for i in range(len(labels_data['categories'])):
        if labels_data['categories'][i]['id'] > label_bound:
            indices_to_remove.append(i)
    filtered_categories = [category for i, category in enumerate(labels_data['categories']) if i not in indices_to_remove]
    labels_data['categories'] = filtered_categories

    # filter images
    images_to_keep = []
    for i in range(len(labels_data['annotations'])):
        images_to_keep.append(labels_data['annotations'][i]['image_id'])
    images_to_keep = set(images_to_keep)

    indices_to_remove = []
    for i in range(len(labels_data['images'])):
        if labels_data['images'][i]['id'] not in images_to_keep:
            indices_to_remove.append(i)
    filtered_images = [image for i, image in enumerate(labels_data['images']) if i not in indices_to_remove]
    labels_data['images'] = filtered_images

    # copy images
    for image in labels_data['images']:
        filename = image['file_name']
        if "test" in filename:
            split = "val"
        else:
            split = "train"
        src_dir = original_path / 'images' / split / filename        
        if not src_dir.exists():
            raise FileNotFoundError(f"Image file {src_dir} does not exist.")
        dest_dir = output_path / 'images' / split / filename
        os.makedirs(dest_dir.parent, exist_ok=True)
        shutil.copy(src_dir, dest_dir)

    # save filtered labels
    os.makedirs(output_path / 'annotations', exist_ok=True)
    with open(output_path / 'annotations' / f"instances_{target_dataset}.json", "w") as file:
        json.dump(labels_data, file)


if __name__ == "__main__":

    original_path = Path("./data/voc")
    output_path   = Path("./data/voc_15")
    # target_datasets = ['train2007', 'val2007', 'test2007', 'train2012', 'val2012']
    target_datasets = ['train', 'val']
    label_bound = 15
    
    for target_dataset in target_datasets:
        print(f"Processing {target_dataset}")
        filter_dataset(original_path, output_path, target_dataset, label_bound)