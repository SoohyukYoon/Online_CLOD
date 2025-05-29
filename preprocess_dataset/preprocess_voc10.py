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
    for image_id in images_to_keep:

        if '2007' in target_dataset:
            image_id = str(image_id).zfill(6)
        elif '2012' in target_dataset:
            image_id = str(image_id)[:4] + '_' + str(image_id)[4:]

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

    original_path = Path("./data/voc")
    output_path   = Path("./data/voc_10")
    target_datasets = ['train2007', 'val2007', 'test2007', 'train2012', 'val2012']
    label_bound = 10
    
    for target_dataset in target_datasets:
        print(f"Processing {target_dataset}")
        filter_dataset(original_path, output_path, target_dataset, label_bound)