from pathlib import Path
import json
import os
import shutil

def filter_dataset(original_path, output_path, target_dataset):

    with open(original_path / 'annotations' / f"instances_{target_dataset}.json", "r") as file:
        labels_data = json.load(file)

    # filter images
    images_to_keep = []
    # only keep clear, daytime images
    for i in range(len(labels_data['images'])):
        if (labels_data['images'][i]['attributes']['weather'] == 'partly cloudy' or labels_data['images'][i]['attributes']['weather'] == 'overcast') and labels_data['images'][i]['attributes']['timeofday'] == 'daytime':
            images_to_keep.append(labels_data['images'][i]['id'])
    images_to_keep = set(images_to_keep)
    print('image num: ', len(images_to_keep))

    indices_to_remove = []
    for i in range(len(labels_data['images'])):
        if labels_data['images'][i]['id'] not in images_to_keep:
            indices_to_remove.append(i)
    filtered_images = [image for i, image in enumerate(labels_data['images']) if i not in indices_to_remove]
    labels_data['images'] = filtered_images
    
    # filter annotations
    indices_to_remove = []
    for i in range(len(labels_data['annotations'])):
        if labels_data['annotations'][i]['image_id'] not in images_to_keep:
            indices_to_remove.append(i)
    filtered_annotations = [annotation for i, annotation in enumerate(labels_data['annotations']) if i not in indices_to_remove]
    labels_data['annotations'] = filtered_annotations

    # copy images
    for image in labels_data['images']:
        filename = image['file_name']

        src_dir = original_path / 'images' / target_dataset

        matches = list(src_dir.glob(f"{filename}"))
        if not matches:
            raise FileNotFoundError(f"No file found for image_id={filename} in {src_dir}")
        
        src_file = matches[0]
        dest_dir = output_path / 'images' / target_dataset
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(src_file, os.path.join(dest_dir, os.path.basename(src_file)))

    # save filtered labels
    os.makedirs(output_path / 'annotations', exist_ok=True)
    with open(output_path / 'annotations' / f"instances_{target_dataset}.json", "w") as file:
        json.dump(labels_data, file)


if __name__ == "__main__":

    original_path = Path("./data/bdd100k")
    output_path   = Path("./data/bdd100k_cloudy")
    target_datasets = ['train', 'val']
    # label_bound = 10
    
    for target_dataset in target_datasets:
        print(f"Processing {target_dataset}")
        filter_dataset(original_path, output_path, target_dataset)