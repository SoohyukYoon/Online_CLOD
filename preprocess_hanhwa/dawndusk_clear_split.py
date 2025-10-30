from pathlib import Path
import json
import os
import shutil

def filter_dataset(original_path, output_base_path, target_dataset):

    with open(original_path / 'annotations' / f"instances_{target_dataset}.json", "r") as file:
        labels_data = json.load(file)

    # filter images
    images_to_keep = []
    # only keep clear, dawn/dusk images
    for img in labels_data['images']:
        attrs = img['attributes']
        if attrs['weather_coarse'] == 'clear' and attrs['timeofday_coarse'] == 'dawn/dusk':
            images_to_keep.append(img['id'])

    images_to_keep = set(images_to_keep)
    print(f"[{target_dataset}] total filtered image num: {len(images_to_keep)}")

    filtered_images = [img for img in labels_data['images'] if img['id'] in images_to_keep]
    labels_data['images'] = filtered_images

    # Split into two halves
    half = len(filtered_images) // 2
    split_sets = {
        "hanhwa_shift_dawndusk_clear1": filtered_images[:half],
        "hanhwa_shift_dawndusk_clear2": filtered_images[half:]
    }

    # Process each half separately
    for split_name, image_subset in split_sets.items():
        print(f"  -> Saving {len(image_subset)} images to {split_name}")

        output_path = output_base_path.parent / split_name
        os.makedirs(output_path / 'images' / target_dataset, exist_ok=True)
        os.makedirs(output_path / 'annotations', exist_ok=True)

        # Copy images
        src_dir = original_path / 'images' / target_dataset
        for image in image_subset:
            filename = image['file_name']
            src_file = src_dir / filename
            if not src_file.exists():
                raise FileNotFoundError(f"No file found for {filename} in {src_dir}")
            dest_file = output_path / 'images' / target_dataset / filename
            shutil.copy(src_file, dest_file)

        # Save filtered labels
        subset_labels = labels_data.copy()
        subset_labels['images'] = image_subset
        with open(output_path / 'annotations' / f"instances_{target_dataset}.json", "w") as file:
            json.dump(subset_labels, file)


if __name__ == "__main__":

    original_path = Path("./data/shift")
    output_path   = Path("./data/hanhwa_shift_dawndusk_clear")
    target_datasets = ['train', 'val']

    for target_dataset in target_datasets:
        print(f"Processing {target_dataset}")
        filter_dataset(original_path, output_path, target_dataset)