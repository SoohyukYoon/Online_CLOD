import json
import os

# Directory containing your individual JSON files

for split, dir_name in {"Training":'train', "Validation":'val'}.items():
    json_dir = f"aihub_military_synthetic/{split}/labels"

    # Output JSON structure
    combined_json = {
        "info": {},
        "licenses": [],
        "categories": [
            {"supercategory": "ship", "id": 1, "name": "fishing vessel"},
            {"supercategory": "ship", "id": 2, "name": "warship"},
            {"supercategory": "ship", "id": 3, "name": "merchant vessel"},
            {"supercategory": "aircraft", "id": 4, "name": "fixed-wing aircraft"},
            {"supercategory": "aircraft", "id": 5, "name": "rotary-wing aircraft"},
            {"supercategory": "aircraft", "id": 6, "name": "Unmanned Aerial Vehicle"},
            {"supercategory": "extra", "id": 7, "name": "bird"},
            {"supercategory": "extra", "id": 8, "name": "leaflet"},
            {"supercategory": "extra", "id": 9, "name": "waste bomb"},
        ],
        "images": [],
        "annotations": []
    }

    cls_idx_mapping = {11: 1,  # 어선
                        12: 2,  # 군함
                        13: 3,  # 상선
                        21: 4,  # 고정익 유인항공기
                        22: 5,  # 회전익 유인항공기
                        23: 6,  # 무인항공기
                        31: 7,  # 새
                        41: 8,  # 삐라
                        42: 9}  # 오물폭탄
    # Counter for unique IDs
    annotation_id = 1

    # Process each JSON file
    for file in os.listdir(json_dir):
        if file.endswith('.json'):
            file_path = os.path.join(json_dir, file)

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                image_info = data['image']
                image_info['file_name'] = image_info['filename']  # Change key name from "filename" to "file_name"
                del image_info['filename']
                image_info['id'] = image_id = file.split('.')[0]  # Use the filename as the image ID
                meta = data['env']
                image_info['attributes'] = meta
                combined_json["images"].append(image_info)
                annotations = data.get('annotations', [])
                
                for annotation in annotations:
                    annotation_info = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": cls_idx_mapping[annotation['sub_class']],
                        "bbox": annotation["bounding_box"],
                        "iscrowd":0,
                    }
                    combined_json["annotations"].append(annotation_info)

                    annotation_id += 1

    # Save combined JSON to a file
    output_path = f"data/military_synthetic/annotations/instances_{dir_name}.json"
    print(len(combined_json['images']), "images")
    print(len(combined_json['annotations']), "annotations")
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(combined_json, outfile, ensure_ascii=False, indent=4)

    print(f"Combined JSON saved to {output_path}")
