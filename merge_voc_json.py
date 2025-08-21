import json
import os

def merge_coco_jsons(json_paths, output_path):
    merged_data = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    img_id_offset = 0
    ann_id_offset = 0

    print(f"총 {len(json_paths)}개의 JSON 파일을 병합합니다...")

    for file_path in json_paths:
        print(f"Processing: {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)

        if not merged_data['categories']:
            merged_data['categories'] = data['categories']

        id_map = {}

        for img in data['images']:
            original_img_id = img['id']
            new_img_id = original_img_id + img_id_offset
            id_map[original_img_id] = new_img_id
            img['id'] = new_img_id
            merged_data['images'].append(img)

        for ann in data['annotations']:
            ann['id'] = ann['id'] + ann_id_offset
            ann['image_id'] = id_map[ann['image_id']]
            merged_data['annotations'].append(ann)

        img_id_offset += max([img['id'] for img in data['images']]) + 1
        ann_id_offset += max([ann['id'] for ann in data['annotations']]) + 1

    print(f"병합 완료. 총 이미지: {len(merged_data['images'])}, 총 어노테이션: {len(merged_data['annotations'])}")

    with open(output_path, 'w') as f:
        json.dump(merged_data, f)
    print(f"결과가 '{output_path}'에 저장되었습니다.")

# --- 스크립트 실행 부분 ---
if __name__ == '__main__':
    base_path = 'data/voc_10/annotations/'

    # 1. 병합할 학습용 JSON 파일 목록
    train_files = [
        base_path + 'instances_train2007.json',
        base_path + 'instances_val2007.json',
        base_path + 'instances_train2012.json',
        base_path + 'instances_val2012.json',
    ]
    # 2. 저장될 통합 JSON 파일 이름
    output_train_file = base_path + 'instances_train.json'

    # 3. 병합 함수 실행
    merge_coco_jsons(train_files, output_train_file)
    
    base_path = 'data/voc/annotations/'

    # 1. 병합할 학습용 JSON 파일 목록
    train_files = [
        base_path + 'instances_train2007.json',
        base_path + 'instances_val2007.json',
        base_path + 'instances_train2012.json',
        base_path + 'instances_val2012.json',
    ]
    # 2. 저장될 통합 JSON 파일 이름
    output_train_file = base_path + 'instances_train.json'

    # 3. 병합 함수 실행
    merge_coco_jsons(train_files, output_train_file)