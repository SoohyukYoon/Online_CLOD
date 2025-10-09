import json

def convert_str_to_int(json_file_path):
    count = 0
    datalist = json.load(open(json_file_path, 'r'))
    for item in datalist['annotations']:
        # if isinstance(item['id'], str):
        #     item['id'] = int(item['id'])
        item['id'] = count
        count += 1
    
    with open(json_file_path, 'w') as f:
        json.dump(datalist, f, indent=4)

if __name__ == "__main__":
    json_file_path = "data/shift/annotations/instances_val.json"
    convert_str_to_int(json_file_path)
    print(f"Converted string IDs to integers in {json_file_path}.")
    
    json_file_path = "data/shift/annotations/instances_train.json"
    convert_str_to_int(json_file_path)
    print(f"Converted string IDs to integers in {json_file_path}.")
    
    
    json_file_path = "data/shift_source/annotations/instances_val.json"
    convert_str_to_int(json_file_path)
    print(f"Converted string IDs to integers in {json_file_path}.")
    
    json_file_path = "data/shift_source/annotations/instances_train.json"
    convert_str_to_int(json_file_path)
    print(f"Converted string IDs to integers in {json_file_path}.")
    
    
    json_file_path = "data/shift_cloudy/annotations/instances_val.json"
    convert_str_to_int(json_file_path)
    print(f"Converted string IDs to integers in {json_file_path}.")
    
    json_file_path = "data/shift_cloudy/annotations/instances_train.json"
    convert_str_to_int(json_file_path)
    print(f"Converted string IDs to integers in {json_file_path}.")
    
    
    json_file_path = "data/shift_dawndusk/annotations/instances_val.json"
    convert_str_to_int(json_file_path)
    print(f"Converted string IDs to integers in {json_file_path}.")
    
    json_file_path = "data/shift_dawndusk/annotations/instances_train.json"
    convert_str_to_int(json_file_path)
    print(f"Converted string IDs to integers in {json_file_path}.")
    
    
    json_file_path = "data/shift_foggy/annotations/instances_val.json"
    convert_str_to_int(json_file_path)
    print(f"Converted string IDs to integers in {json_file_path}.")
    
    json_file_path = "data/shift_foggy/annotations/instances_train.json"
    convert_str_to_int(json_file_path)
    print(f"Converted string IDs to integers in {json_file_path}.")
    
    
    json_file_path = "data/shift_night/annotations/instances_val.json"
    convert_str_to_int(json_file_path)
    print(f"Converted string IDs to integers in {json_file_path}.")
    
    json_file_path = "data/shift_night/annotations/instances_train.json"
    convert_str_to_int(json_file_path)
    print(f"Converted string IDs to integers in {json_file_path}.")
    
    
    json_file_path = "data/shift_rainy/annotations/instances_val.json"
    convert_str_to_int(json_file_path)
    print(f"Converted string IDs to integers in {json_file_path}.")
    
    json_file_path = "data/shift_rainy/annotations/instances_train.json"
    convert_str_to_int(json_file_path)
    print(f"Converted string IDs to integers in {json_file_path}.")
    
    
    json_file_path = "data/shift_overcast/annotations/instances_val.json"
    convert_str_to_int(json_file_path)
    print(f"Converted string IDs to integers in {json_file_path}.")
    
    json_file_path = "data/shift_overcast/annotations/instances_train.json"
    convert_str_to_int(json_file_path)
    print(f"Converted string IDs to integers in {json_file_path}.")