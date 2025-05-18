import json
import os
import glob
from pathlib import Path

def convert_coco_to_yolo(coco_json_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading annotations from {coco_json_path}...")
    with open(coco_json_path, 'r') as f:
        data = json.load(f)
    
    id_to_filename = {}
    for img in data['images']:
        id_to_filename[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }
    
    if 'annotations' in data:
        # Standard COCO format with separate annotations list
        annotations_by_image = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
    else:
        annotations_by_image = {}
        for img in data['images']:
            img_id = img['id']
            if 'annotations' in img:
                annotations_by_image[img_id] = img['annotations']
    
    processed_count = 0
    for img_id, img_info in id_to_filename.items():
        file_name = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        file_stem = os.path.splitext(file_name)[0]
        label_path = os.path.join(output_dir, file_stem + '.txt')
        
        if img_id in annotations_by_image:
            with open(label_path, 'w') as f:
                for ann in annotations_by_image[img_id]:
                    # Get class ID (YOLO uses 0-indexed classes)
                    class_id = ann.get('class_id', 1) - 1  # Default to class 0 if not specified
                    
                    bbox = ann.get('bbox', [0, 0, 0, 0])
                    x, y, w, h = bbox
                    
                    if w <= 0 or h <= 0:
                        continue
                    
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w_norm = w / width
                    h_norm = h / height
                    
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    w_norm = max(0, min(1, w_norm))
                    h_norm = max(0, min(1, h_norm))
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            
            processed_count += 1
        else:
            with open(label_path, 'w') as f:
                pass
    
    print(f"Conversion complete: {processed_count} images with annotations converted. Labels saved in '{output_dir}'")

def convert_coco_files(annotation_dir, dataset_dir):
    splits = ['train', 'val', 'test']
    split_files_found = False
    
    for split in splits:
        json_path = os.path.join(annotation_dir, f"{split}.json")
        if os.path.exists(json_path):
            print(f"Processing {split} split...")
            output_dir = os.path.join(dataset_dir, split, 'labels')
            convert_coco_to_yolo(json_path, output_dir)
            split_files_found = True
    
    if not split_files_found:
        combined_path = os.path.join(annotation_dir, "combined.json")
        if os.path.exists(combined_path):
            print("Processing combined annotation file...")
            
            with open(combined_path, 'r') as f:
                data = json.load(f)
            
            if 'splits' in data:
                for split in splits:
                    if split in data['splits']:
                        print(f"Processing {split} split from combined file...")
                        
                        temp_data = {
                            'images': [],
                            'categories': data.get('categories', []),
                            'info': data.get('info', {})
                        }
                        
                        split_filenames = set(data['splits'][split])
                        for img in data['images']:
                            if img['file_name'] in split_filenames:
                                temp_data['images'].append(img)
                        
                        output_dir = os.path.join(dataset_dir, split, 'labels')
                        
                        temp_json = f"temp_{split}.json"
                        with open(temp_json, 'w') as f:
                            json.dump(temp_data, f)
                        
                        convert_coco_to_yolo(temp_json, output_dir)
                        
                        os.remove(temp_json)
            else:
                print("No split information found. Converting all to train...")
                output_dir = os.path.join(dataset_dir, 'train', 'labels')
                convert_coco_to_yolo(combined_path, output_dir)
        else:
            print("No COCO annotation files found!")

if __name__ == "__main__":
    dataset_dir = "nn_dataset"
    annotation_dir = os.path.join(dataset_dir, "annotations")
    
    convert_coco_files(annotation_dir, dataset_dir)
    
    print("Conversion completed!")