#wanted to make sure that bboxes are correct after transformation from coco to yolo
import cv2
import os
import json
import argparse
import numpy as np
from pathlib import Path

def draw_coco_boxes(image, annotations, color=(0, 0, 255), thickness=2):
    img = image.copy()
    
    for ann in annotations:
        x, y, w, h = ann.get('bbox', [0, 0, 0, 0])
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        if w <= 0 or h <= 0:
            continue
        
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        
        class_id = ann.get('class_id', 0)
        class_name = ann.get('class_name', f"Class {class_id}")
        label = f"{class_name}"
        
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, color, thickness//2 + 1)
    
    return img

def draw_yolo_boxes(image, label_path, color=(0, 255, 0), thickness=2):
    img = image.copy()
    h, w = img.shape[:2]
    
    if not os.path.exists(label_path):
        print(f"Warning: Label file not found: {label_path}")
        return img
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            class_id, x_center, y_center, width, height = parts
            class_id = int(class_id)
            x_center, y_center, width, height = map(float, (x_center, y_center, width, height))
            
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            label = f"Class {class_id}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, color, thickness//2 + 1)
    
    return img

def compare_annotations(image_path, coco_json_path, yolo_label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    image_filename = os.path.basename(image_path)
    image_name = os.path.splitext(image_filename)[0]
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    image_id = None
    image_info = None
    
    for img_info in coco_data['images']:
        if img_info['file_name'] == image_filename:
            image_id = img_info['id']
            image_info = img_info
            break
    
    if image_id is None:
        print(f"Warning: Image {image_filename} not found in COCO annotations")
        return
    
    coco_annotations = []
    
    if 'annotations' in coco_data:
        for ann in coco_data['annotations']:
            if ann['image_id'] == image_id:
                coco_annotations.append(ann)
    elif 'annotations' in image_info:
        coco_annotations = image_info['annotations']
    
    yolo_label_path = os.path.join(yolo_label_dir, f"{image_name}.txt")
    
    img_coco = draw_coco_boxes(img, coco_annotations, color=(0, 0, 255))
    
    img_yolo = draw_yolo_boxes(img, yolo_label_path, color=(0, 255, 0))
    
    img_combined = img.copy()
    img_combined = draw_coco_boxes(img_combined, coco_annotations, color=(0, 0, 255))
    img_combined = draw_yolo_boxes(img_combined, yolo_label_path, color=(0, 255, 0))
    
    h, w = img.shape[:2]
    label_height = 30
    font_scale = 0.7
    
    header = np.zeros((label_height, w, 3), dtype=np.uint8)
    
    cv2.putText(header, "COCO Annotations (Red)", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
    img_coco_with_header = np.vstack([header, img_coco])
    
    header = np.zeros((label_height, w, 3), dtype=np.uint8)
    cv2.putText(header, "YOLO Annotations (Green)", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
    img_yolo_with_header = np.vstack([header, img_yolo])
    
    header = np.zeros((label_height, w, 3), dtype=np.uint8)
    cv2.putText(header, "Combined (Red=COCO, Green=YOLO)", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    img_combined_with_header = np.vstack([header, img_combined])
    
    comparison = np.hstack([img_coco_with_header, img_yolo_with_header])
    
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_coco.jpg"), img_coco_with_header)
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_yolo.jpg"), img_yolo_with_header)
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_combined.jpg"), img_combined_with_header)
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_comparison.jpg"), comparison)
    
    print(f"Comparison images saved to {output_dir}")
    
    return comparison

def batch_compare_annotations(dataset_dir, split="train", num_samples=5):
    images_dir = os.path.join(dataset_dir, split, "images")
    coco_json_path = os.path.join(dataset_dir, "annotations", f"{split}.json")
    yolo_label_dir = os.path.join(dataset_dir, split, "labels")
    output_dir = os.path.join(dataset_dir, "bbox_comparison", split)
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    if not os.path.exists(coco_json_path):
        coco_json_path = os.path.join(dataset_dir, "annotations", "combined.json")
        if not os.path.exists(coco_json_path):
            print(f"Error: No annotation file found for {split} split")
            return
    
    if not os.path.exists(yolo_label_dir):
        print(f"Error: YOLO labels directory not found: {yolo_label_dir}")
        return
    
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(Path(images_dir).glob(f"*{ext}")))
    
    if not image_files:
        print(f"Error: No images found in {images_dir}")
        return
    
    import random
    if len(image_files) > num_samples:
        image_files = random.sample(image_files, num_samples)
    
    for img_path in image_files:
        print(f"Processing {img_path.name}...")
        compare_annotations(str(img_path), coco_json_path, yolo_label_dir, output_dir)
    
    print(f"Processed {len(image_files)} images. Comparison images saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare COCO and YOLO annotations.")
    parser.add_argument("--dataset", default="nn_dataset", help="Dataset directory")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--num", type=int, default=5, help="Number of images to process")
    parser.add_argument("--image", help="Process a specific image (optional)")
    
    args = parser.parse_args()
    
    if args.image:
        image_path = args.image
        split = args.split
        coco_json_path = os.path.join(args.dataset, "annotations", f"{split}.json")
        
        if not os.path.exists(coco_json_path):
            coco_json_path = os.path.join(args.dataset, "annotations", "combined.json")
        
        yolo_label_dir = os.path.join(args.dataset, split, "labels")
        output_dir = os.path.join(args.dataset, "bbox_comparison", split)
        
        compare_annotations(image_path, coco_json_path, yolo_label_dir, output_dir)
    else:
        batch_compare_annotations(args.dataset, args.split, args.num)