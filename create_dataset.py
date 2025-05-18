import os
import random
import json
import math
import numpy as np
from PIL import Image
from collections import defaultdict
from datetime import datetime

try:
    resample = Image.Resampling.BICUBIC
except AttributeError:
    resample = Image.BICUBIC

def load_images_from_folder(folder, filetypes={'.png', '.jpg', '.jpeg'}):
    paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in filetypes):
                paths.append(os.path.join(root, file))
    return paths

def balance_object_classes(object_folder):
    class_to_images = defaultdict(list)
    for cls_name in sorted(os.listdir(object_folder)):
        cls_path = os.path.join(object_folder, cls_name)
        if os.path.isdir(cls_path):
            for file in os.listdir(cls_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    class_to_images[cls_name].append(os.path.join(cls_path, file))
    return class_to_images

def paste_object_with_rotation(background, object_img, position, angle):
    rotated_obj = object_img.rotate(angle, expand=True, resample=resample)
    
    background.paste(rotated_obj, position, rotated_obj)
    return background, rotated_obj

def create_binary_mask(mask, object_img, position):
    alpha = object_img.getchannel('A')
    binary = alpha.point(lambda p: 255 if p > 0 else 0)
    mask.paste(binary, position, binary)
    return mask

def generate_coco_annotation(image_id, file_name, width, height, annotations):
    return {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
        "annotations": annotations
    }

def calculate_rotated_dimensions(width, height, angle_degrees):
    angle_rad = math.radians(angle_degrees)
    cos_a = abs(math.cos(angle_rad))
    sin_a = abs(math.sin(angle_rad))
    
    new_width = int(width * cos_a + height * sin_a)
    new_height = int(width * sin_a + height * cos_a)
    
    return new_width, new_height

def calculate_tight_bbox_from_mask(mask, position, object_size):
    mask_array = np.array(mask)
    
    y_indices, x_indices = np.where(mask_array > 0)
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        return [position[0], position[1], object_size[0], object_size[1]]
    
    min_x, max_x = np.min(x_indices), np.max(x_indices)
    min_y, max_y = np.min(y_indices), np.max(y_indices)
    
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    
    return [int(min_x), int(min_y), int(width), int(height)]

def generate_dataset(
    background_folder='random_images',
    object_folder='objects',  
    output_folder='dataset',
    image_count=100
):
    os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'annotations'), exist_ok=True)

    backgrounds = load_images_from_folder(background_folder)
    object_classes = balance_object_classes(object_folder)

    class_name_to_id = {name: i+1 for i, name in enumerate(sorted(object_classes))}
    id_to_class_name = {v: k for k, v in class_name_to_id.items()}

    all_annotations = []
    image_id = 0

    for i in range(image_count):
        bg_path = random.choice(backgrounds)
        background = Image.open(bg_path).convert('RGB')
        width, height = background.size
        mask = Image.new('L', (width, height), 0)

        annotations = []

        cls = random.choice(list(object_classes.keys()))
        obj_path = random.choice(object_classes[cls])
        object_img = Image.open(obj_path).convert('RGBA')

        angle = random.randint(0, 359)
        
        obj_width, obj_height = object_img.size
        rotated_width, rotated_height = calculate_rotated_dimensions(obj_width, obj_height, angle)
        
        scale_factor = random.uniform(0.05, 0.8)
        
        width_scale = (width * scale_factor) / rotated_width
        height_scale = (height * scale_factor) / rotated_height
        
        final_scale = min(width_scale, height_scale)
        
        new_width = int(obj_width * final_scale)
        new_height = int(obj_height * final_scale)
        
        object_img = object_img.resize((new_width, new_height), resample)
        
        rotated_width, rotated_height = calculate_rotated_dimensions(new_width, new_height, angle)
        
        max_x = max(1, width - rotated_width)
        max_y = max(1, height - rotated_height)
        
        x_pos = random.randint(0, max(0, max_x))
        y_pos = random.randint(0, max(0, max_y))
        
        x_offset = (rotated_width - new_width) // 2
        y_offset = (rotated_height - new_height) // 2
        position = (x_pos - x_offset, y_pos - y_offset)
        
        background, rotated_obj = paste_object_with_rotation(background, object_img, position, angle)
        
        mask = create_binary_mask(mask, rotated_obj, position)
        
        tight_bbox = calculate_tight_bbox_from_mask(mask, position, (rotated_width, rotated_height))
        
        annotations.append({
            "class_id": class_name_to_id[cls],
            "class_name": cls,
            "bbox": tight_bbox,
            "rotation_angle": angle
        })

        img_name = f'image_{i+1:04d}.jpg'
        mask_name = f'mask_{i+1:04d}.png'
        background.save(os.path.join(output_folder, 'images', img_name))
        mask.save(os.path.join(output_folder, 'masks', mask_name))

        coco_data = generate_coco_annotation(
            image_id=image_id,
            file_name=img_name,
            width=width,
            height=height,
            annotations=annotations
        )
        all_annotations.append(coco_data)
        image_id += 1

        print(f"[{i+1}/{image_count}] Image saved with object: {cls}, rotation: {angle}°")

    # COCO format annotations
    with open(os.path.join(output_folder, 'annotations', 'annotations.json'), 'w') as f:
        json.dump({
            "info": {
                "description": "Synthetic Dataset with Rotated Objects",
                "version": "1.0",
                "date_created": datetime.now().isoformat()
            },
            "categories": [
                {"id": class_id, "name": class_name}
                for class_name, class_id in class_name_to_id.items()
            ],
            "images": all_annotations
        }, f, indent=2)

    print("Dataset generation complete.")

def visualize_dataset_sample(dataset_folder, num_samples=5):
    import cv2
    import random
    
    vis_folder = os.path.join(dataset_folder, 'visualization')
    os.makedirs(vis_folder, exist_ok=True)
    
    annotation_path = os.path.join(dataset_folder, 'annotations', 'annotations.json')
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    images = data['images']
    
    if num_samples < len(images):
        samples = random.sample(images, num_samples)
    else:
        samples = images
    
    for sample in samples:
        file_name = sample['file_name']
        image_path = os.path.join(dataset_folder, 'images', file_name)
        
        img = cv2.imread(image_path)
        
        for ann in sample['annotations']:
            x, y, w, h = [int(val) for val in ann['bbox']]  # Ensure all values are ints
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            class_name = ann.get('class_name', f"Class {ann['class_id']}")
            cv2.putText(img, class_name, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        vis_path = os.path.join(vis_folder, f"vis_{file_name}")
        cv2.imwrite(vis_path, img)
        print(f"Saved visualization: {vis_path}")
    
    print(f"✅ Generated {len(samples)} visualization samples in {vis_folder}")

if __name__ == "__main__":
    #dataset size is dependent on the random_images and objects folders size
    background_folder = 'random_images'
    object_folder = 'objects'
    
    bg_files = load_images_from_folder(background_folder)
    num_back_files = len(bg_files)
    
    object_classes = balance_object_classes(object_folder)
    total_obj_files = sum(len(files) for files in object_classes.values())
    
    if num_back_files == 0:
        print(f"Error: No background images found in {background_folder}")
        exit(1)
    
    if total_obj_files == 0:
        print(f"Error: No object images found in subfolders of {object_folder}")
        print("Make sure you have class subfolders with images inside the objects folder.")
        exit(1)
    
    dataset_size = max(1, int(num_back_files * total_obj_files / 2))
    
    print(f"Found {num_back_files} background images and {total_obj_files} object images")
    print(f"Will generate {dataset_size} synthetic images")
    
    generate_dataset(
        background_folder=background_folder,
        object_folder=object_folder,
        output_folder='dataset',
        image_count=dataset_size
    )
    
    print("\nGenerating visualization samples to verify bounding box accuracy...")
    visualize_dataset_sample('dataset', num_samples=10)