#doesnt parse images correctly, leaves black background
import random
import json
import math
import shutil
from PIL import Image, ImageOps, ImageEnhance
from datetime import datetime
import os

try:
    resample = Image.Resampling.BICUBIC
except AttributeError:
    resample = Image.BICUBIC

def load_dataset(dataset_folder):
    images_folder = os.path.join(dataset_folder, 'images')
    masks_folder = os.path.join(dataset_folder, 'masks')
    annotations_file = os.path.join(dataset_folder, 'annotations', 'annotations.json')
    
    if not (os.path.exists(images_folder) and 
            os.path.exists(masks_folder) and 
            os.path.exists(annotations_file)):
        raise FileNotFoundError(f"Dataset structure not found in {dataset_folder}")
    
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    return {
        'images_folder': images_folder,
        'masks_folder': masks_folder,
        'annotations': annotations
    }

def create_augmented_dataset(source_dataset_folder, output_folder, 
                           augmentation_factor=2, 
                           rotation_angles=[-30, -15, 15, 30],
                           scale_factors=[0.8, 0.9, 1.1, 1.2],
                           flip_horizontal=True,
                           brightness_factors=[0.8, 1.2],
                           contrast_factors=[0.8, 1.2]):
    
    os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'annotations'), exist_ok=True)
    
    dataset = load_dataset(source_dataset_folder)
    source_images_folder = dataset['images_folder']
    source_masks_folder = dataset['masks_folder']
    annotations = dataset['annotations']
    
    print("Copying original dataset...")
    image_files = os.listdir(source_images_folder)
    for filename in image_files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            shutil.copy2(
                os.path.join(source_images_folder, filename),
                os.path.join(output_folder, 'images', filename)
            )
            
            mask_filename = filename.replace('image_', 'mask_').replace('.jpg', '.png')
            mask_path = os.path.join(source_masks_folder, mask_filename)
            if os.path.exists(mask_path):
                shutil.copy2(
                    mask_path,
                    os.path.join(output_folder, 'masks', mask_filename)
                )
    
    original_image_count = len(image_files)
    target_image_count = original_image_count * augmentation_factor
    transformations_needed = target_image_count - original_image_count
    
    print(f"Original dataset: {original_image_count} images")
    print(f"Target dataset: {target_image_count} images")
    print(f"Will generate {transformations_needed} new augmented images")
    
    new_image_id = original_image_count
    new_annotations = annotations.copy()
    
    all_transformations = []
    
    if rotation_angles:
        for angle in rotation_angles:
            all_transformations.append(('rotate', angle))
    
    if scale_factors:
        for scale in scale_factors:
            all_transformations.append(('scale', scale))
    
    if flip_horizontal:
        all_transformations.append(('flip_h', None))
    
    if brightness_factors:
        for factor in brightness_factors:
            all_transformations.append(('brightness', factor))
    
    if contrast_factors:
        for factor in contrast_factors:
            all_transformations.append(('contrast', factor))
    
    generated_count = 0
    
    while generated_count < transformations_needed:
        source_image_idx = random.randint(0, original_image_count - 1)
        source_image_name = f'image_{source_image_idx+1:04d}.jpg'
        source_mask_name = f'mask_{source_image_idx+1:04d}.png'
        
        source_image_path = os.path.join(source_images_folder, source_image_name)
        source_mask_path = os.path.join(source_masks_folder, source_mask_name)
        
        if not (os.path.exists(source_image_path) and os.path.exists(source_mask_path)):
            continue
        
        source_annotation = None
        for img in annotations['images']:
            if img['file_name'] == source_image_name:
                source_annotation = img
                break
        
        if source_annotation is None:
            continue
        
        transform_type, transform_value = random.choice(all_transformations)
        
        image = Image.open(source_image_path).convert('RGB')
        mask = Image.open(source_mask_path).convert('L')
        
        width, height = image.size
        
        transformed_image = image.copy()
        transformed_mask = mask.copy()
        
        valid_transform = True
        
        transformed_annotation = {
            "id": new_image_id,
            "file_name": f'image_{new_image_id+1:04d}.jpg',
            "width": width,
            "height": height,
            "annotations": []
        }
        
        for obj in source_annotation['annotations']:
            new_obj = obj.copy()
            
            bbox = obj['bbox']
            x, y, w, h = bbox
            
            if transform_type == 'rotate':
                angle = transform_value
                
                padding = int(max(w, h) * 0.5)
                padded_width = width + padding * 2
                padded_height = height + padding * 2
                
                corners = [
                    transformed_image.getpixel((0, 0)),
                    transformed_image.getpixel((width-1, 0)),
                    transformed_image.getpixel((0, height-1)),
                    transformed_image.getpixel((width-1, height-1))
                ]
                bg_color = tuple(sum(c) // len(corners) for c in zip(*corners))
                
                padded_image = Image.new('RGB', (padded_width, padded_height), bg_color)
                padded_image.paste(transformed_image, (padding, padding))
                
                padded_mask = Image.new('L', (padded_width, padded_height), 0)
                padded_mask.paste(transformed_mask, (padding, padding))
                
                rotated_image = padded_image.rotate(angle, resample=resample)
                rotated_mask = padded_mask.rotate(angle, resample=resample)
                
                mask_array = list(rotated_mask.getdata())
                mask_width = rotated_mask.width
                
                non_zero_pixels = [(i % mask_width, i // mask_width) 
                                  for i, pixel in enumerate(mask_array) if pixel > 0]
                
                if non_zero_pixels:
                    min_x = min(x for x, y in non_zero_pixels)
                    min_y = min(y for x, y in non_zero_pixels)
                    max_x = max(x for x, y in non_zero_pixels)
                    max_y = max(y for x, y in non_zero_pixels)
                    
                    center_x = (min_x + max_x) // 2
                    center_y = (min_y + max_y) // 2
                    
                    crop_left = max(0, center_x - width // 2)
                    crop_top = max(0, center_y - height // 2)
                    
                    crop_right = min(padded_width, crop_left + width)
                    crop_bottom = min(padded_height, crop_top + height)
                    
                    if crop_right - crop_left < width:
                        if crop_left == 0:
                            crop_right = width
                        else:
                            crop_left = crop_right - width
                    
                    if crop_bottom - crop_top < height:
                        if crop_top == 0:
                            crop_bottom = height
                        else:
                            crop_top = crop_bottom - height
                    
                    transformed_image = rotated_image.crop((crop_left, crop_top, crop_right, crop_bottom))
                    transformed_mask = rotated_mask.crop((crop_left, crop_top, crop_right, crop_bottom))
                    
                    if transformed_image.size != (width, height):
                        transformed_image = transformed_image.resize((width, height), resample=resample)
                        transformed_mask = transformed_mask.resize((width, height), resample=resample)
                else:
                    crop_left = (padded_width - width) // 2
                    crop_top = (padded_height - height) // 2
                    transformed_image = rotated_image.crop((crop_left, crop_top, 
                                                         crop_left + width, crop_top + height))
                    transformed_mask = rotated_mask.crop((crop_left, crop_top, 
                                                       crop_left + width, crop_top + height))
                
                new_obj['rotation_applied'] = angle
                
                mask_array = list(transformed_mask.getdata())
                non_zero_indices = [i for i, pixel in enumerate(mask_array) if pixel > 0]
                
                if non_zero_indices:
                    coords = [(i % width, i // width) for i in non_zero_indices]
                    min_x = min(x for x, y in coords)
                    min_y = min(y for x, y in coords)
                    max_x = max(x for x, y in coords)
                    max_y = max(y for x, y in coords)
                    
                    new_obj['bbox'] = [min_x, min_y, max_x - min_x + 1, max_y - min_y + 1]
                
            elif transform_type == 'scale':
                scale = transform_value
                
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                if new_width < 300 or new_height < 300:
                    valid_transform = False
                    break
                
                resized_image = transformed_image.resize((new_width, new_height), resample=resample)
                resized_mask = transformed_mask.resize((new_width, new_height), resample=resample)
                
                transformed_image = Image.new('RGB', (width, height), (0, 0, 0))
                transformed_mask = Image.new('L', (width, height), 0)
                
                pos_x = (width - new_width) // 2
                pos_y = (height - new_height) // 2
                
                transformed_image.paste(resized_image, (pos_x, pos_y))
                transformed_mask.paste(resized_mask, (pos_x, pos_y))
                
                new_obj['bbox'] = [
                    int(x * scale) + pos_x,
                    int(y * scale) + pos_y,
                    int(w * scale),
                    int(h * scale)
                ]
                new_obj['scale_applied'] = scale
                
            elif transform_type == 'flip_h':
                transformed_image = ImageOps.mirror(transformed_image)
                transformed_mask = ImageOps.mirror(transformed_mask)
                
                # Update bounding box
                new_obj['bbox'] = [
                    width - (x + w),  # Flip x-coordinate 
                    y,                # y stays the same
                    w,                # width stays the same
                    h                 # height stays the same
                ]
                new_obj['horizontally_flipped'] = True
                
            elif transform_type == 'brightness':
                factor = transform_value
                enhancer = ImageEnhance.Brightness(transformed_image)
                transformed_image = enhancer.enhance(factor)
                new_obj['brightness_adjusted'] = factor
                
            elif transform_type == 'contrast':
                factor = transform_value
                enhancer = ImageEnhance.Contrast(transformed_image)
                transformed_image = enhancer.enhance(factor)
                new_obj['contrast_adjusted'] = factor
            
            if 'bbox' in new_obj:
                new_x, new_y, new_w, new_h = new_obj['bbox']
                if (new_x < 0 or new_y < 0 or 
                    new_x + new_w > width or 
                    new_y + new_h > height):
                    valid_transform = False
                    break
            
            transformed_annotation['annotations'].append(new_obj)
        
        if valid_transform:
            new_image_name = f'image_{new_image_id+1:04d}.jpg'
            new_mask_name = f'mask_{new_image_id+1:04d}.png'
            
            transformed_image.save(os.path.join(output_folder, 'images', new_image_name))
            transformed_mask.save(os.path.join(output_folder, 'masks', new_mask_name))
            
            new_annotations['images'].append(transformed_annotation)
            
            new_image_id += 1
            generated_count += 1
            
            if generated_count % 10 == 0:
                print(f"Generated {generated_count}/{transformations_needed} augmented images")
    
    with open(os.path.join(output_folder, 'annotations', 'annotations.json'), 'w') as f:
        json.dump(new_annotations, f, indent=2)
    
    print(f"Augmentation complete. Dataset expanded from {original_image_count} to {new_image_id} images.")
    return new_image_id

if __name__ == "__main__":
    source_dataset = "dataset"
    output_dataset = "augmented_dataset"
    
    augmentation_factor = 3  
    
    create_augmented_dataset(
        source_dataset_folder=source_dataset,
        output_folder=output_dataset,
        augmentation_factor=augmentation_factor,
        rotation_angles=[-45, -30, -15, 15, 30, 45],
        scale_factors=[0.7, 0.8, 0.9, 1.1, 1.2],
        flip_horizontal=True,
        brightness_factors=[0.8, 0.9, 1.1, 1.2],
        contrast_factors=[0.8, 0.9, 1.1, 1.2]
    )