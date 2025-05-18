import os
import json
import random
import shutil
from pathlib import Path
import argparse

def split_dataset(
    source_folder='dataset',
    output_folder='nn_dataset',
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    random_seed=42
):

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        raise ValueError("Split ratios must sum to 1.0")
    
    random.seed(random_seed)
    
    splits = ['train', 'val', 'test']
    for split in splits:
        for subdir in ['images', 'masks']:
            os.makedirs(os.path.join(output_folder, split, subdir), exist_ok=True)
    
    os.makedirs(os.path.join(output_folder, 'annotations'), exist_ok=True)
    
    annotation_path = os.path.join(source_folder, 'annotations', 'annotations.json')
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation file not found at {annotation_path}")
    
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    
    all_images = annotations['images']
    
    random.shuffle(all_images)
    
    total_images = len(all_images)
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)
    
    train_images = all_images[:train_size]
    val_images = all_images[train_size:train_size + val_size]
    test_images = all_images[train_size + val_size:]
    
    print(f"Total images: {total_images}")
    print(f"Train: {len(train_images)} images ({train_ratio*100:.1f}%)")
    print(f"Validation: {len(val_images)} images ({val_ratio*100:.1f}%)")
    print(f"Test: {len(test_images)} images ({test_ratio*100:.1f}%)")
    
    split_data = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    for split_name, images in split_data.items():
        print(f"Processing {split_name} split...")
        
        split_annotation = {
            'info': annotations.get('info', {}),
            'categories': annotations.get('categories', []),
            'images': []
        }
        
        for img_data in images:
            img_file = img_data['file_name']
            mask_file = img_file.replace('image_', 'mask_').replace('.jpg', '.png')
            
            src_img = os.path.join(source_folder, 'images', img_file)
            src_mask = os.path.join(source_folder, 'masks', mask_file)
            
            dst_img = os.path.join(output_folder, split_name, 'images', img_file)
            dst_mask = os.path.join(output_folder, split_name, 'masks', mask_file)
            
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            else:
                print(f"Warning: Image file not found: {src_img}")
                
            if os.path.exists(src_mask):
                shutil.copy2(src_mask, dst_mask)
            else:
                print(f"Warning: Mask file not found: {src_mask}")
            
            split_annotation['images'].append(img_data)
        
        with open(os.path.join(output_folder, 'annotations', f'{split_name}.json'), 'w') as f:
            json.dump(split_annotation, f, indent=2)
    
    combined_annotation = {
        'info': annotations.get('info', {}),
        'categories': annotations.get('categories', []),
        'splits': {
            'train': [img['file_name'] for img in train_images],
            'val': [img['file_name'] for img in val_images],
            'test': [img['file_name'] for img in test_images]
        },
        'images': all_images
    }
    
    with open(os.path.join(output_folder, 'annotations', 'combined.json'), 'w') as f:
        json.dump(combined_annotation, f, indent=2)
    
    print(f"\n Dataset successfully split into {output_folder}/")
    print(f"  Training:   {len(train_images)} images")
    print(f"  Validation: {len(val_images)} images")
    print(f"  Testing:    {len(test_images)} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split a dataset into train, validation, and test sets.')
    parser.add_argument('--source', default='dataset', help='Source dataset folder')
    parser.add_argument('--output', default='nn_dataset', help='Output folder for split dataset')
    parser.add_argument('--train', type=float, default=0.7, help='Training set ratio (default: 0.7)')
    parser.add_argument('--val', type=float, default=0.2, help='Validation set ratio (default: 0.2)')
    parser.add_argument('--test', type=float, default=0.1, help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    total_ratio = args.train + args.val + args.test
    if abs(total_ratio - 1.0) > 1e-10:
        print(f"Warning: Split ratios (train:{args.train}, val:{args.val}, test:{args.test}) "
              f"sum to {total_ratio}, not 1.0")
        norm_factor = 1.0 / total_ratio
        args.train *= norm_factor
        args.val *= norm_factor
        args.test *= norm_factor
        print(f"Normalized ratios: train:{args.train:.2f}, val:{args.val:.2f}, test:{args.test:.2f}")
    
    split_dataset(
        source_folder=args.source,
        output_folder=args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        random_seed=args.seed
    )