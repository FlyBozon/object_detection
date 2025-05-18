import os
import cv2
import albumentations as A
import numpy as np

def augment_images(input_dir, output_dir, augmentations_per_image=5):
    os.makedirs(output_dir, exist_ok=True)

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Rotate(limit=20, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.GaussianBlur(p=0.2),
        A.RandomCrop(width=224, height=224, p=1.0),
    ])

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            for i in range(augmentations_per_image):
                augmented = transform(image=image)
                augmented_image = augmented['image']
                output_filename = f"{os.path.splitext(filename)[0]}_aug_{i+1}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, augmented_image)
                print(f"Saved: {output_path}")

augment_images(input_dir='dataset/images', output_dir='dataset/augmented_images', augmentations_per_image=3)
