import os
import random
import json
from PIL import Image
from collections import defaultdict
from datetime import datetime

try:
    resample = Image.Resampling.LANCZOS
except AttributeError:
    resample = Image.ANTIALIAS

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
                if file.lower().endswith('.png'):
                    class_to_images[cls_name].append(os.path.join(cls_path, file))
    return class_to_images

def paste_object(background, object_img, position):
    background.paste(object_img, position, object_img)
    return background

def create_class_mask(mask, object_img, position, class_id):
    alpha = object_img.getchannel('A')
    binary = alpha.point(lambda p: 255 if p > 0 else 0)
    
    color_mask = Image.new('L', object_img.size, class_id)
    
    mask.paste(color_mask, position, binary)
    return mask

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

def generate_dataset(
    background_folder='random_images',
    object_folder='for_dataset',
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

        #scalling 30-80% depending on the background
        scale = random.uniform(0.3, 0.8)
        obj_w = int(width * scale)
        obj_h = int(object_img.height * (obj_w / object_img.width))
        object_img = object_img.resize((obj_w, obj_h), resample)

        max_x = max(1, width - obj_w)
        max_y = max(1, height - obj_h)
        position = (random.randint(0, max_x), random.randint(0, max_y))

        background = paste_object(background, object_img, position)
        mask = create_binary_mask(mask, object_img, position)#, class_name_to_id[cls])

        annotations.append({
            "class_id": class_name_to_id[cls],
            "class_name": cls,
            "bbox": [position[0], position[1], obj_w, obj_h]
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

        print(f"[{i+1}/{image_count}] Image saved with object: {cls}")

    #COCO
    with open(os.path.join(output_folder, 'annotations', 'annotations.json'), 'w') as f:
        json.dump({
            "info": {
                "description": "Synthetic Dataset",
                "version": "1.0",
                "date_created": datetime.now().isoformat()
            },
            "categories": [
                {"id": class_id, "name": class_name}
                for class_name, class_id in class_name_to_id.items()
            ],
            "images": all_annotations
        }, f, indent=2)

    print("✅ Dataset generation complete.")

#run
generate_dataset(image_count=100)
