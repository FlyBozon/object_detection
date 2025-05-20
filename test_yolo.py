from ultralytics import YOLO
import os

model = YOLO('best_last_one.pt')  
image_folder = 'nn_dataset/test/images'

output_folder = 'predictions'
subfolder_name = 'nn_dataset'  

os.makedirs(os.path.join(output_folder, subfolder_name), exist_ok=True)

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        image_path = os.path.join(image_folder, filename)
        results = model(
            image_path,
            save=True,
            save_txt=True,
            project=output_folder,
            name=subfolder_name,
            exist_ok=True
        )

print(f"Inference completed. Check the '{output_folder}/{subfolder_name}' folder.")
