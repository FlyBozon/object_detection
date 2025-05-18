import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import cv2

try:
    import clearml
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    print("ClearML not available. Install with 'pip install clearml' for additional monitoring.")

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("GradCAM not available. Install with 'pip install grad-cam' for model explainability.")

current_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_path, 'nn_dataset')
output_path = os.path.join(current_path, 'yolo_output')
os.makedirs(output_path, exist_ok=True)

def create_dataset_yaml():
    data = {
        'path': dataset_path,  # Path to dataset root dir
        'train': os.path.join('train', 'images'),  # Relative to 'path'
        'val': os.path.join('val', 'images'),      # Relative to 'path'
        'test': os.path.join('test', 'images'),    # Relative to 'path'
        'names': {}  # Will be filled with class names
    }
    
    categories_path = os.path.join(dataset_path, 'annotations', 'annotations.json')
    if os.path.exists(categories_path):
        import json
        with open(categories_path, 'r') as f:
            annotations = json.load(f)
        
        if 'categories' in annotations:
            for category in annotations['categories']:
                class_id = category['id'] - 1  # YOLO uses 0-indexed classes
                class_name = category['name']
                data['names'][class_id] = class_name
    
    if not data['names']:
        data['names'] = {0: 'object'}
    
    yaml_path = os.path.join(current_path, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Created dataset.yaml with the following content:")
    print(yaml.dump(data, default_flow_style=False))
    
    return yaml_path

def init_clearml(experiment_name="YOLO Detection Training"):
    if CLEARML_AVAILABLE:
        task = clearml.Task.init(
            project_name="Object Detection",
            task_name=experiment_name,
            output_uri=True  
        )
        return task
    return None


def train_yolo_model(yaml_path, model_name="yolov8s.pt", epochs=100, img_size=640, batch_size=16, clearml_task=None):
    print(f"Starting YOLO training with model: {model_name}")
    
    model = YOLO(model_name)
    
    train_args = {
        'data': yaml_path,               # Dataset configuration
        'epochs': epochs,                # Number of epochs
        'imgsz': img_size,               # Image size
        'batch': batch_size,             # Batch size
        'project': output_path,          # Project directory
        'name': 'training_run',          # Run name
        'cache': True,                   # Cache images for faster training
        'device': 'cpu' if not torch.cuda.is_available() else 0,  

        # TensorBoard 
        'plots': True,                   #saving plots for TensorBoard
        
        #regularization parameters
        'weight_decay': 0.0005,          # L2 regularization
        'dropout': 0.1,                  #add dropout for better generalization
        'patience': 20,                  # Early stopping patience
        
        # Data augmentation (helps with generalization)
        'fliplr': 0.5,                   # Horizontal flip 50% of the time
        'scale': 0.5,                    # Random scaling
        'mosaic': 1.0,                   # Mosaic augmentation
        'mixup': 0.1,                    # MixUp augmentation
        
        # Save options
        'save': True,                    # Save models
        'save_period': 10,               # Save checkpoint every 10 epochs
    }
    
    #hyperparameters for ClearML
    if clearml_task:
        clearml_task.connect(train_args)
    
    print("\nTraining configuration:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    
    print("\nStarting training...")
    results = model.train(**train_args)
    
    best_model_path = os.path.join(output_path, 'training_run', 'weights', 'best.pt')
    return model, best_model_path, results

def evaluate_model(model_path, yaml_path, clearml_task=None):
    print(f"\nEvaluating model: {model_path}")
    
    model = YOLO(model_path)
    
    results = model.val(data=yaml_path)
    
    metrics = {
        'mAP50-95': float(results.box.map),   # Mean Average Precision @IoU=0.5:0.95
        'mAP50': float(results.box.map50),    # Mean Average Precision @IoU=0.5
        'Precision': float(results.box.p),    # Precision
        'Recall': float(results.box.r),       # Recall
        'F1-Score': float(2 * (results.box.p * results.box.r) / 
                         (results.box.p + results.box.r + 1e-16))  # F1 Score
    }
    
    print("\nEvaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    #metrics to ClearML
    if clearml_task:
        for metric_name, metric_value in metrics.items():
            clearml_task.get_logger().report_scalar(
                title="Validation Metrics",
                series=metric_name,
                value=metric_value,
                iteration=0
            )
    
    metrics_explanation = """
Metrics Explanation:
-------------------
1. Precision (P): The ratio of correctly predicted positive observations to the total predicted positives.
   - Formula: TP / (TP + FP)
   - Focuses on: "How many of the detected objects are actually correct?"
   - High precision means few false positives.

2. Recall (R): The ratio of correctly predicted positive observations to all actual positives.
   - Formula: TP / (TP + FN)
   - Focuses on: "How many of the actual objects did the model detect?"
   - High recall means few false negatives.

3. F1-Score: The harmonic mean of Precision and Recall, providing a balance between them.
   - Formula: 2 * (P * R) / (P + R)
   - Useful when: You need a single metric that balances false positives and false negatives.

4. mAP (mean Average Precision): The mean of AP calculated across all classes and/or IoU thresholds.
   - mAP50: Using an IoU threshold of 0.5
   - mAP50-95: Average over IoU thresholds from 0.5 to 0.95
   - Provides a comprehensive assessment of detection quality across classes.

5. Accuracy: The ratio of correct predictions to total predictions.
   - Formula: (TP + TN) / (TP + TN + FP + FN)
   - Less useful for object detection because:
     a) Background pixels would dominate (many true negatives)
     b) Doesn't account for localization quality
     c) Less informative with multiple objects per image
    """
    
    print(metrics_explanation)
    
    metrics_file = os.path.join(output_path, 'metrics_explanation.txt')
    with open(metrics_file, 'w') as f:
        f.write(metrics_explanation)
    
    if clearml_task:
        clearml_task.upload_artifact("Metrics Explanation", metrics_file)
    
    return metrics

# ----- exporting model for Netron visualization -----
def export_model_for_visualization(model_path, clearml_task=None):
    print(f"\nExporting model for visualization: {model_path}")
    
    model = YOLO(model_path)
    
    #to ONNX format for Netron
    onnx_path = model.export(format='onnx')
    
    print(f"Model exported to: {onnx_path}")
    print("You can now visualize this model using Netron (https://netron.app/)")
    
    #upload to ClearML
    if clearml_task:
        clearml_task.upload_artifact("ONNX Model", onnx_path)
    
    return onnx_path

# ----- generation of CAM visualizations -----
def generate_cam_visualizations(model_path, test_images_dir, clearml_task=None):
    if not GRADCAM_AVAILABLE:
        print("Skipping CAM visualization - grad-cam package not installed")
        return None
    
    print(f"\nGenerating CAM visualizations using test images from: {test_images_dir}")
    
    cam_output_dir = os.path.join(output_path, 'cam_visualizations')
    os.makedirs(cam_output_dir, exist_ok=True)
    
    model = YOLO(model_path)
    
    target_layers = [model.model.model[-2]]
    
    #initialization of gradcam
    cam = GradCAM(model=model.model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    
    image_files = list(Path(test_images_dir).glob('*.jpg')) + list(Path(test_images_dir).glob('*.png'))
    output_paths = []
    
    for i, image_path in enumerate(image_files[:5]):  
        print(f"Processing image: {image_path}")
        
        img = cv2.imread(str(image_path))
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float().div(255).unsqueeze(0)
        
        grayscale_cam = cam(input_tensor=input_tensor)
        grayscale_cam = grayscale_cam[0, :]
        
        cam_image = show_cam_on_image(rgb_img / 255.0, grayscale_cam, use_rgb=True)
        
        results = model.predict(str(image_path))
        
        for det in results:
            boxes = det.boxes.xyxy.cpu().numpy()
            cls = det.boxes.cls.cpu().numpy()
            conf = det.boxes.conf.cpu().numpy()
            
            for box, cl, cf in zip(boxes, cls, conf):
                x1, y1, x2, y2 = box.astype(int)
                cls_name = results[0].names[int(cl)]
                cv2.rectangle(cam_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cam_image, f"{cls_name} {cf:.2f}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        output_path = os.path.join(cam_output_dir, f"cam_{i}_{image_path.name}")
        plt.figure(figsize=(10, 10))
        plt.imshow(cam_image)
        plt.title("Class Activation Map Visualization")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        output_paths.append(output_path)
        
        if clearml_task:
            clearml_task.get_logger().report_image(
                title="CAM Visualizations",
                series=f"Image {i}",
                image=output_path,
                iteration=0
            )
    
    cam_explanation = """
Class Activation Mapping (CAM) Explanation:
------------------------------------------
The Class Activation Map visualizations show which parts of the image the model is focusing on 
when making detection decisions. Warmer colors (red/yellow) indicate areas of high importance 
to the model's decision, while cooler colors (blue/green) indicate less important areas.

This visualization helps to:
1. Verify that the model is looking at the actual objects and not background elements
2. Understand potential biases in the model's attention
3. Debug cases where the model makes incorrect predictions
4. Ensure the model is using appropriate features for classification

These CAM visualizations help make the deep learning model more transparent and explainable.
    """
    
    cam_explanation_path = os.path.join(cam_output_dir, "cam_explanation.txt")
    with open(cam_explanation_path, 'w') as f:
        f.write(cam_explanation)
    
    if clearml_task:
        clearml_task.upload_artifact("CAM Explanation", cam_explanation_path)
    
    print(f"Generated CAM visualizations: {len(output_paths)} images saved to {cam_output_dir}")
    return output_paths

def main():
    print("===============================================")
    print("     YOLO Object Detection Training Pipeline")
    print("===============================================")
    
    yaml_path = create_dataset_yaml()
    
    train_images_path = os.path.join(dataset_path, 'train', 'images')
    val_images_path = os.path.join(dataset_path, 'val', 'images')
    
    if not os.path.exists(train_images_path):
        sys.exit(f"Error: Training images not found at {train_images_path}")
    
    if not os.path.exists(val_images_path):
        print(f"Warning: Validation images not found at {val_images_path}")
        print("Creating a validation set from training set...")
        
        os.makedirs(val_images_path, exist_ok=True)
        os.makedirs(os.path.join(dataset_path, 'val', 'labels'), exist_ok=True)
        
        train_files = list(Path(train_images_path).glob('*.jpg'))[:5] 
        for file in train_files:
            img = Image.open(file)
            val_file = os.path.join(val_images_path, file.name)
            img.save(val_file)
            
            train_label = os.path.join(dataset_path, 'train', 'labels', 
                                      file.stem + '.txt')
            if os.path.exists(train_label):
                val_label = os.path.join(dataset_path, 'val', 'labels', 
                                        file.stem + '.txt')
                with open(train_label, 'r') as src, open(val_label, 'w') as dst:
                    dst.write(src.read())
    
    #initialize ClearML
    clearml_task = init_clearml("YOLO Detection with Additional Features")
    
    #training
    model, best_model_path, results = train_yolo_model(
        yaml_path=yaml_path,
        model_name="yolov8s.pt",
        epochs=50, 
        img_size=640,
        batch_size=16,
        clearml_task=clearml_task
    )
    
    metrics = evaluate_model(best_model_path, yaml_path, clearml_task)
    
    onnx_path = export_model_for_visualization(best_model_path, clearml_task)
    
    test_images_dir = os.path.join(dataset_path, 'test', 'images')
    if not os.path.exists(test_images_dir):
        test_images_dir = os.path.join(dataset_path, 'val', 'images')  # Fallback to val if test doesn't exist
    
    cam_paths = generate_cam_visualizations(best_model_path, test_images_dir, clearml_task)
    
    print("\n===============================================")
    print("            Training Complete!")
    print("===============================================")
    print(f"Model saved to: {best_model_path}")
    print(f"ONNX model for Netron: {onnx_path}")
    print(f"TensorBoard logs: {os.path.join(output_path, 'training_run')}")
    print(f"Use 'tensorboard --logdir {os.path.join(output_path, 'training_run')}' to view training curves")
    print("\nAdditional metrics and explanations saved to output directory")
    
    if CLEARML_AVAILABLE:
        print("\nClearML dashboard contains all logs, metrics, and artifacts")
    
    return best_model_path

if __name__ == "__main__":
    main()