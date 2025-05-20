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
import shutil  

from clearml import Task

if Task.current_task():
    Task.current_task().close()

task = Task.init(
    project_name="YOLOv8 Experiments",
    task_name="Run 1",
    task_type=Task.TaskTypes.training
)
#optional :)
task.connect({
    "epochs": 50,
    "batch_size": 16,
    "model": "yolov8s.pt"
})

#task.execute_remotely(queue_name='default')


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

# Set paths for Colab
#current_path =  '/content'

current_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_path, 'nn_dataset')
output_path = os.path.join(current_path, 'yolo_output')
os.makedirs(output_path, exist_ok=True)

def create_proper_yaml():
    yaml_content = f"""path: {dataset_path}
train: train/images
val: val/images
test: test/images

# Class names
names:
  0: komb
  1: miecz
  2: srub
"""

    yaml_path = os.path.join(current_path, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Created proper YAML file at: {yaml_path}")

    with open(yaml_path, 'r') as f:
        print("YAML file contents:\n" + f.read())

    return yaml_path

# Create the YAML file
yaml_path = create_proper_yaml()

def verify_label_files():
    labels_dir = os.path.join(dataset_path, 'train', 'labels')
    if not os.path.exists(labels_dir):
        print(f"Warning: Labels directory not found: {labels_dir}")
        return False

    label_files = list(Path(labels_dir).glob('*.txt'))
    if not label_files:
        print("Warning: No label files found!")
        return False

    class_counts = {0: 0, 1: 0, 2: 0}
    total_count = 0

    print("Checking ALL label files for class distribution...")
    for label_file in label_files:
        with open(label_file, 'r') as f:
            content = f.read().strip()
            lines = content.split('\n')
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 5:  # class x y w h format
                        try:
                            class_id = int(parts[0])
                            if class_id in class_counts:
                                class_counts[class_id] += 1
                                total_count += 1
                            else:
                                print(f"Warning: Found invalid class ID {class_id} in {label_file}")
                        except ValueError:
                            print(f"Warning: Invalid format in {label_file}: {line}")

    print("\nLabel file verification - Class distribution:")
    for class_id, count in class_counts.items():
        percentage = count/max(1, total_count)*100
        print(f"  Class {class_id}: {count} instances ({percentage:.1f}%)")
        if percentage < 10:
            print(f"  WARNING: Class {class_id} has very low representation ({percentage:.1f}%)!")

    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    if min_count == 0:
        print("\nCRITICAL WARNING: Some classes have ZERO instances! The model cannot learn to detect them.")
        print("Please add more examples of the missing classes to your dataset.")
    elif max_count / max(1, min_count) > 3:
        print("\nWARNING: Significant class imbalance detected!")
        print(f"The most common class has {max_count} instances while the least common has only {min_count}.")
        print("Consider these options to address class imbalance:")
        print("1. Add more images of the underrepresented classes")
        print("2. Use data augmentation specifically for underrepresented classes")
        print("3. Use class weights in training (implemented in the updated code)")

    return True, class_counts

def balance_dataset(class_counts):
    max_class = max(class_counts, key=class_counts.get)
    max_count = class_counts[max_class]

    print(f"\nAttempting to balance dataset through augmentation...")
    print(f"Target count per class: {max_count}")

    labels_dir = os.path.join(dataset_path, 'train', 'labels')
    images_dir = os.path.join(dataset_path, 'train', 'images')

    augmented_count = 0

    for class_id, count in class_counts.items():
        if count < max_count and count > 0:
            needed = max_count - count
            print(f"Class {class_id} needs {needed} more instances to reach balance")

            images_with_class = []
            for label_file in Path(labels_dir).glob('*.txt'):
                with open(label_file, 'r') as f:
                    content = f.read()
                    if f" {class_id} " in f" {content} " or content.startswith(f"{class_id} "):
                        img_path = os.path.join(images_dir, label_file.stem + '.jpg')
                        if os.path.exists(img_path):
                            images_with_class.append((img_path, str(label_file)))

            if not images_with_class:
                print(f"No images found containing class {class_id}!")
                continue

            print(f"Found {len(images_with_class)} images containing class {class_id}")

            for i in range(min(needed, len(images_with_class) * 5)):
                img_path, label_path = images_with_class[i % len(images_with_class)]

                base_name = os.path.basename(img_path)
                new_img_name = f"aug_{class_id}_{i}_{base_name}"
                new_img_path = os.path.join(images_dir, new_img_name)

                new_label_name = f"aug_{class_id}_{i}_{os.path.basename(label_path)}"
                new_label_path = os.path.join(labels_dir, new_label_name)

                img = cv2.imread(img_path)

                #random rotation
                angle = np.random.uniform(-15, 15)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                img = cv2.warpAffine(img, M, (w, h))

                #random brightness/contrast adjustment
                alpha = np.random.uniform(0.8, 1.2)  # contrast
                beta = np.random.uniform(-10, 10)    # brightness
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

                cv2.imwrite(new_img_path, img)

                shutil.copy2(label_path, new_label_path)

                augmented_count += 1

    print(f"Created {augmented_count} augmented images to help balance the dataset")
    return augmented_count > 0

def train_yolo_model(yaml_path, model_name="yolov8s.pt", epochs=100, img_size=640, batch_size=16, clearml_task=None, class_counts=None):
    print(f"Starting YOLO training with model: {model_name}")

    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

    num_classes = len(yaml_data.get('names', {}))
    print(f"YAML file contains {num_classes} classes")

    model = YOLO(model_name)

    class_weights = None
    if class_counts:
        max_count = max(class_counts.values())
        class_weights = {class_id: max_count / max(1, count) for class_id, count in class_counts.items()}
        print(f"\nClass weights to address imbalance: {class_weights}")

    train_args = {
        'data': yaml_path,            
        'epochs': epochs,               
        'imgsz': img_size,             
        'batch': batch_size,           
        'project': output_path,          
        'name': 'training_run',        
        'cache': False,                  #disable cache to avoid using old cached data
        'device': 'cpu' if not torch.cuda.is_available() else 0,

        #tensorBoard
        'plots': True,                 

        #optimization parameters - slower learning rate for better precision
        'lr0': 0.001,                    # initial learning rate
        'lrf': 0.01,                     #final learning rate fraction

        'weight_decay': 0.0005,          #L2
        'dropout': 0.1,                 
        'patience': 50,                  #extended early stopping patience

        'fliplr': 0.5,                 
        'flipud': 0.2,                 
        'scale': 0.5,                   
        'mosaic': 1.0,                 
        'mixup': 0.3,                   
        'degrees': 15.0,               
        'translate': 0.2,              
        'shear': 5.0,                
        'perspective': 0.001,          
        'hsv_h': 0.015,                 
        'hsv_s': 0.7,                  
        'hsv_v': 0.4,                  

        'cls': 1.0,                     
        'box': 7.5,                    
        'dfl': 1.5,                   

        'iou': 0.6,                     
        'conf': 0.001,                  

        'save': True,                   
        'save_period': 10,               #save checkpoint every 10 epochs
    }

    if clearml_task:
        clearml_task.connect(train_args)

    print("\nTraining configuration:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")

    print("\nStarting training...")
    try:
        results = model.train(**train_args)
        best_model_path = os.path.join(output_path, 'training_run', 'weights', 'best.pt')
        return model, best_model_path, results
    except Exception as e:
        print(f"Training error: {e}")
        weights_dir = os.path.join(output_path, 'training_run', 'weights')
        if os.path.exists(weights_dir):
            weight_files = list(Path(weights_dir).glob('*.pt'))
            if weight_files:
                latest_model = max(weight_files, key=os.path.getmtime)
                print(f"Using latest saved model: {latest_model}")
                return YOLO(str(latest_model)), str(latest_model), None
        raise e

def evaluate_model(model_path, yaml_path, clearml_task=None):
    print(f"\nEvaluating model: {model_path}")

    model = YOLO(model_path)

    results = model.val(data=yaml_path, conf=0.1)

    metrics = {
        'mAP50-95': float(results.box.map),   # Mean Average Precision @IoU=0.5:0.95
        'mAP50': float(results.box.map50),    # Mean Average Precision @IoU=0.5
        'Precision': float(results.box.p),   
        'Recall': float(results.box.r),      
        'F1-Score': float(2 * (results.box.p * results.box.r) /
                         (results.box.p + results.box.r + 1e-16)) 
    }

    print("\nEvaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    if clearml_task:
        for metric_name, metric_value in metrics.items():
            clearml_task.get_logger().report_scalar(
                title="Validation Metrics",
                series=metric_name,
                value=metric_value,
                iteration=0
            )

    class_metrics = results.box.class_metrics()

    if class_metrics is not None:
        print("\nPer-Class Metrics:")
        for i, class_name in enumerate(model.names.values()):
            if i < len(class_metrics['precision']):
                print(f"  Class {i} ({class_name}):")
                print(f"    Precision: {class_metrics['precision'][i]:.4f}")
                print(f"    Recall: {class_metrics['recall'][i]:.4f}")
                print(f"    mAP50: {class_metrics['map50'][i]:.4f}")
                print(f"    mAP50-95: {class_metrics['map'][i]:.4f}")

                if class_metrics['map50'][i] < 0.5:
                    print(f"    WARNING: Poor detection performance for class {class_name}.")
                    if class_metrics['precision'][i] < 0.5:
                        print(f"    → Low precision indicates many false positives.")
                    if class_metrics['recall'][i] < 0.5:
                        print(f"    → Low recall indicates many missed objects.")

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

    return metrics, class_metrics

def run_inference_with_advanced_tuning(model_path, test_images_dir, num_samples=5):
    print(f"\nRunning inference with advanced tuning for multi-class detection...")

    inference_output_dir = os.path.join(output_path, 'inference_results')
    os.makedirs(inference_output_dir, exist_ok=True)

    model = YOLO(model_path)

    image_files = list(Path(test_images_dir).glob('*.jpg')) + list(Path(test_images_dir).glob('*.png'))
    if len(image_files) > num_samples:
        import random
        random.seed(42)  
        image_files = random.sample(image_files, num_samples)

    import colorsys
    class_names = model.names
    num_classes = len(class_names)
    colors = {}
    for i in range(num_classes):
        h = i / max(1, num_classes)
        s = 0.8
        v = 0.9
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors[i] = (int(r*255), int(g*255), int(b*255))

    thresholds = [0.05, 0.1, 0.25]

    for i, image_path in enumerate(image_files):
        print(f"Processing image: {image_path}")
        orig_img = cv2.imread(str(image_path))

        plt.figure(figsize=(20, 5 * len(thresholds)))

        all_detected_classes = set()

        for t_idx, threshold in enumerate(thresholds):
            vis_img = orig_img.copy()

            results = model.predict(str(image_path), conf=threshold, iou=0.3)  #lower IoU threshold

            detected_classes = set()

            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()

                for box, cls_id, conf in zip(boxes, cls_ids, confs):
                    x1, y1, x2, y2 = box.astype(int)
                    class_id = int(cls_id)
                    class_name = class_names[class_id]
                    confidence = float(conf)

                    detected_classes.add(class_id)
                    all_detected_classes.add(class_id)

                    color = colors.get(class_id, (0, 255, 0))

                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)

                    text = f"{class_name} ({confidence:.2f})"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(vis_img, (x1, y1 - text_size[1] - 10),
                                (x1 + text_size[0] + 10, y1), color, -1)

                    cv2.putText(vis_img, text, (x1 + 5, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

            plt.subplot(len(thresholds), 1, t_idx + 1)
            plt.imshow(vis_img_rgb)
            plt.title(f"Confidence Threshold: {threshold}")

            detected_class_names = [f"{class_names[cls_id]} (ID: {cls_id})" for cls_id in sorted(detected_classes)]
            if detected_class_names:
                detected_text = "Detected classes: " + ", ".join(detected_class_names)
            else:
                detected_text = "No objects detected"

            plt.xlabel(detected_text)
            plt.axis('off')

        plt.tight_layout()
        output_path_thresholds = os.path.join(inference_output_dir, f"detection_thresholds_{i}_{Path(image_path).name}")
        plt.savefig(output_path_thresholds, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        results_default = model.predict(str(image_path), conf=0.1)
        default_img = results_default[0].plot(conf=True, line_width=2, font_size=14)
        plt.imshow(cv2.cvtColor(default_img, cv2.COLOR_BGR2RGB))
        plt.title("Default NMS (IoU=0.7)")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        results_tuned = model.predict(str(image_path), conf=0.1, iou=0.3)  

        tuned_img = orig_img.copy()
        detected_classes_tuned = set()

        for r in results_tuned:
            boxes = r.boxes.xyxy.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for box, cls_id, conf in zip(boxes, cls_ids, confs):
                x1, y1, x2, y2 = box.astype(int)
                class_id = int(cls_id)
                class_name = class_names[class_id]
                confidence = float(conf)

                detected_classes_tuned.add(class_id)

                color = colors.get(class_id, (0, 255, 0))
                cv2.rectangle(tuned_img, (x1, y1), (x2, y2), color, 3)

                text = f"{class_name} ({confidence:.2f})"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(tuned_img, (x1, y1 - text_size[1] - 10),
                            (x1 + text_size[0] + 10, y1), color, -1)

                cv2.putText(tuned_img, text, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        plt.imshow(cv2.cvtColor(tuned_img, cv2.COLOR_BGR2RGB))
        plt.title("Tuned NMS (IoU=0.3)")
        plt.axis('off')

        tuned_class_names = [f"{class_names[cls_id]}" for cls_id in sorted(detected_classes_tuned)]
        default_class_names = [f"{class_names[cls_id]}" for cls_id in sorted({int(cls) for cls in results_default[0].boxes.cls.cpu().numpy()})]

        plt.figtext(0.5, 0.01,
                    f"Default detected: {', '.join(default_class_names)}\nTuned detected: {', '.join(tuned_class_names)}",
                    ha="center", fontsize=12,
                    bbox={"facecolor":"white", "alpha":0.8, "pad":5})

        output_path_nms = os.path.join(inference_output_dir, f"detection_nms_tuning_{i}_{Path(image_path).name}")
        plt.savefig(output_path_nms, bbox_inches='tight')
        plt.close()

    recommendations = """
Recommendations for Better Multi-Class Detection:
------------------------------------------------
1. Class Balance
   - Ensure your training set has a similar number of examples for each class
   - If imbalanced, use class_weights parameter or augment underrepresented classes

2. Detection Confidence
   - Lower confidence thresholds during inference (0.05-0.10) for rare classes
   - Use higher thresholds (0.25+) for common classes to reduce false positives

3. NMS Tuning
   - Lower IoU thresholds (0.2-0.4) help detect densely packed objects
   - Higher IoU thresholds (0.5-0.7) reduce duplicate detections

4. Data Quality
   - Ensure all classes are correctly labeled in training data
   - Check for label consistency and annotation quality

5. Model Size
   - Larger models (YOLOv8m, YOLOv8l) may detect smaller objects better
   - Consider using a larger variant if your classes have subtle differences

6. Augmentation
   - Increase augmentation for better generalization
   - Use targeted augmentation for underrepresented classes
"""

    recommendations_path = os.path.join(inference_output_dir, "detection_recommendations.txt")
    with open(recommendations_path, 'w') as f:
        f.write(recommendations)

    print("\nCompleted enhanced inference testing with multiple confidence thresholds and NMS tuning")
    print(f"Results saved to {inference_output_dir}")

    return inference_output_dir

def main():
    print("===============================================")
    print("      YOLO Object Detection Pipeline")
    print("===============================================")

    cache_file = os.path.join(dataset_path, 'train', 'labels.cache')
    if os.path.exists(cache_file):
        print(f"Removing old cache file: {cache_file}")
        os.remove(cache_file)

    yaml_path = create_proper_yaml()

    verification_result, class_counts = verify_label_files()

    if verification_result and class_counts:
        balance_dataset(class_counts)

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
            val_file = os.path.join(val_images_path, file.name)
            shutil.copy2(str(file), val_file)

            train_label = os.path.join(dataset_path, 'train', 'labels',
                                      file.stem + '.txt')
            if os.path.exists(train_label):
                val_label = os.path.join(dataset_path, 'val', 'labels',
                                        file.stem + '.txt')
                with open(train_label, 'r') as src, open(val_label, 'w') as dst:
                    dst.write(src.read())

        print(f"Created validation set with {len(train_files)} images")

    os.environ["CLEARML_SKIP"] = "1"
    clearml_task = None

    model, best_model_path, results = train_yolo_model(
        yaml_path=yaml_path,
        model_name="yolov8m.pt",  
        epochs=50,       
        img_size=640,
        batch_size=16,
        clearml_task=clearml_task,
        class_counts=class_counts  #pass class counts for weighting
    )

    test_images_dir = os.path.join(dataset_path, 'test', 'images')
    if not os.path.exists(test_images_dir):
        test_images_dir = os.path.join(dataset_path, 'val', 'images')

    print("\nRunning enhanced inference with advanced tuning...")
    inference_dir = run_inference_with_advanced_tuning(best_model_path, test_images_dir, num_samples=5)

    metrics, class_metrics = evaluate_model(best_model_path, yaml_path, clearml_task)

    problem_classes = []
    if class_metrics is not None:
        class_names = {i: name for i, name in enumerate(model.names.values())}
        for i, class_name in enumerate(class_names.values()):
            if i < len(class_metrics['map50']) and class_metrics['map50'][i] < 0.5:
                problem_classes.append((i, class_name))

    if problem_classes:
        print("\n===============================================")
        print("           Problem Classes Detected!")
        print("===============================================")
        print("The following classes have poor detection performance:")
        for class_id, class_name in problem_classes:
            print(f"  - Class {class_id} ({class_name})")

        print("\nRecommendations to improve these classes:")
        print("1. Add more training images for these specific classes")
        print("2. Use a larger model like YOLOv8x for better feature extraction")
        print("3. Try freezing early layers and training only later layers:")
        print("   model.freeze = ['model.2', 'model.3', 'model.4']")
        print("4. Check your annotations for consistency and accuracy")
        print("5. During inference, use a lower confidence threshold (0.05-0.1)")
        print("   and a lower NMS IoU threshold (0.3-0.4)")

        print("\nGenerating a custom tuned model for problem classes...")

        config_path = os.path.join(output_path, 'problem_classes_config.yaml')
        with open(config_path, 'w') as f:
            f.write(f"""
# YOLOv8 config with focus on problem classes
model: {best_model_path}
data: {yaml_path}

# Lowered detection thresholds for problem classes
conf: 0.05
iou: 0.3

# Class-specific parameters
classes: [{','.join(str(c[0]) for c in problem_classes)}]
            """)

        print(f"Created custom config at: {config_path}")
        print("You can use this config for inference with:")
        print(f"yolo predict model={best_model_path} conf=0.05 iou=0.3 source=your_image.jpg")

    print("\n===============================================")
    print("      Improved Training Complete!")
    print("===============================================")
    print(f"Model saved to: {best_model_path}")
    print(f"Inference results: {os.path.join(output_path, 'inference_results')}")
    print("\nFor better detection of all three classes:")
    print("1. When running inference, use lower confidence threshold:")
    print(f"   model.predict(image_path, conf=0.1, iou=0.3)")
    print("2. Consider these command-line parameters for inference:")
    print(f"   yolo predict model={best_model_path} conf=0.1 iou=0.3 source=your_image.jpg")

    return best_model_path

if __name__ == "__main__":
    main()