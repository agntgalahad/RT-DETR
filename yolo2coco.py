import os
import json
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Paths to your dataset
yolo_images_path = "/content/dataset/YOLOv8/groundOnly_pothole/images"
yolo_labels_path = "/content/dataset/YOLOv8/groundOnly_pothole/labels"
output_train_json_path = "coco_train_annotations.json"
output_val_json_path = "coco_val_annotations.json"

def initialize_coco_structure():
    return {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "pothole"}]
    }

def convert_yolo_to_coco(yolo_annotations_path, yolo_images_path, coco_structure):
    annotation_id = 1
    for dirpath, _, filenames in os.walk(yolo_annotations_path):
        for filename in filenames:
            if filename.endswith(".txt"):
                image_filename = filename.replace(".txt", ".jpg")
                image_path = os.path.join(yolo_images_path, image_filename)
                if not os.path.exists(image_path):
                    continue
                # Get image size
                with Image.open(image_path) as img:
                    width, height = img.size

                # Add image info to COCO structure
                image_id = len(coco_structure["images"]) + 1
                coco_structure["images"].append({
                    "id": image_id,
                    "file_name": os.path.relpath(image_path, yolo_images_path),
                    "width": width,
                    "height": height
                })

                # Read YOLO annotations
                with open(os.path.join(dirpath, filename), 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        category_id = int(parts[0]) + 1  # YOLO categories are 0-based, COCO are 1-based
                        x_center = float(parts[1]) * width
                        y_center = float(parts[2]) * height
                        bbox_width = float(parts[3]) * width
                        bbox_height = float(parts[4]) * height
                        x_min = x_center - bbox_width / 2
                        y_min = y_center - bbox_height / 2

                        # Add annotation info to COCO structure
                        coco_structure["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": [x_min, y_min, bbox_width, bbox_height],
                            "area": bbox_width * bbox_height,
                            "iscrowd": 0
                        })
                        annotation_id += 1

# Initialize COCO structures
coco_train = initialize_coco_structure()
coco_val = initialize_coco_structure()

# Convert train and val datasets
convert_yolo_to_coco(os.path.join(yolo_labels_path, "train"), os.path.join(yolo_images_path, "train"), coco_train)
convert_yolo_to_coco(os.path.join(yolo_labels_path, "val"), os.path.join(yolo_images_path, "val"), coco_val)

# Save to COCO JSON files
with open(output_train_json_path, 'w') as json_file:
    json.dump(coco_train, json_file, indent=4)

with open(output_val_json_path, 'w') as json_file:
    json.dump(coco_val, json_file, indent=4)

print(f"Conversion completed. COCO annotations saved to {output_train_json_path} and {output_val_json_path}")
