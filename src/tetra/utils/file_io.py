from PIL import Image
import os

def coco_to_yolo(annotations, image_path):
    image = Image.open(image_path)
    img_width, img_height = image.size
    yolo_labels = []

    for ann in annotations:
        x, y, w, h = ann["bbox"]
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        class_id = ann["category_id"]
        score = ann["score"]
        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f} {score:.6f}")

    return yolo_labels

def save_yolo_labels(yolo_labels, label_dir, file_name):
    os.makedirs(label_dir, exist_ok=True)
    path = os.path.join(label_dir, file_name)
    with open(path, "w") as f:
        for line in yolo_labels:
            f.write(line + "\n")

