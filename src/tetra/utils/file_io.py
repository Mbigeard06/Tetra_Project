from PIL import Image
import os
from pathlib import Path
import shutil
import json


def coco_to_yolo(annotations, image_path, pred):
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
        if pred:
            score = ann["score"]
            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f} {score:.6f}")
        else:
            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    return yolo_labels

def save_yolo_labels(yolo_labels, dir, file_name):
    os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, file_name)
    with open(path, "w") as f:
        for line in yolo_labels:
            f.write(line + "\n")

def save_coco_labels(coco_label: dict, dir: str, file_name: str = "annotations.coco.json"):
    os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, file_name)
    with open(path, "w") as f:
        json.dump(coco_label, f, indent=4)


def order_dataset(dataset_path):
    path = Path(dataset_path)
    images_dir = path / "images"
    labels_dir = path / "labels"

    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    for f in path.iterdir():
        if f.is_file():
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                shutil.move(str(f), images_dir / f.name)
            elif f.suffix.lower() == ".txt":
                shutil.move(str(f), labels_dir / f.name)

def coco_to_yolo_ds(annotations_path, img_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(annotations_path) as f:
        coco = json.load(f)
    
    images = {img["id"] : img for img in coco["images"]}

    #save annotations for each images
    image_to_anns = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in image_to_anns:
            #Create the key
            image_to_anns[img_id] = []
        image_to_anns[img_id].append(ann)
    
    #Transform to yolo
    for img_id, img in images.items():
        anns = image_to_anns.get(img_id, [])
        image_path = Path(img_dir) / img["file_name"]
        yolo_ann = coco_to_yolo(anns, image_path)
        label_name = Path(img["file_name"]).stem + ".txt"
        save_yolo_labels(yolo_labels=yolo_ann, label_dir=output_dir, file_name=label_name)

def flatten_pred(nested_dict):
    flat = []
    for ann_list in nested_dict:
        for ann in ann_list:
            flat.append({
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "score": ann["score"]
            })
    return flat

def json_to_coco(input, output):
    with open(input, "r") as fin, open(output, "w") as fout:
        predictions = []
        for line in fin:
            ann = json.loads(line)
            # Keep valid field
            coco_ann = {
                "image_id": ann["image_id"],
                "bbox": ann["bbox"],
                "score": ann["score"],
                "category_id": ann["category_id"]
            }
            predictions.append(coco_ann)
        json.dump(predictions, fout, indent=4)