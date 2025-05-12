from sahi.predict import get_sliced_prediction
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
import json

def inference(model_path, image_path):
    #Load the model
    detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.357,
    device="cpu", 
    )
    #Get the result
    result = get_sliced_prediction(
    image_path,
    detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    )

    return result

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
        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    return yolo_labels




model_path = "../../../../runs/train/lyrurus_yolov11m/weights/best.pt"
image_path = "../../../../dataset_og/04160110_JPG.rf.54148cf7aa340b30973069c8001b8254.jpg"  
save_path = "../../../../outputs/labels"

res = inference(model_path=model_path, image_path=image_path)
name = os.path.basename(image_path)
print(coco_to_yolo(res.to_coco_annotations(), image_path=image_path))

