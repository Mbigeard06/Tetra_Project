from sahi.predict import get_sliced_prediction
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from tetra.utils import file_io as fi
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





model_path = "../../../../runs/train/lyrurus_yolov11m/weights/best.pt"
image_path = "../../../../dataset_og/04160110_JPG.rf.54148cf7aa340b30973069c8001b8254.jpg"  
save_path = "../../../../outputs/labels"

res = inference(model_path=model_path, image_path=image_path)
name = os.path.basename(image_path)
print(fi.coco_to_yolo(res.to_coco_annotations(), image_path=image_path))

