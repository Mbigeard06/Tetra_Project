from sahi.predict import get_sliced_prediction
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
import json
from tetra.utils import file_io 
from sahi.predict import predict
from pathlib import Path

def solo_inference(detection_model, image_path):
    
    return get_sliced_prediction(
    image_path,
    detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    perform_standard_pred = False,
    postprocess_class_agnostic= True,
    verbose=False
    )

def batch_inference(model, inference_dataset, file_to_id, exts, out_dir=None):
    i = 0
    predictions = []
    for f in Path(inference_dataset).iterdir():
            if f.suffix.lower() in exts:
                prediction = solo_inference(model, str(f))
                prediction = prediction.to_coco_annotations()
                if out_dir:
                     # Save YOLO predictions
                    yolo_pred = file_io.coco_to_yolo(prediction, f, True)
                    file_io.save_yolo_labels(yolo_pred, out_dir, f.stem + ".txt")
                for ann in prediction:
                    #Update id
                    ann["image_id"] = file_to_id[f.name]
                    #Update category
                    ann["category_id"] = 1
                    #Change category name
                    ann["category_name"] = "Lyrurus_tetrix"
                predictions.extend(prediction)
                i += 1
                if i % 100 == 0 :
                    print(f"Processing image {i}...")
    return predictions



