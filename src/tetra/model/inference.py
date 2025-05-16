from sahi.predict import get_sliced_prediction
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
import json
from tetra.utils import file_io 
from sahi.predict import predict
from pathlib import Path

def solo_inference(model_path, image_path):
    #Load the model
    detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    device="mps", 
    confidence_threshold=0.386
    )
    #Get the result
    result = get_sliced_prediction(
    image_path,
    detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    perform_standard_pred = False
    )

    return result

def batch_inference(inference_dataset, file_to_Id):
    """
    Return coco format annotations
    Args:
        inference_dataset: path to the dataset
        file_to_id : Dictionary file_name => ground truth id
    """
    predictions = []
    for f in Path(inference_dataset).iterdir():
        return None    

