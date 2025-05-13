from sahi.predict import get_sliced_prediction
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
import json
from tetra.utils import file_io 
from sahi.predict import predict

def inference(model_path, image_path):
    #Load the model
    detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
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

def predict_batch(model_path, images_dir):
    return predict(
        model_type="ultralytics",
        model_path=model_path,
        model_device="cpu",  
        model_confidence_threshold=0.378,
        source=images_dir,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )
