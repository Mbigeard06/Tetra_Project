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
    verbose=False
    )




