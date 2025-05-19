from tetra.model import val
from tetra.model import inference
from tetra.data import data_loader
from tetra.utils import file_io
from pathlib import Path
import importlib
import shutil
import os
from pycocotools.coco import COCO
import json
from sahi import AutoDetectionModel

model_path = "../../runs/train/lyrurus_yolov8m_new_dataset_iou=0.5/weights/best.pt"

#Load the model
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    device="mps", 
    confidence_threshold=0.383
    )


coco_gt = COCO("../../dataset/og/_annotations.coco.json")
file_to_id = {img["file_name"]: img["id"] for img in coco_gt.dataset["images"]}
predictions = inference.batch_inference(detection_model, "../../dataset/og/images", file_to_id, [".jpg"], out_dir=  "../../dataset/inference/labels/lyrurus_yolov8m_new_dataset_iou=0.5")
file_io.save_coco_labels(predictions, "../../dataset/inference/labels/lyrurus_yolov8m_new_dataset_iou=0.5")
#file_io.json_to_coco("../../dataset/inference/labels/lyrurus_yolov11m_new_dataset_iou=0.5/annotations.coco.json", "../../dataset/inference/labels/lyrurus_yolov11m_new_dataset_iou=0.5.2")
#predictions.accumulate()
#predictions.summarize()


