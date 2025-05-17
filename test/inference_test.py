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


def batch_inference(model, inference_dataset, label_dir, file_to_id, exts, output_json_path):
    """
    Perform inference, convert predictions to COCO, and stream to JSONL file.

    Args:
        model : Path to the model weights
        inference_dataset: path to the dataset
        label_dir : where to store YOLO-format predictions
        file_to_id : dict mapping filenames to image IDs
        exts : list of accepted file extensions (ex: [".png"])
        output_json_path : path to the output JSON file
    """

    os.makedirs(Path(output_json_path).parent, exist_ok=True)
    with open(output_json_path, "w") as out_json:
        i = 0
        for f in Path(inference_dataset).iterdir():
            if f.suffix.lower() in exts:
                prediction = inference.solo_inference(model, str(f))
                prediction = prediction.to_coco_annotations()

                # Save YOLO predictions
                yolo_pred = file_io.coco_to_yolo(prediction, f, True)
                file_io.save_yolo_labels(yolo_pred, label_dir, f.stem + ".txt")

                # Add predictions to the json
                for ann in prediction:
                    ann["image_id"] = file_to_id[f.name]
                    ann["category_name"] = "Lyrurus_tetrix"
                    ann["category_id"] = int(ann["category_id"]) + 1
                    json.dump(ann, out_json)
                    out_json.write("\n")

                i += 1
                if i % 100 == 0:
                    print(f"Processing image {i} ...")



model_path = "../../runs/train/lyrurus_yolov11l_background_dataset/weights/best.pt"

#Load the model
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    device="mps", 
    confidence_threshold=0.39
    )


coco_gt = COCO("../../dataset/og/_annotations.coco.json")
file_to_id = {img["file_name"]: img["id"] for img in coco_gt.dataset["images"]}
batch_inference("../../runs/train/lyrurus_yolov11l_background_dataset/weights/best.pt", "../../dataset/og/images", "../../dataset/inference/labels/lyrurus_yolov11l_background", file_to_id, [".jpg"], "../../dataset/inference/labels/lyrurus_yolov11l_background/lyrurus_yolov11l_background_dataset.coco.json")
#file_io.json_to_coco("../../dataset/inference/labels/lyrurus_yolov11l_background/lyrurus_yolov11l_background_dataset.coco.json", "../../dataset/inference/labels/lyrurus_yolov11l_background/lyrurus_yolov11l_background_dataset2.coco.json")
#predictions = val.eval_hr("../../dataset/og/_annotations.coco.json", "../../dataset/inference/labels/lyrurus_yolov11l_background/lyrurus_yolov11l_background_dataset2.coco.json")
#predictions.evaluate()
#predictions.accumulate()
#predictions.summarize()