from ultralytics import YOLO
from tetra.model import inference
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def eval_model(path, data_yaml, iou):
    # Load the model
    #val3 lyrurus_yolov11m_new_dataset_iou=0.5
    #val4 lyrurus_yolov11m_new_dataset
    model = YOLO(path)

    # Run the evaluation with proper params
    return model.val(
        data=data_yaml,
        split="val",
        verbose=True,
        save=True,
        save_txt=True,
        save_conf=True,
        plots=True,
        device="mps",
        iou=iou
    )

def eval_hr_images(path):
    # Charger annotations ground truth
    coco_gt = COCO("path/to/ground_truth.json")

    # Charger résultats prédits (format COCO)
    coco_dt = coco_gt.loadRes("path/to/predictions.json")

    # Lancer l’évaluation
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


