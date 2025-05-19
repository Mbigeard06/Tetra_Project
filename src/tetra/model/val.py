from ultralytics import YOLO
from tetra.model import inference
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def eval_model(model_path, data_yaml, iou):

    model = YOLO(model_path)

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

def eval_hr(gt, pred):
    """
    Evaluates the predictions of the model on the hr images
    Args:
        gt : ground thruth annotations json path
        pred : prediction in coco json format
    Returns:
        dict: Dictionary of evaluation metrics (AP, AR, etc.).
    """
    #Load gt and preddiction
    coco_gt = COCO(gt)
    coco_preds = coco_gt.loadRes(pred)
    return COCOeval(coco_gt, coco_preds, iouType='bbox')




