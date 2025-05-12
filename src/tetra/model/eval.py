from ultralytics import YOLO

# Load the model
model = YOLO("../weights/train/yolov11l/best.pt")

# Run the evaluation with proper params
results = model.val(
    data="../../../dataset_sliced/split_1/split_1_dataset.yaml",
    split="val",
    verbose=True,
    save=True,
    save_txt=True,
    save_conf=True,
    plots=True,
    device="mps",
    iou=0.5
)


# Print specific metrics
print(f"F1-score global : {results.box.f1:.4f}")
print(f"mAP@0.5        : {results.box.map50:.4f}")
print(f"mAP@0.5:0.95   : {results.box.map:.4f}")
print(f"Précision      : {results.box.precision:.4f}")
print(f"Rappel         : {results.box.recall:.4f}")