from ultralytics import YOLO

# Load the model
model = YOLO("../../../../runs/train/lyrurus_yolov8m_new_dataset/weights/best.pt")

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


