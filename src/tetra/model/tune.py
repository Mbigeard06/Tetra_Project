from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11m.pt")

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(
    data="../../../dataset_sliced/2025-05-02_5-Fold_Cross-val/split_1/split_1_dataset.yaml",
    imgsz=640,
    epochs=10,
    #number of hyperparameter sets tested
    iterations=10,
    batch=16,
    optimizer='AdamW',
    project="../runs/tune",
    name="tune1",
    exist_ok=True,
    verbose=True,
    cos_lr=True,
    seed=42
)
