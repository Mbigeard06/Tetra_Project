from ultralytics import YOLO

# Load a base model (or your previously trained checkpoint)
model = YOLO('yolo11m.pt')  # or 'runs/detect/train/weights/best.pt'

# Train (finetune) with evolved hyperparameters
model.train(
    data="../../../dataset_sliced/2025-05-02_5-Fold_Cross-val/split_1/split_1_dataset.yaml",
    epochs=100,
    time = 15,
    imgsz=640,
    patience = 10,
    batch=16,
    save = True,
    device="cpu",
    project="./runs/train",
    name="lyrurus_finetuned",
    exist_ok=True,
    pretrained=True,
    optimizer="AdamW",
    single_cls=True,
    multi_scale = False,
    cos_lr = True,
    fraction = 0.1, #TO change
    #freeze to change for fine tuning,
    lr0 = 1e-3,
    lrf = 0.01,
    plots = True
)
