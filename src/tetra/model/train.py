from ultralytics import YOLO

# Load a base model (or your previously trained checkpoint)
model = YOLO('yolov8n.pt')  # or 'runs/detect/train/weights/best.pt'

# Train (finetune) with evolved hyperparameters
model.train(
    data="lyrurus.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    hyp="runs/tune/lyrurus_tune/hyp_evolved.yaml",
    name="lyrurus_finetuned",
    project="runs/train",
    device= '0'
)
