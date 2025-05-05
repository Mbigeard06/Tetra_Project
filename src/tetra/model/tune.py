from ultralytics import YOLO
from pathlib import Path

# Load a pretrained YOLO11n model
model = YOLO("yolo11m.pt")


search_space = {
    # Optimization
    "lr0": (1e-5, 1e-1),            # Initial learning rate
    "lrf": (0.01, 1.0),             # Final learning rate multiplier (lr_final = lr0 * lrf)
    "momentum": (0.6, 0.98),        # SGD momentum
    "weight_decay": (0.0, 0.001),   # L2 weight regularization

    # Warmup
    "warmup_epochs": (0.0, 5.0),    # Number of warmup epochs
    "warmup_momentum": (0.0, 0.95), # Starting momentum during warmup
    "warmup_bias_lr": (0.0, 0.2),   # Starting learning rate for bias during warmup

    # Loss components
    "box": (0.02, 0.2),             # Box loss gain
    #"cls": (0.2, 4.0),              # Class loss gain

    # Data augmentation
    "hsv_h": (0.0, 0.1),            # HSV-Hue augmentation
    "hsv_s": (0.0, 0.9),            # HSV-Saturation augmentation
    "hsv_v": (0.0, 0.9),            # HSV-Value augmentation
    #"degrees": (0.0, 45.0),         # Rotation degrees
    #"translate": (0.0, 0.9),        # Translation fraction
    #"scale": (0.0, 0.9),            # Scaling factor
    #"shear": (0.0, 10.0),           # Shear angle
    #"perspective": (0.0, 0.001),    # Perspective transformation
    #"flipud": (0.0, 1.0),           # Vertical flip probability
    "fliplr": (0.0, 1.0),           # Horizontal flip probability
    "mosaic": (0.0, 1.0),           # Mosaic augmentation probability
    "mixup": (0.0, 1.0),            # MixUp augmentation probability
    #"copy_paste": (0.0, 1.0)        # Copy-paste augmentation probability
}

path = Path("C:/Users/bima2564/dataset/split_1_dataset.yaml")

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(
    data= path,
    imgsz=640,
    space = search_space,
    epochs=10,
    #number of hyperparameter sets tested
    iterations=10,
    batch=16,
    optimizer='AdamW',
    project="./runs/tune",
    name="lyrurus_tune",
    exist_ok=True,
    verbose=True,
    cos_lr=True
)
