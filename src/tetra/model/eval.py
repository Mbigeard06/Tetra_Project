from ultralytics import YOLO

# Load the model
model = YOLO("yolo11n.pt")

# Run the evaluation
results = model.val(data="../../../dataset_sliced/2025-05-02_5-Fold_Cross-val/split_1/split_1_dataset.yaml")

# Print specific metrics
print("Class indices with average precision:", results.ap_class_index)
print("Average precision for all classes:", results.box.all_ap)
print("Mean average precision at IoU=0.50:", results.box.map50)
print("Mean recall:", results.box.mr)