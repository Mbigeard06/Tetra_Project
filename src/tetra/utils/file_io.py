from PIL import Image

def coco_to_yolo(annotations, image_path):
    image = Image.open(image_path)
    img_width, img_height = image.size
    yolo_labels = []

    for ann in annotations:
        x, y, w, h = ann["bbox"]
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        class_id = ann["category_id"]
        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    return yolo_labels