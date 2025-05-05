import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob


def view_image(img_path, label_path, class_names):

    with open(class_names, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    if not os.path.exists(label_path):
        print(f"No label for {img_path}")
        return
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            print(parts)
            class_id, x_c, y_c, bw, bh = map(float, parts)
            class_id = int(class_id)

            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, class_names[class_id], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return image

def view_bb_repartition(label_dir, img_size=640, n=500):
    canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    label_files = glob.glob(os.path.join(label_dir, "*.txt"))
    files = label_files[:n]

    for label_file in files:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # skip bad lines

                try:
                    class_id, x_center, y_center, w, h = map(float, parts)
                except ValueError:
                    continue  # skip if conversion fails
                
                # Convert YOLO to pixel coords
                x_center *= img_size
                y_center *= img_size
                w *= img_size
                h *= img_size

                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)

                # Draw rectangle
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 1)

    return canvas

canvas = view_bb_repartition(
    label_dir="../../../dataset_sliced/2025-05-02_5-Fold_Cross-val/split_1/train/labels",
    img_size=640,
    n=500
)

plt.figure(figsize=(6,6))
plt.imshow(canvas)  # Pas besoin de cvtColor (image blanche en RGB)
plt.axis('off')
plt.title("Bounding box distribution (first 500 labels)")
plt.show()


        
