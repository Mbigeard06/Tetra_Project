import os
import cv2
import matplotlib.pyplot as plt

def draw_boxes(img_path, label_path, class_names):

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

# Chemins
image_dir = "../dataset_sliced/images"
label_dir = "../dataset_sliced/labels"
class_file = "../classes.txt"

# Récupérer uniquement les fichiers image
all_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")]

# Visualiser les 5 premières images
for filename in all_files[4995:5000]:
    image_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")

    image_with_boxes = draw_boxes(image_path, label_path, class_file)

    plt.imshow(image_with_boxes)
    plt.title(filename)
    plt.axis("off")
    plt.show()