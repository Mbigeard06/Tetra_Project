import cv2
import os
import matplotlib.pyplot as plt


#dataset path
dataset_path = "../dataset/images"

images = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]

#Store number of image of each size
size_counts = {}

for image in images:
    #Get the images
    image_path = os.path.join(dataset_path, image)
    try:
        image = cv2.imread(image_path)
        if image is None:
            continue
        height, width = image.shape[:2]
        dimension = (width, height)
        size_counts[dimension] = size_counts.get(dimension, 0) + 1
    except Exception as e:
        print(f"Erreur avec l'image {image_path}: {e}")


for dimension, count in size_counts.items():
    print(f"{dimension}: {count} images")

    

