import os 
import shutil

dataset_path = "../dataset"
images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")

os.makedirs(images_path, exist_ok=True)
os.makedirs(labels_path, exist_ok=True)

for filename in os.listdir(dataset_path):
    file_path = os.path.join(dataset_path, filename)

    if filename.endswith('.jpg') or filename.endswith('.png'):
        shutil.move(file_path, os.path.join(images_path, filename))
    elif filename.endswith('.txt'):
        shutil.move(file_path, os.path.join(labels_path, filename))



