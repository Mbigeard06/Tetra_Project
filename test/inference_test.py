from PIL import Image
import os
from tetra.model import inference
from tetra.utils import file_io, viewer
import sahi
import matplotlib.pyplot as plt
from pathlib import Path
import importlib

#path
image_path = os.path.abspath("../../dataset")  
label_dir = os.path.abspath("../../outputs/labels")
class_file = os.path.abspath("../classes.txt")


def display_images(image_dir, label_dir, class_file):
    #Get the images
    images = list(Path(image_dir).glob("*.jpg"))[:10]
    for img in Path(image_dir).iterdir():
        if img.suffix.lower() == ".jpg":
            print(img.stem + "text")
            image = viewer.view_image(img, os.path.join(label_dir,img.stem + ".txt"), class_file,0.385)
            plt.imshow(image)
            plt.title("Inference")
            plt.axis("off")
            plt.show()

display_images(image_dir=image_path, label_dir=label_dir, class_file=class_file)