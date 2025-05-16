from PIL import Image
import os
from tetra.model import inference
from tetra.utils import file_io, viewer
import sahi
import matplotlib.pyplot as plt
from pathlib import Path
import importlib

#path
image_path = os.path.abspath("../../dataset/og/images")  
label_dir = os.path.abspath("../../outputs/labels")
class_file = os.path.abspath("../classes.txt")


def display_images(image_dir, label_dir, class_file, n):
    """
    Display all the images associated with the labels of the dir.
    Usefull if you want to see the result of your predictions but if
    you want to make assumptions on your preidction metrics you shloud
    perform real calculations with pycocotools.
    """
    #Get the images
    labels = list(Path(label_dir).glob("*.txt"))[:n]
    for label in labels:
        if label.suffix.lower() == ".txt":
            print(label.stem + ".jpg")
            image = viewer.view_image(os.path.join(image_dir, label.stem + ".jpg"), label, class_file,0.385)
            plt.imshow(image)
            plt.title(label.stem)
            plt.axis("off")
            plt.show()

display_images(image_dir=image_path, label_dir=label_dir, class_file=class_file,n=10)