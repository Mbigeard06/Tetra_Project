from pathlib import Path
from PIL import Image
import json

def count_files(path):
    path = Path(path)
    return sum(1 for f in path.iterdir() if f.is_file()) 

def asser_dim(path, dim):
    path = Path(path)
    for f in path.iterdir():
        if f.is_file():
            with Image.open(f) as img:
                if img.size != (dim[0], dim[1]):
                    raise ValueError(f"Image {f.name} has size {img.size}, expected")

def formats(path):
    path = Path(path)
    return {f.suffix.lower() for f in path.glob("*") if f.is_file()}



print(count_files("../dataset_sliced/images"))
#assert nb_annotated("../dataset/train/_annotations.coco.json") <= nb_annotated("../dataset_sliced/train/_annotations.coco.json")
#print(formats("../dataset_sliced"))
