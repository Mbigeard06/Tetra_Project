from pathlib import Path
from PIL import Image
import json

def count_files(path, extensions):
    path = Path(path)
    return sum(1 for f in path.iterdir() if f.is_file() and f.suffix.lower() in extensions) 

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


print(count_files("../../dataset/sliced/images", [".png"]))
print(count_files("../../dataset/2025-05-14_5-Fold_Cross-val/split_1/val/images", [".jpg", "png"]))
