import os
import pandas as pd
from pathlib import Path
import random




def df_bb(dir_path, splits=("train", "val")):
    """
    Loads YOLO-format bounding box annotations from a dataset directory 
    and returns a Pandas DataFrame containing all bounding boxes from 
    specified dataset splits.

    Parameters:
    -----------
    dir_path : str
        The root directory of the dataset. This directory should contain
        subdirectories for each dataset split (e.g., "train", "val", "test").
    splits : list of str
        A list of dataset split names to process (e.g., ["train", "val"]).

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with one row per bounding box, containing the following columns:
        - file       : Image filename (derived from annotation file, with .png extension)
        - class_id   : Class index of the object (as string)
        - x_center   : Normalized x-center of the bounding box (as string)
        - y_center   : Normalized y-center of the bounding box (as string)
        - width      : Normalized width of the bounding box (as string)
        - height     : Normalized height of the bounding box (as string)
        - split      : The dataset split name from which the annotation comes

    Notes:
    ------
    - This function expects YOLO-format annotations in a "labels/" subfolder
      within each split directory.
    - Each label file must have one or more lines formatted as:
        <class_id> <x_center> <y_center> <width> <height>
      with all coordinates normalized to [0, 1].
    - Image filenames are inferred by replacing `.txt` with `.png`.
    """
    data = []
    for split in splits:
        split_path = os.path.join(dir_path, split)
        print(split_path)
        labels_path = os.path.join(split_path, "labels")
        if(os.path.isdir(labels_path)):
            for label in os.listdir(labels_path):
                if label.endswith(".txt") :
                    label_file_path = os.path.join(labels_path,label)
                    with open(label_file_path, "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            parts = line.strip().split()
                            #Check line is normalized
                            if len(parts) == 5:
                                class_id, x_center, y_center, width, height = parts
                                data.append({
                                    "file": label.replace(".txt", ".png"),
                                    "class_id": class_id,
                                    "x_center": x_center,
                                    "y_center": y_center,
                                    "width": width,
                                    "height": height,
                                    "split": split
                                })
        else:
            print("repertory not found !")
    return pd.DataFrame(data)

def df_images(dir_path, splits=("train", "val")):
    data = []
    for split in splits:
        split_dir = Path(dir_path) / split
        img_dir = split_dir / "images"
        label_dir = split_dir / "labels"
        for img in os.listdir(img_dir):
            if img.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = img_dir / img
                label_path = label_dir / (Path(img.stem) + ".txt")
            
            if label_path.exists():
                with open(label_path, "r") as f:
                    bbox_count = sum(1 for _ in f)
            else:
                bbox_count = 0
            
            data.append({
                "img" : img,
                "bbox_count": bbox_count,
                "split": split
            }
            )
    return pd.DataFrame(data)

def get_backgrounds(dir):
    background = []
    label_dir = Path(dir) / "labels"
    image_dir = Path(dir) / "images"

    for img in image_dir.iterdir():
        
        if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue  

        label_file = label_dir / img.with_suffix(".txt").name

        if not label_file.exists() or label_file.stat().st_size == 0:
            background.append(img.name)  

    return background

def random_data_sample(dataset_dir, f, extensions):
    """
    Select a random fraction of the dataset based on file extensions.

    Args:
        dataset_dir (str): Directory containing the dataset.
        f (float): Fraction to sample (e.g. 0.2 for 20%).
        extensions (list): List of allowed file extensions (e.g. ['.jpg', '.png']).

    Returns:
        list: List of randomly selected filenames (str).
    """
    dataset_dir = Path(dataset_dir)
    files = [file.name for file in dataset_dir.iterdir() if file.suffix.lower() in extensions]
    nb_samples = int(len(files) * f)
    return random.sample(files, nb_samples)
