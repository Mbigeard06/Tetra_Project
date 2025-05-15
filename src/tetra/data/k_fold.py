from pathlib import Path
import yaml
import pandas as pd
from collections import Counter
from sklearn.model_selection import KFold
import random
import datetime
from tqdm import tqdm
import shutil


def k_fold(dataset_path, yaml_file, ksplit):
    dataset_path = Path(dataset_path)
    yaml_file = Path(yaml_file)
    labels = sorted(dataset_path.rglob("*labels/*.txt"))
    with open(yaml_file, encoding="utf-8") as y:
        classes = yaml.safe_load(y)["names"]
    cls_idx = list(range(len(classes)))
    print(cls_idx)

    index = [label.stem for label in labels]
    labels_df = pd.DataFrame([], columns=cls_idx, index=index)

    #Count labels => only one class here
    for label in labels:
        lbl_counter = Counter()
        with open(label) as lf:
            lines = lf.readlines()
            for line in lines: 
                lbl_counter[int(line.split(" ")[0])] += 1
        
        labels_df.loc[label.stem] = lbl_counter

    labels_df = labels_df.fillna(0.0)

    random.seed(0)
    kf  = KFold(n_splits=ksplit, shuffle=True, random_state=20)
    kfolds = list(kf.split(labels_df))

    folds = [f"split_{n}" for n in range(1, ksplit + 1)]
    folds_df = pd.DataFrame(index=index, columns=folds)

    for i, (train, val) in enumerate(kfolds, start=1):
        folds_df[f"split_{i}"].loc[labels_df.iloc[train].index] = "train"
        folds_df[f"split_{i}"].loc[labels_df.iloc[val].index] = "val"

    supported_extensions = [".jpg", ".jpeg", ".png"]
    images =  []

    for ext in supported_extensions:
        images.extend(sorted((dataset_path/"images").rglob(f"*{ext}")))

    save_path = Path(dataset_path / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val")
    save_path.mkdir(parents=True, exist_ok=True)
    ds_yamls = []

    for split in folds_df.columns:
        # Create directories
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

        # Create dataset YAML files
        dataset_yaml = split_dir / f"{split}_dataset.yaml"
        ds_yamls.append(dataset_yaml)

        with open(dataset_yaml, "w") as ds_y:
            yaml.safe_dump(
                {
                    "path": split_dir.as_posix(),
                    "train": "train",
                    "val": "val",
                    "names": classes,
                },
                ds_y,
            )

    for image, label in tqdm(zip(images, labels), total=len(images), desc="Copying files"):
        for split, k_split in folds_df.loc[image.stem].items():
            # Destination directory
            img_to_path = save_path / split / k_split / "images"
            lbl_to_path = save_path / split / k_split / "labels"

            # Copy image and label files to new directory (SamefileError if file already exists)
            shutil.copy(image, img_to_path / image.name)
            shutil.copy(label, lbl_to_path / label.name)


dataset_path = "../../../../dataset/dataset2/1"
yaml_file = "../../../../dataset/dataset2/1/dataset2.yaml"
k_fold(dataset_path, yaml_file, 3)

