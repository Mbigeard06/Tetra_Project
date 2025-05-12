from pathlib import Path
import os

def remove_imgs_without_annotations(img_dir, label_dir, extensions={".png", ".jpg", ".jpeg"}):
    """
    Delete images without annotations
    """
    for img_file in img_dir.iterdir():
        if img_file.suffix.lower() in extensions:
            label_file = label_dir / img_file.with_suffix(".txt").name
            if not label_file.exists() or label_file.stat().st_size == 0:
                print(f"🗑️ Suppression: {img_file.name} (No annotations)")
                img_file.unlink()  # delete the image


def filter_labels(label_dir, condition_fn):
    total_removed_boxes = 0
    total_deleted_files = 0

    for label_file in label_dir.iterdir():
        if not label_file.name.endswith(".txt"):
            continue

        with open(label_file, "r") as f:
            lines = f.readlines()

        filtered_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x, y, w, h = parts
            if condition_fn(class_id, float(x), float(y), float(w), float(h)):
                filtered_lines.append(line)

        removed = len(lines) - len(filtered_lines)
        total_removed_boxes += removed

        if filtered_lines:
            with open(label_file, "w") as f:
                f.writelines(filtered_lines)
        else:
            label_file.unlink()
            total_deleted_files += 1
            print(f"Deleted empty label: {label_file}")

    print(f"Total removed boxes: {total_removed_boxes}")
    print(f"Total deleted label files: {total_deleted_files}")