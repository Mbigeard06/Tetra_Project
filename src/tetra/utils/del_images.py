from pathlib import Path

def remove_imgs_without_annotations(img_dir, label_dir, extensions={".png", ".jpg", ".jpeg"}):
    """
    Delete images without annotations
    """
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)

    for img_file in img_dir.iterdir():
        if img_file.suffix.lower() in extensions:
            label_file = label_dir / img_file.with_suffix(".txt").name
            if not label_file.exists() or label_file.stat().st_size == 0:
                print(f"🗑️ Suppression: {img_file.name} (No annotations)")
                img_file.unlink()  # Supprime l’image

