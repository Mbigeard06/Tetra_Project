import json

def nb_annotations(path):
    with open(path, 'r') as f:
        coco = json.load(f)
        annoted_img_ids = {ann['id'] for ann in coco['annotations']}
    
    return len(annoted_img_ids)

def nb_images(path):
    with open(path, 'r') as f:
        coco = json.load(f)
        imgs_id = {img['id'] for img in coco['images']}
    
    return len(imgs_id)
    


print(nb_images("../../dataset/sliced/_annotations.coco.json"))