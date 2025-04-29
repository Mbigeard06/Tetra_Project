from sahi.slicing import slice_coco

slice_height = 640
slide_width = 640
overlap_height_ratio = 0.2
overlap_width_ratio =0.2

subsets =  ['train', 'val', 'test']

for subset in subsets:
    image_dir = f'dataset/{subset}'
    annotation_path = f'dataset/{subset}/_annotations.coco.json'
    output_dir = f'dataset_sliced/{subset}'
    output_annotation_path = f'data_sliced/{subset}/_annotations.coco.json'

    slice_coco(
        coco_annotation_file_path=annotation_path,
        image_dir=image_dir,
        output_coco_annotation_file_name=output_annotation_path,
        output_dir=output_dir,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        verbose=True
    )