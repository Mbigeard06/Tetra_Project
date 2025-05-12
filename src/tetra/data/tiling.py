from sahi.slicing import slice_coco

slice_height = 640
slice_width = 640
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2

image_dir = f'../dataset/'
annotation_path = f'../dataset/_annotations.coco.json'
output_dir = f'../dataset_sliced'
output_annotation_path = f'../data_sliced/_annotations.coco.json'

slice_coco(
    coco_annotation_file_path=annotation_path,
    ignore_negative_samples=True,
    image_dir=image_dir,
    output_coco_annotation_file_name=output_annotation_path,
    output_dir=output_dir,
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_height_ratio,
    overlap_width_ratio=overlap_width_ratio,
    verbose=True
)
