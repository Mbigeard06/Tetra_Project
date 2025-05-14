from sahi.slicing import slice_coco

slice_height = 640
slice_width = 640
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2

image_dir = f'../../../../dataset/og/images'
annotation_path = f'../../../../dataset/og/_annotations.coco.json'
output_dir = f'../../../../dataset/sliced/images'
output_annotation_path = f'../../../../dataset/sliced/_annotations.coco.json'

slice_coco(
    coco_annotation_file_path=annotation_path,
    ignore_negative_samples=False,
    image_dir=image_dir,
    output_coco_annotation_file_name=output_annotation_path,
    output_dir=output_dir,
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_height_ratio,
    overlap_width_ratio=overlap_width_ratio,
    verbose=True,
    min_area_ratio=0.1
)


