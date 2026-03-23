from PIL import Image
# function to crop a region from an image using bounding box coordinates
def crop_box(image_path, box):
    # loading an image from file and ensuring that it is in RGB format
    image = Image.open(image_path).convert("RGB")
    return image.crop((
        box["xmin"],
        box["ymin"],
        box["xmax"],
        box["ymax"]
    ))