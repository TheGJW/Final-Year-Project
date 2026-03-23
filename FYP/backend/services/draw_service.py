import cv2
# function to draw the bounding boxes of an image
def draw_boxes(image_path, boxes, selected_box=None,output_path="output.jpg",show_label=True):
    # read image from file
    image = cv2.imread(image_path)
    # loop through each detected bounding box
    for box in boxes:
        # extract and convert coordinates to integers
        x1 = int(box["xmin"])
        y1 = int(box["ymin"])
        x2 = int(box["xmax"])
        y2 = int(box["ymax"])

        # green color for bounding box 
        color = (0, 255, 0)
        # red for selected
        if selected_box and box == selected_box:
            color = (0, 0, 255)  
        # drawing tectangle on image
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        # display label if enabled and class name exist
        if show_label and "class_name" in box:
            if "class_name" in box:
                cv2.putText(
                    image,
                    box["class_name"],
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

    cv2.imwrite(output_path, image)