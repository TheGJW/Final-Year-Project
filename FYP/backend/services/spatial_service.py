def is_on_top(small, large):
    # check horizontal overlap exists two boxes
    horizontal_overlap = (
        min(small["xmax"], large["xmax"]) -
        max(small["xmin"], large["xmin"])
    )
    # if no overlap, cannot be on top
    if horizontal_overlap <= 0:
        return False

    # compute vertical center of the smaller object
    small_center_y = (small["ymin"] + small["ymax"]) / 2
    # check if the center lies within the vertical bounds of the larger object
    vertically_inside = (
        small_center_y >= large["ymin"] and
        small_center_y <= large["ymax"]
    )
    # if both horizontal overlap and vertical condition is met then return true
    return vertically_inside
# checking if two objects are beside each other
def is_beside(boxA, boxB):
    # compute vertical centers of both boxes
    centerA_y = (boxA["ymin"] + boxA["ymax"]) / 2
    centerB_y = (boxB["ymin"] + boxB["ymax"]) / 2
    # check if they are roughly aligned vertically (same horizontal level)
    vertical_alignment = abs(centerA_y - centerB_y) < 50
    # checking if boxes are seperated horizontally (no overlap)
    horizontal_gap = (
        boxA["xmax"] < boxB["xmin"] or
        boxB["xmax"] < boxA["xmin"]
    )
    # return true if both conditions are satisfied
    return vertical_alignment and horizontal_gap

# function to infer spatial relationship between selected object and others
def infer_relation(user_object_name, selected_box, all_boxes, label_conf_threshold=0.5):
    # possible classes in the COCO dataset
    on_support_classes = [
    "bed",
    "couch",
    "chair",
    "dining table",
    "bench",
    "laptop"
    ]

    beside_classes = [
    "handbag",
    "backpack",
    "suitcase",
    "laptop",
    "chair",
    "couch",
    "tv"
    ]

    # check on relations

    # track best supporting object
    best_on_candidate = None
    largest_area = 0

    for other in all_boxes:
        # skipping the selected object itself
        if other == selected_box:
            continue
        # skipping low confidence detections
        if other["confidence"] < label_conf_threshold:
            continue
        # only consider valid support objects
        if other["class_name"] not in on_support_classes:
            continue
        # check if selected object is on top of this object
        if is_on_top(selected_box, other):
            # compute area of the candidate box
            area = (other["xmax"] - other["xmin"]) * (other["ymax"] - other["ymin"])
            # selecting the largest supporting box
            if area > largest_area:
                largest_area = area
                best_on_candidate = other
    # if there is a valid on relation then return it
    if best_on_candidate:
        return f"{user_object_name} on {best_on_candidate['class_name']}"

    # checking besides relations

    # tracking closest object beside
    closest_candidate = None
    smallest_distance = float("inf")

    # loop through all detected objects
    for other in all_boxes:
        # slip the selected object
        if other == selected_box:
            continue
        # skipping the low confidence detections
        if other["confidence"] < label_conf_threshold:
            continue
        # only consider the valid classes for "beside"
        if other["class_name"] not in beside_classes:
            continue
        # check if object is beside each other 
        if is_beside(selected_box, other):

            # compute horizontal distance between boxes
            if selected_box["xmax"] < other["xmin"]:
                distance = other["xmin"] - selected_box["xmax"]
            else:
                distance = selected_box["xmin"] - other["xmax"]
            # select closest object
            if distance < smallest_distance:
                smallest_distance = distance
                closest_candidate = other
    # if a valid "beside" relation is found, return it
    if closest_candidate:
        return f"{user_object_name} beside {closest_candidate['class_name']}"
    # default is no relation found 
    return f"{user_object_name}"