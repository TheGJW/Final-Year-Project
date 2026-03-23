import cv2
import numpy as np
from sklearn.cluster import KMeans

# predefined color palette in LAB color space
COLOR_PALETTE = {
    "black": [20, 128, 128],
    "white": [230, 128, 128],
    "red": [136, 208, 195],
    "green": [180, 70, 170],
    "blue": [82, 207, 20],
    "yellow": [233, 80, 114],
    "brown": [120, 150, 160],
    "gray": [150, 128, 128]
}
# function to segment the main object from the background
def segment_object(image_path):
    # read img from file
    img = cv2.imread(image_path)
    # initialize mask (same height and width as image)
    mask = np.zeros(img.shape[:2], np.uint8)
    # allocate memory for background and foreground models
    bg = np.zeros((1,65), np.float64)
    fg = np.zeros((1,65), np.float64)
    # define rectangle slightly inside the image borders
    rect = (5,5,img.shape[1]-10,img.shape[0]-10)
    # algorithm to seperate the foreground from background
    cv2.grabCut(img, mask, rect, bg, fg, 5, cv2.GC_INIT_WITH_RECT)
    # convert mask: background to foreground
    mask2 = np.where((mask==2) | (mask==0),0,1).astype('uint8')
    # apply mask to image to keep only foreground object
    segmented = img * mask2[:,:,np.newaxis]

    return segmented

# function to detect dominant color using clustering
def detect_dominant_color(segmented_img):
    # convert image from BGR to LAB color space
    lab = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2LAB)
    # reshape image into list of pixels
    pixels = lab.reshape((-1,3))
    # remove very dark pixels (where those are likely background/noise)
    pixels = pixels[pixels[:,0] > 40]
    # if too few pixels remain, return default value
    if len(pixels) < 3:
        return np.array([0,0,0])
    # applying kMeans clustering to group clusters
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(pixels)

    # getting cluster labels for each pixel
    labels = kmeans.labels_
    # count number of pixels in each cluster
    counts = np.bincount(labels)
    # select the cluster center with the highest count(dominant color)
    dominant = kmeans.cluster_centers_[np.argmax(counts)]

    return dominant


def lab_to_color_name(lab_color):
    # converting input to numpy array
    lab_color = np.array(lab_color)
    # initialize minimum distance and default color
    min_dist = float("inf")
    closest_color = "unknown"
    # loop through predefined color palette
    for color_name, ref in COLOR_PALETTE.items():
        # convert reference color to numpy array
        ref = np.array(ref)
        # compute euclidean distance in LAB space
        dist = np.linalg.norm(lab_color - ref)
        # update closest color if distance is smaller
        if dist < min_dist:
            min_dist = dist
            closest_color = color_name
    # return closest matching color name
    return closest_color