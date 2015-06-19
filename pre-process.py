__author__ = 'angad'

from scipy import misc
from matplotlib import pyplot as plt
import cv2
from os import walk, path


def crop_image(image, threshold):
    # Crop out the black background from an image
    (rows, columns) = image.shape

    for i in range(rows):
        if image[i].max()>threshold:
            top = i
            break

    for i in reversed(range(rows)):
        if image[i].max()>threshold:
            bottom = i
            break

    image_t = image.transpose()
    for i in range(columns):
        if image_t[i].max()>threshold:
            left = i
            break

    for i in reversed(range(columns)):
        if image_t[i].max()>threshold:
            right = i
            break

    # print top, bottom, left, right

    return image[top:bottom, left:right]


def set_contrast(image, low, high):
    ret, thres = cv2.threshold(image, low, high, cv2.THRESH_BINARY_INV)
    return thres


def get_circle_params(image):
    # Returns (radius, (center_x, center_y)) for the circle
    # expects black circle on white background
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = False
    detector = cv2.SimpleBlobDetector(params)
    [keypoint] = detector.detect(image)
    return (keypoint.size, keypoint.pt)


def get_circle_params_2(image):
    image = image.transpose()
    (rows, columns) = image.shape
    threshold = 15
    for i in range(rows):
        if image[i].max()>threshold:
            left_r = i
            left_c = image[i].argmax()
            break

    for i in reversed(range(rows)):
        if image[i].max()>threshold:
            right_r = i
            right_c = image[i].argmax()
            break

    radius = (right_r - left_r)/2
    center = ((left_r+right_r)/2,(left_c+right_c)/2)
    return (radius, center)


def crop_circle(image, radius, center):
    (x, y) = center
    (rows, columns) = image.shape
    r = radius
    top = int(y - r)
    if top < 0:
        top = 0
    bottom = int(y + r)
    if bottom > rows:
        bottom = rows
    left = int(x - r)
    if left < 0:
        left = 0
    right = int(x + r)
    if right > columns:
        right = columns
    return image[top:bottom, left:right]


def is_inverted(image):
    # Output: True or False based on whether the image is 'inverted'
    # as defined by the Kaggle contest on Diabetic Retinopathy
    # Input: Expects images with equalized histograms as input
    (r_max, c_max) = image.shape
    maxval = image.max()
    (t, image) = cv2.threshold(image, maxval-10, maxval, cv2.THRESH_BINARY)
    (r, c) = image.nonzero()
    inverted = r.mean() > (r_max/2)
    return inverted


def is_inverted_vert(image, filename):
    # Output: True or False based on whether the image is 'inverted'
    # as defined by the Kaggle contest on Diabetic Retinopath
    # This function is based on Ravi's idea of testing
    # to see if the bright spot is on the left or the
    # right of the vertical center line
    # Input: Expects images with equalized histograms as inpu
    (r_max, c_max) = image.shape
    maxval = image.max()
    (t, image) = cv2.threshold(image, maxval-10, maxval, cv2.THRESH_BINARY)
    (r, c) = image.nonzero()
    if 'right' in filename:
        if c.mean() > (c_max/2):
            inverted = False
        else:
            inverted = True
    elif 'left' in filename:
        if c.mean() < (c_max/2):
            inverted = False
        else:
            inverted = True
    else:
        inverted = None
    return inverted


def get_image_filenames(dir):
    # Output: List of filenames with .jpeg extension
    # Input: Relative directory path to search for filenames
    filenames = []
    for (dp, dn, fs) in walk(dir):
        for f in fs:
            if '.jpeg' in f:
                filenames += [f]
    return filenames


def main():
    # Import image as grayscale
    # image = misc.imread('data/sample/16_left.jpeg', flatten=1)
    dir = 'processed/run-normal/data/sample/'
    filenames = get_image_filenames(dir)
    print filenames

    for filename in filenames:
        image = misc.imread(path.join(dir,filename), flatten=1)
        print filename, is_inverted_vert(image, filename)

    # image = misc.imread('processed/run-normal/data/sample/15_left.jpeg', flatten=1)
    # print is_inverted(image)

    # # Resize image to 20% of its size
    # (x,y) = image.shape
    # x = int(x*0.2)
    # y = int(y*0.2)
    # image = misc.imresize(image, (x, y))
    #
    # # image_high_contrast = set_contrast(image, 10, 255)
    #
    # # (radius, center) = get_circle_params(image_high_contrast)
    # (radius, center) = get_circle_params_2(image)
    #
    # print radius, center
    # # Crop black background out of the image
    # # image_cropped = crop_image(image=image, threshold=9)
    # image_cropped = crop_circle(image, radius, center)
    #
    # # Show grayscale image
    # # plt.imshow(image, cmap=plt.cm.gray)
    # plt.imshow(image_cropped, cmap=plt.cm.gray)
    # plt.show()



if __name__ == "__main__":
    main()
