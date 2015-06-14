__author__ = 'angad'

from scipy import misc
from matplotlib import pyplot as plt


# Import image as grayscale
eye_image = misc.imread('sample/10_left.jpeg', flatten=1)

# Show grayscale image
plt.imshow(eye_image, cmap=plt.cm.gray)
plt.show()
