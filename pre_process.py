__author__ = 'angad'

from os import walk, path
import pandas as pd
import numpy as np
from scipy import misc


def get_image_filenames(dir):
    # Output: List of filenames with .jpeg extension
    # Input: Relative directory path to search for filenames
    filenames = []
    for (dp, dn, fs) in walk(dir):
        for f in fs:
            if '.jpeg' in f:
                filenames += [f]
    return filenames


def extract(m, dimension):
    #m = 5000
    #dimension = 512
    size = str(m)+'-'+str(dimension)+'x'+str(dimension)
    dir = 'processed/data/train-'+size+'/'
    filenames = get_image_filenames(dir)

    trainLabels = pd.read_csv('data/trainLabels.csv', index_col=0)

    images = []
    y = []
    print "Importing images..."
    for (i, filename) in enumerate(filenames):
        #print i
        image = misc.imread(path.join(dir, filename), flatten=1)
        (r, c) = image.shape
        image = image.reshape(r*c)
        images += [image]
        y_val = trainLabels.loc[filename.split('.')[0]][0]
        y += [y_val]
    print "Converting images to a numpy array..."
    images = np.array(images)
    print "Saving images"
    np.save('images-'+size, images)
    print "Converting y to a numpy array..."
    y = np.array(y)
    print "Saving y"
    np.save('y-'+size, y)
    print "Done."
    print 'Image binary and Y values saved: images-'+size+' and y-'+size
    return (images, y)
