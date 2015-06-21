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


def main():
    m = 5000
    dimension = 1024
    size = str(m)+'-'+str(dimension)+'x'+str(dimension)
    dir = 'processed/data/train-'+size+'/'
    filenames = get_image_filenames(dir)

    trainLabels = pd.read_csv('data/trainLabels.csv', index_col=0)

    images = []
    y = []
    for (i, filename) in enumerate(filenames):
        print i
        image = misc.imread(path.join(dir, filename), flatten=1)
        (r, c) = image.shape
        image = image.reshape(r*c)
        images += [image]
        y_val = trainLabels.loc[filename.split('.')[0]][0]
        y += [y_val]
    images = np.array(images)
    y = np.array(y)
    np.save('images-'+size, images)
    np.save('y-'+size, y)
    print 'Image binary and Y values saved: images-'+size+' and y-'+size


if __name__ == "__main__":
    main()
