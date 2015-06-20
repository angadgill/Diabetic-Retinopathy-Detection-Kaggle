__author__ = 'angad'

import numpy as np
from sklearn.decomposition import IncrementalPCA
from scipy import misc
# from matplotlib import pyplot as plt

from os import walk, path

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
    size='1000-256x256'
    dir = 'processed/data/train-'+size+'/'
    filenames = get_image_filenames(dir)

    images = []
    for (i, filename) in enumerate(filenames):
        print i
        image = misc.imread(path.join(dir, filename), flatten=1)
        (r, c) = image.shape
        image = image.reshape(r*c)
        # np.save('saved-images/image'+str(i)+'-1000-1000x1000', image)
        images += [image]
    images = np.array(images)
    np.save('images-'+size, images)
    print 'Image binary saved: images-'+size

    n_components = 40
    batch_size = 100
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    images40 = ipca.fit_transform(images)
    print "PCA done"
    np.save('images40-'+size, images40)
    print 'Image binary with PCA saved: images40-'+size


if __name__ == "__main__":
    main()
