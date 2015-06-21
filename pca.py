__author__ = 'angad'

import numpy as np
from sklearn.decomposition import PCA
# from sklearn.decomposition import IncrementalPCA


def fit_transform(m, dimension, images, n_components):
    #m = 5000
    #dimension = 512
    size = str(m)+'-'+str(dimension)+'x'+str(dimension)
    #print "Loading images data..."
    #images = np.load('images-'+size+'.npy')
    #n_components = 40
    # batch_size = 100
    # pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    pca = PCA(n_components=n_components)
    print "Running PCA..."
    images_reduced = pca.fit_transform(images)
    print "Done."
    np.save('images'+str(n_components)+'-'+size, images_reduced)
    print 'Image binary with PCA saved: images'+str(n_components)+'-'+size
    return images_reduced
