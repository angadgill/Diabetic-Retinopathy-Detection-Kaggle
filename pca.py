__author__ = 'angad'

import numpy as np
from sklearn.decomposition import PCA
# from sklearn.decomposition import IncrementalPCA


def main():
    m = 5000
    dimension = 1024
    size = str(m)+'-'+str(dimension)+'x'+str(dimension)
    images = np.load('images-'+size+'.npy')
    n_components = 40
    # batch_size = 100
    # pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    pca = PCA(n_components=n_components)
    images40 = pca.fit_transform(images)
    print "PCA done"
    np.save('images40-'+size, images40)
    print 'Image binary with PCA saved: images40-'+size


if __name__ == "__main__":
    main()
