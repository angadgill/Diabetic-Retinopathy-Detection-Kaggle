__author__ = 'angad'

import numpy as np
from sklearn.svm import SVC


def main():
    size='1000-256x256'
    image_filename = 'images40-'+size+'.npy'
    images = np.load(image_filename)
    y = np.loadtxt('data/trainLabels.csv', delimiter=',', skiprows=1, usecols=(1,))
    y = y[:1000]
    clf = SVC()
    data_split = 500
    clf.fit(images[:data_split],y[:data_split])
    print clf.score(images[data_split:], y[data_split:])


if __name__ == "__main__":
    main()
