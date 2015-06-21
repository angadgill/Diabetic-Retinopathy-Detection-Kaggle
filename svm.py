__author__ = 'angad'

import numpy as np
from sklearn.svm import SVC


def main():
    m = 5000
    dimension = 1024
    size = str(m)+'-'+str(dimension)+'x'+str(dimension)
    images = np.load('images40-'+size+'.npy')
    y = np.load('y-'+size+'.npy')
    clf = SVC()
    data_split = int(m/2)
    clf.fit(images[:data_split],y[:data_split])
    print 'Score:', clf.score(images[data_split:], y[data_split:])

    predictions = np.array((clf.predict(images), y)).T
    np.savetxt('predictions-'+size+'.csv', predictions, delimiter=',', fmt='%d')

if __name__ == "__main__":
    main()
