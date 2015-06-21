__author__ = 'angad'

import pre_process
import pca
import svm
import kappa


def main():
    m = 500
    dimension = 256
    (images, y) = pre_process.extract(m, dimension)

    n_components = 40
    images_reduced = pca.fit_transform(m, dimension, images, n_components)

    (pred, svm_score) = svm.predict(m, dimension, images_reduced, y)

    kappa_score = kappa.score(y, pred)

    print "kappa score: ", kappa_score
    print "svm score: ", svm_score


if __name__ == '__main__':
    main()