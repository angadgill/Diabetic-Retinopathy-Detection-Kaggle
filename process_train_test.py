__author__ = 'angad'

import pre_process
import pca
import svm
from quadratic_weighted_kappa import quadratic_weighted_kappa


def do(m, dimension, n_components, FIX_INVERTED=True, FIX_RIGHT_LEFT=True, SAVE=True):
    #m = 1000
    #dimension = 256
    (images, y) = pre_process.extract(m, dimension, FIX_INVERTED, FIX_RIGHT_LEFT, SAVE)

    #n_components = 100
    images_reduced = pca.fit_transform(m, dimension, images, n_components, SAVE)

    (pred, svm_score) = svm.predict(m, dimension, images_reduced, y, SAVE)

    kappa_score = quadratic_weighted_kappa(pred, y, min_rating=0, max_rating=4)

    print "kappa score: ", kappa_score
    print "svm score: ", svm_score
