# several functions are from https://github.com/xingjunm/lid_adversarial_subspace_detection
from __future__ import print_function
import numpy as np
import os
import utils.calculate_log as callog

from scipy.spatial.distance import pdist, cdist, squareform


def block_split(X, Y, out):
    """
    Split the data training and testing
    :return: X (data) and Y (label) for training / testing
    """
    num_samples = X.shape[0]
    partition=num_samples//2

    X_train = X[:partition]
    Y_train = Y[:partition]

    X_test = X[partition:]
    Y_test = Y[partition:]

    return X_train, Y_train, X_test, Y_test


def block_split_adv(X, Y):
    """
    Split the data training and testing
    :return: X (data) and Y (label) for training / testing
    """
    num_samples = X.shape[0]
    partition = int(num_samples / 3)
    X_adv, Y_adv = X[:partition], Y[:partition]
    X_norm, Y_norm = X[partition: 2*partition], Y[partition: 2*partition]
    X_noisy, Y_noisy = X[2*partition:], Y[2*partition:]
    num_train = int(partition*0.1)
    X_train = np.concatenate((X_norm[:num_train], X_noisy[:num_train], X_adv[:num_train]))
    Y_train = np.concatenate((Y_norm[:num_train], Y_noisy[:num_train], Y_adv[:num_train]))

    X_test = np.concatenate((X_norm[num_train:], X_noisy[num_train:], X_adv[num_train:]))
    Y_test = np.concatenate((Y_norm[num_train:], Y_noisy[num_train:], Y_adv[num_train:]))

    return X_train, Y_train, X_test, Y_test

def detection_performance(regressor, X, Y, outf, split, adv_noise, random_noise_size):
    """
    Measure the detection performance
    return: detection metrics
    """
    num_samples = X.shape[0]
    l1 = open(os.path.join(outf,'confidence_TMP_In_{}.txt'.format(split)), 'w')
    l2 = open(os.path.join(outf,'confidence_TMP_Out_{}.txt'.format(split)), 'w')
    y_pred = regressor.predict_proba(X)[:, 1]

    for i in range(num_samples):
        if Y[i] == 0:
            l1.write("{}\n".format(-y_pred[i]))
        else:
            l2.write("{}\n".format(-y_pred[i]))
    l1.close()
    l2.close()
    results = callog.new_metric(outf, split, ['TMP'], adv_noise, random_noise_size)
    return results

def get_histogram_front(regressor, X, Y, outf, split, adv_noise, random_noise_size):
    """
    Draw Histogram
    """
    callog.get_histogram(outf, split, ['TMP'], adv_noise, random_noise_size)
    
def load_characteristics(score, dataset, out, outf, split):
    """
    Load the calculated scores
    return: data and label of input score
    """
    X, Y = None, None
    
    file_name = os.path.join(outf, "%s_%s_%s_%s.npy" % (score, dataset.replace("/","_"), out.replace("/","_"),split))
    data = np.load(file_name)
    
    if X is None:
        X = data[:, :-1]
    else:
        X = np.concatenate((X, data[:, :-1]), axis=1)
    if Y is None:
        Y = data[:, -1] # labels only need to load once
         
    return X, Y