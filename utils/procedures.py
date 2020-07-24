"""
Code form dlmacedo/isoropic-maximization-loss-and-entropic-score
"""

import os
import pickle
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch
import torch.nn.functional as F
import csv
import numpy as np
from sklearn import metrics


def compute_weights(iterable):
    return [sum(iterable) / (iterable[i] * len(iterable)) if iterable[i] != 0 else float("inf") for i in range(len(iterable))]


def print_format(iterable):
    #return ["{0:.8f}".format(i) if i is not None else "{0}".format(i) for i in iterable]
    return ["{0:.8f}".format(i) if i is not float("inf") else "{0}".format(i) for i in iterable]


def probabilities(outputs):
    return F.softmax(outputs, dim=1)


def max_probabilities(outputs):
    return F.softmax(outputs, dim=1).max(dim=1)[0]


def predictions(outputs):
    #return outputs.max(dim=1)[1]
    return outputs.argmax(dim=1)


def predictions_total(outputs):
    #print(outputs.argmax(dim=1))
    return outputs.argmax(dim=1).bincount(minlength=outputs.size(1)).tolist()


def entropies(outputs):
    probabilities_log_probabilities = F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1)
    # we could make a weighted sum to compute entropy!!! I think we should use expande_as... or boradcast??? Weighted Entropy!!!
    #probabilities_log_probabilities * weights.expand_as(probabilities_log_probabilities)
    return -1.0 * probabilities_log_probabilities.sum(dim=1)


def entropies_grads(outputs):
    entropy_grads = - (1.0 + F.log_softmax(outputs, dim=1))
    #entropy_grads = - (1.0 + F.log_softmax(outputs, dim=1)) * F.softmax(outputs, dim=1) * (1.0 - (F.softmax(outputs, dim=1)))
    return entropy_grads.sum(dim=0).tolist()


def cross_entropies(outputs, targets):
    """ New function... """
    return - 1.0 * F.log_softmax(outputs, dim=1)[range(outputs.size(0)), targets]


def cross_entropies_grads(outputs, targets):
    """ quando tiver targets... ou self-targets... kkkkkk... """
    #cross_entropies_grads = [0 for i in range(len(predictions_total(outputs)))]
    cross_entropies_grads = [0 for i in range(outputs.size(1))]
    for i in range(len(predictions(outputs))):
        #cross_entropies_grads[predictions(outputs)[i]] += - (1.0 / (F.softmax(outputs, dim=1)[i, targets[i]].item()))
        cross_entropies_grads[predictions(outputs)[i]] += - (1.0 - (F.softmax(outputs, dim=1)[i, targets[i]].item()))
    return cross_entropies_grads


def entropies_from_probabilities(probabilities):
    #eps = float(torch.finfo(torch.float32).eps)
    probabilities_log_probabilities = probabilities * torch.log(probabilities + float(torch.finfo(torch.float32).eps))
    return -1.0 * probabilities_log_probabilities.sum(dim=1)


def save_object(object, path, file):
    with open(os.path.join(path, file + '.pkl'), 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)


def load_object(path, file):
    with open(os.path.join(path, file + '.pkl'), 'rb') as f:
        return pickle.load(f)


def save_dict_list_to_csv(dict_list, path, file):
    with open(os.path.join(path, file + '.csv'), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dict_list[0].keys())
        writer.writeheader()
        for dict in dict_list:
            writer.writerow(dict)


def load_dict_list_from_csv(path, file):
    dict_list = []
    with open(os.path.join(path, file + '.csv'), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for dict in reader:
            dict_list.append(dict)
    return dict_list


class MeanMeter(object):
    """Computes and stores the current averaged current mean"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def add(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asinh(x):
    return torch.log(x+(x**2+1)**0.5)


def acosh(x):
    return torch.log(x+(x**2-1)**0.5)


def atanh(x):
    return 0.5*torch.log(((1+x)/((1-x)+0.000001))+0.000001)


def sinh(x):
    return (torch.exp(x)-torch.exp(-x))/2


def euclidean_distances(features, prototypes, pnorm):
    return F.pairwise_distance(features.unsqueeze(2), prototypes.t().unsqueeze(0), p=pnorm)


def mahalanobis_distances(features, prototypes, precisions):
    diff = features.unsqueeze(2) - prototypes.t().unsqueeze(0)
    diff2 = features.t().unsqueeze(0) - prototypes.unsqueeze(2)
    precision_diff = torch.matmul(precisions.unsqueeze(0), diff)
    extended_product = torch.matmul(diff2.permute(2, 0, 1), precision_diff)
    mahalanobis_square = torch.diagonal(extended_product, offset=0, dim1=1, dim2=2)
    mahalanobis = torch.sqrt(mahalanobis_square)
    return mahalanobis


def multiprecisions_mahalanobis_distances(features, prototypes, multiprecisions):
    mahalanobis_square = torch.Tensor(features.size(0), prototypes.size(0)).cuda()
    for prototype in range(prototypes.size(0)):
        diff = features - prototypes[prototype]
        multiprecisions.unsqueeze(0)
        diff.unsqueeze(2)
        precision_diff = torch.matmul(multiprecisions.unsqueeze(0), diff.unsqueeze(2))
        product = torch.matmul(diff.unsqueeze(1), precision_diff).squeeze()
        mahalanobis_square[:, prototype] = product
    mahalanobis = torch.sqrt(mahalanobis_square)
    return mahalanobis