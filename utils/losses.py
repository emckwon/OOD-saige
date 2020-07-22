
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Loss functions."""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Return zero
def dont_care(logits, targets, cfg):
    return {
        'loss': 0,
    }

# only care in-distribution's cross entropy loss.
def cross_entropy_in_distribution(logits, targets, cfg):
    """
    Cross entropy loss when logits include outlier's logits also.(ignore outlier's)
    """
    return {
        'loss': F.cross_entropy(logits[:len(targets)], targets),
    }

def outlier_exposure(logits, targets, cfg):
    loss = F.cross_entropy(logits[:len(targets)], targets)
    probs = F.softmax(logits, dim=1)
    prob_diff_out = probs[len(targets):][:] - (1/logits.size(1))
    loss += cfg['oe_weight'] * torch.sum(torch.abs(prob_diff_out))
    return {
        'loss': loss,
    }


# Add new loss here!!!

_LOSSES = {
               "cross_entropy_in_distribution": cross_entropy_in_distribution,
               "dont_care": dont_care,
               "oe": outlier_exposure, 
           }

def getLoss(cfg):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    loss_name = cfg['loss']
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]


