
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Loss functions."""

import torch.nn as nn
import torch.nn.functional as F

_LOSSES = {
               "cross_entropy_in_distribution": cross_entropy_in_distribution,
               "dont_care": dont_care,
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

# Return zero
def dont_care(logits, targets):
    return 0

# only care in-distribution's cross entropy loss.
def cross_entropy_in_distribution(logits, targets):
    return F.cross_entropy(logits[:len(targets)], targets)
