
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Loss functions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.procedures as utils

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

def outlier_exposure_confidence_control(logits, targets, cfg):
    loss = F.cross_entropy(logits[:len(targets)], targets)
    ## OECC Loss Function
    A_tr = cfg['train_acc'] # maximum Training Accuracy of given training set
    sm = torch.nn.Softmax(dim=1) # Create a Softmax 
    probabilities = sm(logits) # Get the probabilites for both In and Outlier Images
    max_probs, _ = torch.max(probabilities, dim=1) # Take the maximum probabilities produced by softmax
    prob_diff_in = max_probs[:len(targets)] - A_tr # Push towards the training accuracy of the baseline
    loss += cfg['lambda_1'] * torch.sum(prob_diff_in**2) ## 1st Regularization term
    prob_diff_out = probabilities[len(targets):][:] - (1/num_classes)
    loss += cfg['lambda_2'] * torch.sum(torch.abs(prob_diff_out)) ## 2nd Regularization term
                            
    return {
        'loss': loss,
    }    
    
    
def learning_confidence_loss(logits, targets, cfg):
    lamda = cfg['lamda']
    num_classes=cfg.cfg['model']['num_classes']

    labels_onehot=encode_onehot(targets,num_classes)

    pred_original=logits[:,:-1]
    confidence=logits[:,-1:]

    pred_original=F.softmax(pred_original)
    confidence=F.sigmoid(confidence)

    eps=1e-12
    pred_original=torch.clamp(pred_original,0.+eps, 1.-eps)
    confidence=torch.clamp(confidence,0.+eps,1.-eps)
    
    # Randomly set half of the confidences to 1 (i.e. no hints)
    b = torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)).cuda()
    conf = confidence * b + (1 - b)
    #print(conf)
    #print(labels_onehot)
    #print(conf.expand_as(pred_original))
    pred_new = pred_original * conf.expand_as(pred_original) + labels_onehot * (1 - conf.expand_as(labels_onehot))
    pred_new = torch.log(pred_new)
    
    xentropy_loss=cross_entropy_in_distribution(pred_new, targets)
    confidence_loss=torch.mean(-torch.log(confidence))
    
    total_loss=xentropy_loss+ (lmbda*confidence_loss)
    
    if cfg['budget']>confidence_loss:
        lmbda=lmbda/1.01
    elif cfg['budget']<=confidence_loss:
        lmbda=lmbda/0.99
            
    return {
        'loss': total_loss,
        'conf_loss': confidence_loss,
        'xentropy_loss': xentropy_loss,
        'lamda': lmbda,
        'confidence': confidence
    }

# IsoMax Loss
# alpha should be configurated.
def GenericLossSecondPart(features, targets, cfg):
    
    alpha = cfg['alpha']
    pnorm = 2
    weights = cfg['model'].classifier.weights
    #print(weights.size())
    #print("alpha {} | pnorm {}".format(alpha, pnorm))

    targets_one_hot = torch.eye(weights.size(0))[targets]

    if features.size(0) != targets_one_hot.size(0):
        targets_one_hot = torch.cat((targets_one_hot, torch.zeros(features.size(0) - targets_one_hot.size(0), weights.size(0))), 0)
    
    targets_one_hot = targets_one_hot.long().cuda()
    
    distances = utils.euclidean_distances(features, weights, pnorm)
    intra_inter_distances = torch.where(targets_one_hot != 0, -distances, distances)
    intra_distances = -intra_inter_distances[intra_inter_distances < 0]
    inter_distances = intra_inter_distances[intra_inter_distances > 0]
    transformed_distances = distances
    regularization = 0

    probabilities_for_training = nn.Softmax(dim=1)(-alpha * transformed_distances)
    probabilities_at_targets = probabilities_for_training[range(len(targets)), targets]
    loss = -torch.log(probabilities_at_targets).mean() + regularization
    probabilities = nn.Softmax(dim=1)(-distances)

    return {
        'loss': loss,
        'distances': -transformed_distances,
        'target_distances': intra_distances.data.cpu().mean(),
        'non_target_distances': inter_distances.data.cpu().mean(),
        'entropies': utils.entropies_from_probabilities(probabilities).tolist()
    }


# Loss for training ova network
def ova_bce_loss(features, targets, cfg):
    features = torch.squeeze(F.sigmoid(features), dim=1)
    targets = (targets == cfg['ova_target']).float()
    loss = F.binary_cross_entropy(features, targets)
    
    return {
        'loss': loss,
    }


def share_ovnni_loss(logits, targets, cfg):
    loss = F.cross_entropy(logits[:len(targets)], targets)
    K = logits.size(1) // 2
    for i in range(K):
        loss += F.binary_cross_entropy(F.sigmoid(logits[:len(targets), K + i]), (targets == i).float())
        
    return {
        'loss': loss 
    }



# Add new loss here!!!

_LOSSES = {
                "cross_entropy_in_distribution": cross_entropy_in_distribution,
                "dont_care": dont_care,
                "oe": outlier_exposure,
                "oecc": outlier_exposure_confidence_control,
                "lc" : learning_confidence_loss,
                "isomax" : GenericLossSecondPart,
                "ova_bce": ova_bce_loss,
                "sovnni": share_ovnni_loss,
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


