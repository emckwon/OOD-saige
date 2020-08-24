
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

def cross_entropy_with_push_cluster(logits, targets, cfg):
    
    ce_loss = F.cross_entropy(logits[:len(targets)], targets)
    weights = cfg['model'].classifier.weights
    
    sim_loss = 0
    num_class = weights.size(0)
    for i in range(num_class):
        temp = F.pairwise_distance(weights, weights[i, :].unsqueeze(0), p=2)
        
        #sim_loss += F.cosine_similarity(weights, weights[i, :].unsqueeze(0)).sum()
        
        for j in range(num_class):
            if i != j:
                sim_loss += temp[j]
    
    loss = ce_loss - cfg['lamda'] * sim_loss
    return {
        'loss': loss,
    }

# IsoMax Loss
# alpha should be configurated.
# def GenericLossSecondPart(features, targets, cfg):
    
#     #alpha = cfg['alpha']
#     pnorm = 2
#     #weights = cfg['model'].classifier.weights
#     #print(weights.size())
#     #print("alpha {} | pnorm {}".format(alpha, pnorm))

    
#     targets_one_hot = targets_one_hot.long().cuda()
#     regularization = 0
#     probabilities_at_targets = probabilities_for_training[range(len(targets)), targets]
#     loss = -torch.log(probabilities_at_targets).mean() + regularization

#     return {
#         'loss': loss,
#     }


# Loss for training ova network
def ova_bce_loss(features, targets, cfg):
    features = torch.squeeze(F.sigmoid(features), dim=1)
    targets = (targets == cfg['ova_target']).float()
    loss = F.binary_cross_entropy(features, targets)
    
    return {
        'loss': loss,
    }


def share_ovnni_loss(logits, targets, cfg):
    (ava_logits, ova_logits) = logits
    loss = 0
    if cfg['model'].cfg['head_training'] is False:
        loss += F.cross_entropy(ava_logits[:len(targets)], targets)
    
    for i in range(ova_logits.size(1)):
        loss += F.binary_cross_entropy(torch.sigmoid(ova_logits[:len(targets), i]), (targets == i).float())
        
    return {
        'loss': loss 
    }

def adversarial_learning_outlier_exposure(logits, targets, cfg):
    in_len = len(targets)
    in_logits = logits[:in_len]
    out_logits = logits[in_len:]
    in_loss = F.cross_entropy(in_logits, targets)
    out_loss = -(out_logits.mean(1) - torch.logsumexp(out_logits, dim=1)).mean()
    loss = cfg['beta1'] * in_loss + cfg['beta3'] * out_loss
    
    return {
        'loss' : loss,
        'in_loss': in_loss,
        'out_loss': out_loss,
    }

# def contrastive_loss(logits, targets, cfg):
#     (g_logits, h_logits) = logits
#     sup_loss = F.cross_entropy(g_logits[:len(targets)], targets)
    
#     ## Constrastive loss
#     con_loss = 0
#     exp_sim = torch.exp(F.cosine_similarity(h_logits.unsqueeze(2), h_logits.t().unsqueeze(0))) / cfg['temperature']
    
#     bs = len(targets)
#     for i in range(h_logits.size(0)):
#         if i < bs:
#             con_loss += -torch.log(exp_sim[i, i + bs] / (exp_sim[i, :].sum() - exp_sim[i, i]))
#         else:
#             con_loss += -torch.log(exp_sim[i, i - bs] / (exp_sim[i, :].sum() - exp_sim[i, i]))
    
#     loss = con_loss + cfg['lamda'] * sup_loss
    
#     return {
#         'loss': loss,
#         'sup_loss': sup_loss,
#         'con_loss': con_loss,
#     }

def contrastive_loss(logits, targets, cfg):
    (g_logits, h_logits) = logits
    bs = len(targets)
    K = g_logits.size(1)
    
    sup_loss = 0
    if cfg['sup_loss']:
        log_likelihood = - F.log_softmax(g_logits[:len(targets)], dim=1)
        one_hot = torch.zeros_like(log_likelihood)
        one_hot.scatter_(1, targets.unsqueeze(0).t(), 1)
        one_hot = one_hot * (1-cfg['alpha']) + cfg['alpha'] / K
        sup_loss = torch.sum(torch.mul(log_likelihood, one_hot), dim=1).mean()
    
    ## Constrastive loss
    con_loss = 0
    exp_sim = torch.exp(F.cosine_similarity(h_logits.unsqueeze(2), h_logits.t().unsqueeze(0))) / cfg['temperature']
    
    for i in range(h_logits.size(0)):
        if i < bs:
            con_loss += -torch.log(exp_sim[i, i + bs] / (exp_sim[i, :].sum() - exp_sim[i, i + bs] - exp_sim[i, i]))
        else:
            con_loss += -torch.log(exp_sim[i, i - bs] / (exp_sim[i, :].sum() - exp_sim[i, i - bs] - exp_sim[i, i]))
    
    loss = con_loss + cfg['lamda'] * sup_loss
    
    return {
        'loss': loss,
        'sup_loss': sup_loss,
        'con_loss': con_loss,
    }

def ovadm_loss(logits, targets, cfg):
    loss = 0
    for i in range(logits.size(1)):
        loss += F.binary_cross_entropy(2 * torch.sigmoid(logits[:len(targets), i]), (targets == i).float())
    return {
        'loss': loss,
    }   
    


# Add new loss here!!!

_LOSSES = {
                "cross_entropy_in_distribution": cross_entropy_in_distribution,
                "dont_care": dont_care,
                "oe": outlier_exposure,
                "oecc": outlier_exposure_confidence_control,
                "lc" : learning_confidence_loss,
                #"isomax" : GenericLossSecondPart,
                "ova_bce": ova_bce_loss,
                "sovnni": share_ovnni_loss,
                "aloe": adversarial_learning_outlier_exposure,
                "isomax_push": cross_entropy_with_push_cluster,
                "contrastive": contrastive_loss,
                "ovadm": ovadm_loss,
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


