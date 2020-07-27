"""
OOD detectors
Given inlier and outlier's logits return confidence score of each sample (0 ~ 1)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Dummy detector
def dummy_overconfident_detector(logits, targets, cfg):
    confidences = torch.ones(logits.size(0), 1) /2
    return {
        'confidences': confidences,
        'inlier_mean': confidences[:len(targets)].data.cpu().mean(),
        'outlier_mean': confidences[len(targets):].data.cpu().mean()
    }

# Dummy detector
def dummy_selfless_detector(logits, targets, cfg):
    confidences = torch.zeros(logits.size(0), 1)
    return {
        'confidences': confidences,
        'inlier_mean': confidences[:len(targets)].data.cpu().mean(),
        'outlier_mean': confidences[len(targets):].data.cpu().mean()
    }

# maximum softmax probablity (MSP) detector
def msp_detector(logits, targets, cfg): 
    confidences =  torch.max(F.softmax(logits, dim=1), dim=1, keepdim=True).values
    return {
        'confidences': confidences,
        'inlier_mean': confidences[:len(targets)].data.cpu().mean(),
        'outlier_mean': confidences[len(targets):].data.cpu().mean()
    }

# Negative entropy score (Entropic score)
def negative_entropy(logits, targets, cfg):
    confidences = (F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(dim=1)
    return {
        'confidences': confidences,
        'inlier_mean': confidences[:len(targets)].data.cpu().mean(),
        'outlier_mean': confidences[len(targets):].data.cpu().mean()
    }

def share_ovnni_detector(logits, targets, cfg):
    K = logits.size(1) // 2
    ava_logits = F.softmax(logits[:, :K], dim=1)
    confidences = torch.zeros_like(ava_logits)
    for i in enumerate(range(K)):
        ova_logit = F.relu(logits[:, K + i])
        confidences[:, i] = ova_logit[:,0] * ava_logits[:, i]
        print(confidences[:, i].size())
    
    confidences = F.softmax(confidences, dim=1)
    return {
        'confidences': confidences,
        'inlier_mean': confidences[:len(targets)].data.cpu().mean(),
        'outlier_mean': confidences[len(targets):].data.cpu().mean()
    }


# ODIN detector
def ODIN(logits, targets, cfg): 
    data = cfg['data']
    model = cfg['model']
    images = Variable(data, requires_grad=True).cuda()
    images.retain_grad()
    
    T = cfg['temperature']
    epsilon = cfg['epsilon']

    model.zero_grad()
    pred = model(images)
    _, pred_idx = torch.max(pred.data, 1)
    labels = Variable(pred_idx)
    pred = pred / T
    loss = F.cross_entropy(pred[:len(targets)], targets)
    loss.backward()

    images = images - epsilon * torch.sign(images.grad)
    images = Variable(images.data, requires_grad=True)

    pred = model(images)

    pred = pred / T
    pred = F.softmax(pred, dim=-1)
    pred = torch.max(pred.data, 1)[0]
#     pred = pred.cpu().numpy()
    
    return {
        'confidences': pred,
        'inlier_mean': pred[:len(targets)].data.cpu().mean(),
        'outlier_mean': pred[len(targets):].data.cpu().mean()
    }


def LC_detector(logits, targets, cfg):
    data = cfg['data']
    model = cfg['model']
    images = Variable(data, requires_grad=True).cuda()
    images.retain_grad()

    epsilon = cfg['epsilon']

    model.zero_grad()
    _, confidence = model(images)
    confidence = F.sigmoid(confidence).view(-1)
    loss = torch.mean(-torch.log(confidence))
    loss.backward()

    images = images - args.epsilon * torch.sign(images.grad)
    images = Variable(images.data, requires_grad=True)

    _, confidence = model(images)
    confidence = F.sigmoid(confidence)
#     confidence = confidence.data.cpu().numpy()
#     out.append(confidence)
    return {
        'confidences': pred
    }
    


# Add new detector here!!!

_DETECTORS = {
                "dummy_overconfident": dummy_overconfident_detector,
                "dummy_selfless": dummy_selfless_detector,
                "msp": msp_detector,
                "ne": negative_entropy,
                "odin": ODIN,
                "lc": LC_detector,
                "sovnni": share_ovnni_detector,
               }

def getDetector(cfg):
    """
    Retrieve the detector given the detector's name.
    Args (int):
        detector_name: the name of the detector to use.
    """
    detector_name = cfg['detector']
    if detector_name not in _DETECTORS.keys():
        raise NotImplementedError("Detector {} is not supported".format(detector_name))
    return _DETECTORS[detector_name]


