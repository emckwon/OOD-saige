"""
OOD detectors
Given inlier and outlier's logits return confidence score of each sample (0 ~ 1)

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Dummy detector
def dummy_overconfident_detector(logits, targets):
    return torch.ones(logtis.size(0), 1)

# Dummy detector
def dummy_selfless_detector(logits, targets):
    return torch.zeros(logtis.size(0), 1)

# maximum softmax probablity (MSP) detector
def msp_detector(logits, targets):    
    return torch.max(logits, dim=1, keepdim=True).values

# Add new detector here!!!

_DETECTORS = {
                "dummy_overconfident": dummy_overconfident_detector,
                "dummy_selfless": dummy_selfless_detector,
                "msp": msp_detector,
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


