import torch
import torch.nn as nn
from models.wrn import WideResNet, WideResNet224

_MODEL_TYPES = {
    "wrn": WideResNet,
    "wrn224": WideResNet224,
}

def getModel(m_cfg):
    
    assert (m_cfg['network_kind'] in _MODEL_TYPES.keys()), "Model type '{}' not supported.".format(m_cfg['network_kind'])

    model = _MODEL_TYPES[m_cfg['network_kind']](m_cfg)
    model.cuda()
      
    return model