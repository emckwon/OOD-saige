import torch
import torch.nn as nn
from models.wrn import WideResNet, WideResNet224, WideResNet256, WideResNetPen, WideResNetPen224
from models.isomax import WideResNetIsoMax224, WideResNetIsoMax32
from models.godin import WideResNetGODIN32, WideResNetGODIN224 
import models.md_resnet as resnet
from models.ovnni import OVNNI

_MODEL_TYPES = {
    "wrn": WideResNet,
    "wrn224": WideResNet224,
    "wrn256": WideResNet256,
    "wrnisomax224": WideResNetIsoMax224,
    "wrnisomax32": WideResNetIsoMax32,
    "resnet34": resnet.ResNet34,
    "ovnni": OVNNI,
    "wrngodin32": WideResNetGODIN32,
    "wrngodin224": WideResNetGODIN224,
    "wrnpen32": WideResNetPen,
    "wrnpen224": WideResNetPen224,
    
}

def getModel(m_cfg):
    
    assert (m_cfg['network_kind'] in _MODEL_TYPES.keys()), "Model type '{}' not supported.".format(m_cfg['network_kind'])
    if m_cfg['network_kind'] in ['resnet34',]:
        model = _MODEL_TYPES[m_cfg['network_kind']](m_cfg['num_classes'])
    else:
        model = _MODEL_TYPES[m_cfg['network_kind']](m_cfg)
    model.cuda()
      
    return model