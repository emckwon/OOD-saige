import torch
import torch.nn as nn
from models.wrn import WideResNet, WideResNet224, WideResNet256, WideResNetPen, WideResNetPen224, WideResNetFeat224
from models.isomax import WideResNetIsoMax224, WideResNetIsoMax32, PTH_IsoMax224
from models.godin import WideResNetGODIN32, WideResNetGODIN224, PTH_GODIN224
import models.md_resnet as resnet
from models.ovnni import OVNNI, SOVNNI, ChannelWiseSOVNNI224
from models.pretrained_model import pretrained_model


class PTH_model(nn.Module):
    def __init__(self, cfg):
        super(PTH_model, self).__init__()
        self.freeze = cfg['freeze']
        pmodel = pretrained_model[cfg['pretrained']](pretrained=True, progress=True)
        num_classes = cfg['num_classes']
        self.nChannels = pmodel.fc.in_features
        self.network = nn.Sequential(*list(pmodel.children())[:-1])
        self.fc = nn.Linear(self.nChannels, num_classes)
        
    def forward(self, x, epoch=None):
        if epoch is not None and epoch < self.freeze:
            with torch.no_grad():
                out = self.network(x)
        else:
            out = self.network(x)
            # [Bs, 512]
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out

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
    "wrnfeat224": WideResNetFeat224,
    "cwsovnni": ChannelWiseSOVNNI224,
    "sovnni": SOVNNI,
    "pgodin": PTH_GODIN224,
    "pmodel": PTH_model,
    "pisomax": PTH_IsoMax224,
    
}

def getModel(m_cfg):
    
    assert (m_cfg['network_kind'] in _MODEL_TYPES.keys()), "Model type '{}' not supported.".format(m_cfg['network_kind'])
    if m_cfg['network_kind'] in ['resnet34',]:
        model = _MODEL_TYPES[m_cfg['network_kind']](m_cfg['num_classes'])
    else:
        model = _MODEL_TYPES[m_cfg['network_kind']](m_cfg)
    model.cuda()
      
    return model