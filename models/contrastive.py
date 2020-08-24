import torch.nn as nn
import torch
import torchvision
import math
import utils.procedures as utils
from models.basic_blocks import BasicBlock, NetworkBlock
from models.pretrained_model import pretrained_model
import torch.nn.functional as F

    
    
class PTH_Contrastive224(nn.Module):
    def __init__(self, cfg):
        super(PTH_Contrastive224, self).__init__()
        self.freeze = cfg['freeze']
        pmodel = pretrained_model[cfg['pretrained']](pretrained=True if 'pretrained_weights' not in cfg.keys() else cfg['pretrained_weights'],
                                                     progress=True)
        num_classes = cfg['num_classes']
        self.nChannels = pmodel.fc.in_features
        self.network = nn.Sequential(*list(pmodel.children())[:-1])
        self.classifier = nn.Linear(self.nChannels, num_classes)
        self.projector1 = nn.Linear(self.nChannels, self.nChannels)
        self.projector2 = nn.Linear(self.nChannels, self.nChannels)
        
    def forward(self, x, epoch=None):
        if epoch is not None and epoch < self.freeze:
            with torch.no_grad():
                out = self.network(x)
        else:
            out = self.network(x)
        out = out.view(-1, self.nChannels)
        h_logits = F.relu(self.projector1(out))
        return (self.classifier(out), self.projector2(h_logits), out)
    
    
class PTH_Contrastive224_Custom(nn.Module):
    def __init__(self, cfg):
        super(PTH_Contrastive224_Custom, self).__init__()
        self.freeze = cfg['freeze']
        pmodel = pretrained_model[cfg['pretrained']](pretrained=True if 'pretrained_weights' not in cfg.keys() else cfg['pretrained_weights'],
                                                     progress=True)
        num_classes = cfg['num_classes']
        alpha = cfg['alpha']
        self.nChannels = pmodel.fc.in_features
        self.network = nn.Sequential(*list(pmodel.children())[:-1])
        self.classifier = GenericLossFirstPart(self.nChannels*2, num_classes, alpha)
        self.linear1 = nn.Linear(self.nChannels, self.nChannels * 2)
        self.bn1 = nn.BatchNorm1d(self.nChannels * 2)
        self.linear2 = nn.Linear(self.nChannels * 2, self.nChannels * 2)
        self.bn2 = nn.BatchNorm1d(self.nChannels * 2)
        
    def forward(self, x, epoch=None):
        if epoch is not None and epoch < self.freeze:
            with torch.no_grad():
                out = self.network(x)
        else:
            out = self.network(x)
            # [Bs, 512]
        out = out.view(-1, self.nChannels)
        out = F.relu(self.bn1(self.linear1(out)))
        out = F.relu(self.bn2(self.linear2(out)))
        return self.classifier(out)

