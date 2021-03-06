import torch.nn as nn
import torch
import torchvision
import math
import utils.procedures as utils
from models.basic_blocks import BasicBlock, NetworkBlock
from models.pretrained_model import pretrained_model
import torch.nn.functional as F

class GenericLossFirstPart(nn.Module):
    """Replaces classifier layer"""
    def __init__(self, in_features, out_features, alpha):
        super(GenericLossFirstPart, self).__init__()
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.constant_(self.weights, 0.0)
        #print("\nPROTOTYPES INITIALIZED:\n", self.weights, "\n")
        print("PROTOTYPES INITIALIZED [MEAN]:\n", self.weights.mean(dim=0).mean(), "\n")
        print("PROTOTYPES INITIALIZED [STD]:\n", self.weights.std(dim=0).mean(), "\n")

    def forward(self, features):
        distances = utils.euclidean_distances(features, self.weights, 2)
        return (-self.alpha * distances, features)
    

class GenericLossFirstPart_Sim(nn.Module):
    """Replaces classifier layer"""
    def __init__(self, in_features, out_features, alpha):
        super(GenericLossFirstPart_Sim, self).__init__()
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.normal_(self.weights)
        print("\nPROTOTYPES INITIALIZED:\n", self.weights, "\n")
        print("PROTOTYPES INITIALIZED [MEAN]:\n", self.weights.mean(dim=0).mean(), "\n")
        print("PROTOTYPES INITIALIZED [STD]:\n", self.weights.std(dim=0).mean(), "\n")

    def forward(self, features):
        distances = F.cosine_similarity(features.unsqueeze(2), self.weights.t().unsqueeze(0))
        return -self.alpha * distances
    
    
class PTH_IsoMax224(nn.Module):
    def __init__(self, cfg):
        super(PTH_IsoMax224, self).__init__()
        self.freeze = cfg['freeze']
        pmodel = pretrained_model[cfg['pretrained']](pretrained=True if 'pretrained_weights' not in cfg.keys() else cfg['pretrained_weights'],
                                                     progress=True)
        num_classes = cfg['num_classes']
        alpha = cfg['alpha']
        self.nChannels = pmodel.fc.in_features
        self.network = nn.Sequential(*list(pmodel.children())[:-1])
        self.classifier = GenericLossFirstPart(self.nChannels, num_classes, alpha)
        
    def forward(self, x, epoch=None):
        if epoch is not None and epoch < self.freeze:
            with torch.no_grad():
                out = self.network(x)
        else:
            out = self.network(x)
            # [Bs, 512]
        out = out.view(-1, self.nChannels)
        return self.classifier(out)
    
    
class PTH_IsoMax224_Custom(nn.Module):
    def __init__(self, cfg):
        super(PTH_IsoMax224_Custom, self).__init__()
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

    

class WideResNetIsoMax224(nn.Module):
    def __init__(self, cfg):
        super(WideResNetIsoMax224, self).__init__()
        depth = cfg['depth']
        num_classes = cfg['num_classes']
        widen_factor = cfg['widen_factor']
        dropRate = cfg['drop_rate']
        alpha = cfg['alpha'] # When validation alpha should be 1
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=2,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 2, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        #self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.classifier = GenericLossFirstPart(self.nChannels, num_classes, alpha)
        
        ################ 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)
            #elif isinstance(m, Bottleneck):
            #    nn.init.constant_(m.bn3.weight, 0)
        ################

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 14)
        out = out.view(-1, self.nChannels)
        return self.classifier(out)
    
    
class WideResNetIsoMax32(nn.Module):
    def __init__(self, cfg):
        super(WideResNetIsoMax32, self).__init__()
        depth = cfg['depth']
        num_classes = cfg['num_classes']
        widen_factor = cfg['widen_factor']
        dropRate = cfg['drop_rate']
        alpha = cfg['alpha'] # When validation alpha should be 1
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        #self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.classifier = GenericLossFirstPart(self.nChannels, num_classes, alpha)
        
        ################ 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)
            #elif isinstance(m, Bottleneck):
            #    nn.init.constant_(m.bn3.weight, 0)
        ################

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.classifier(out)