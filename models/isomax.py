import torch.nn as nn
import torch
import math
import utils.procedures as utils
from models.basic_blocks import BasicBlock, NetworkBlock
import torch.nn.functional as F

class GenericLossFirstPart(nn.Module):
    """Replaces classifier layer"""
    def __init__(self, in_features, out_features):
        super(GenericLossFirstPart, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.metrics_evaluation_mode = False
        nn.init.constant_(self.weights, 0.0)
        #print("\nPROTOTYPES INITIALIZED:\n", self.weights, "\n")
        print("PROTOTYPES INITIALIZED [MEAN]:\n", self.weights.mean(dim=0).mean(), "\n")
        print("PROTOTYPES INITIALIZED [STD]:\n", self.weights.std(dim=0).mean(), "\n")

    def forward(self, features):
        #if self.training or self.metrics_evaluation_mode:
            return features
        #else:
         #   distances = utils.euclidean_distances(features, self.weights, 2)
         #   return -distances


class WideResNetIsoMax224(nn.Module):
    def __init__(self, cfg):
        super(WideResNetIsoMax224, self).__init__()
        depth = cfg['depth']
        num_classes = cfg['num_classes']
        widen_factor = cfg['widen_factor']
        dropRate = cfg['drop_rate']
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
        self.classifier = GenericLossFirstPart(self.nChannels, num_classes)
        
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
        self.classifier = GenericLossFirstPart(self.nChannels, num_classes)
        
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