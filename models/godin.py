import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
#sys.path.append('./')
from models.basic_blocks import BasicBlock, NetworkBlock


class H_I(nn.Module):
    def __init__(self, in_features, out_features):
        super(H_I, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.uniform_(self.weights, a=-math.sqrt(3/self.in_features),
                         b=math.sqrt(3/self.in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        nn.init.uniform_(self.bias, a=-math.sqrt(3/self.in_features),
                         b=math.sqrt(3/self.in_features))
        
        print("Use H_I projection")
        
    def forward(self, features):
        return features.matmul(self.weights.t()) + self.bias
    

class H_C(nn.Module):
    def __init__(self, in_features, out_features):
        super(H_C, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.uniform_(self.weights, a=-math.sqrt(3/self.in_features),
                         b=math.sqrt(3/self.in_features))
        
        print("Use H_C projection")
        
    def forward(self, features):
        fnorm = torch.norm(features, p=2, dim=1).unsqueeze(1)
        wnorm = torch.norm(self.weights, p=2, dim=1).unsqueeze(0)
        D = wnorm * fnorm
        features = features.matmul(self.weights.t())
        return features / D
    
    
class G_I(nn.Module):
    def __init__(self, in_features, out_features):
        super(G_I, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.uniform_(self.weights, a=-math.sqrt(3/self.in_features),
                         b=math.sqrt(3/self.in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        nn.init.uniform_(self.bias, a=-math.sqrt(3/self.in_features),
                         b=math.sqrt(3/self.in_features))
        
        self.bn = nn.BatchNorm1d(out_features)
        
        print("Use G_I projection")
        
    def forward(self, features):
        features = features.matmul(self.weights.t()) + self.bias
        features = self.bn(features)
        return torch.sigmoid(features)
        
design_choice = {
    "hi" : H_I,
    #"he" : H_E,
    "hc" : H_C,
    "gi" : G_I,
}

class WideResNetGODIN32(nn.Module):
    def __init__(self, cfg):
        super(WideResNetGODIN32, self).__init__()
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
        self.H = design_choice[cfg['H']](nChannels[3], num_classes)
        self.G = design_choice[cfg['G']](nChannels[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        h = self.H(out)
        g = self.G(out)
        f = h / g
        return f, h

    
    
class WideResNetGODIN224(nn.Module):
    def __init__(self, cfg):
        super(WideResNetGODIN224, self).__init__()
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
        
        self.nChannels = nChannels[3]
        self.H = design_choice[cfg['H']](nChannels[3], num_classes)
        self.G = design_choice[cfg['G']](nChannels[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 14)
        out = out.view(-1, self.nChannels)
        h = self.H(out)
        g = self.G(out)
        f = h / g
        return f, h
    
    