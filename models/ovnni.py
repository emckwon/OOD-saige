import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from models.wrn import WideResNet224, WideResNetFeat224
from models.basic_blocks import BasicBlock, NetworkBlock

class OVNNI(nn.Module):
    def __init__(self, cfg):
        super(OVNNI, self).__init__()
        self.ava_cfg = cfg['AVA']
        self.ova_cfg = cfg['OVA']
        num_classes = self.ava_cfg['num_classes']
        self.ava_network = WideResNet224(self.ava_cfg)
        self.ova_networks = nn.ModuleList([WideResNet224(self.ova_cfg) for i in range(num_classes)])
        
        
    def forward(self, x):
        ava_logits = F.softmax(self.ava_network(x), dim=1)
        out = torch.zeros_like(ava_logits).cuda()
        for idx, ova_network in enumerate(self.ova_networks):
            ova_logit = F.relu(ova_network(x))

            out[:, idx] = ova_logit[:,0] * ava_logits[:, idx]
            
        return out
    

class OVAHead(nn.Module):
    def __init__(self, nChannels, dropRate):
        super(OVAHead, self).__init__()
        self.nChannels = nChannels
        self.b1 = BasicBlock(nChannels, nChannels, stride=1, dropRate=dropRate)
        self.b2 = BasicBlock(nChannels, nChannels, stride=1, dropRate=dropRate)
        self.b3 = BasicBlock(nChannels, nChannels, stride=1, dropRate=dropRate)
        self.b4 = BasicBlock(nChannels, nChannels, stride=1, dropRate=dropRate)
        self.fc = nn.Linear(nChannels, 1)
        
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = F.avg_pool2d(x, 14)
        x = x.view(-1, self.nChannels)
        return self.fc(x)

class SOVNNI(nn.Module):
    def __init__(self, cfg):
        super(SOVNNI, self).__init__()
        self.cfg = cfg
        if self.cfg['head_training']:
            print("Train OVA head only")
        num_classes = cfg['num_classes']
        self.network = WideResNetFeat224(cfg)
        nChannels = self.network.nChannels
        self.ova_heads = nn.ModuleList([OVAHead(nChannels, dropRate=cfg['drop_rate']) for i in range(num_classes)])
        
        
    def forward(self, x):
        if self.cfg['head_training']:
            with torch.no_grad():
                ava_logits, pens = self.network(x)
        else:
            ava_logits, pens = self.network(x) 
            
        ova_logits = []
        for idx, ova_head in enumerate(self.ova_heads):
            ova_logit = ova_head(pens)
            ova_logits.append(ova_logit)
            
        ova_logits = torch.cat(ova_logits, 1)
        
        #ava_logits: [Bs, num_class]
        #ova_logits: [Bs, num_class]
        
        return (ava_logits, ova_logits)
    
    
    
class ChannelWiseSOVNNI224(nn.Module):
    def __init__(self, cfg):
        super(ChannelWiseSOVNNI224, self).__init__()
        self.cfg = cfg
        depth = cfg['depth']
        num_classes = cfg['num_classes']
        #widen_factor = cfg['widen_factor']
        widen_factor = num_classes * 4
        print("Use widen factor as {}".format(widen_factor))
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
        self.num_classes = num_classes
        self.fc = nn.Linear(128 * num_classes, num_classes)
        self.ova_heads = nn.ModuleList([nn.Linear(128, 1) for i in range(num_classes)])
        for ova in self.ova_heads:
            ova.bias.data.zero_()
    
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
        ava_logits = self.fc(out[:, :128 * self.num_classes])
        
        ova_logits = []
        for idx, ova_head in enumerate(self.ova_heads):
            ova_logit = ova_head(out[:, 128 * (self.num_classes + idx): 128 * (self.num_classes + idx + 1)])
            ova_logits.append(ova_logit)
            
        ova_logits = torch.cat(ova_logits, 1)
        
        return (ava_logits, ova_logits)
            
            
            
            
            
        
            
            
        
        
        
        

