import torch
import torch.nn as nn
import torch.nn.functional as F
from models.wrn import WideResNet224

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
            
            
            
            
        
            
            
        
        
        
        

