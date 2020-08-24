import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.spectral import SpectralNorm
import numpy as np

# class CondNorm2d(nn.Module):
#     def __init__(self, feature_dim, channel_dim):
#         super(CondNorm2d, self).__init__()
#         self.feature_dim = feature_dim
#         self.channel_dim = channel_dim
#         self.fc_gamma = nn.Linear(feature_dim, channel_dim)
#         self.fc_alpha = nn.Linear(feature_dim, channel_dim)
        
#     def forward(self, x, feature):
        
#         gamma = self.fc_gamma(feature)
#         alpha = self.fc_alpha(feature)
        
#         c_mean = x.transpose(0,1).contiguous().view(x.size(1),-1).mean(1)
#         c_std = torch.sqrt(x.transpose(0,1).contiguous().view(x.size(1),-1).var(1) + 1e-05)
        
#         out = ((x - c_mean.view(1, -1, 1, 1)) / c_std.view(1, -1, 1, 1)) * gamma.view(gamma.size(0), -1 , 1, 1) + alpha.view(alpha.size(0), -1, 1, 1)
#         return out
class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = linear

    def forward(self, input):
        return self.linear(input)


class CondNorm2d(nn.Module):
    def __init__(self, style_dim, in_channel):
        super(CondNorm2d, self).__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out
        

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

class Generator(nn.Module):
    """Generator."""

    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64, feat_dim=512):
        super(Generator, self).__init__()
        self.imsize = image_size
        self.z_dim = z_dim
        layer1 = []
        layer2 = []
        layer3 = []
        last = []
        
        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num # 8
        
        self.l1 = SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4))
        #layer1.append(nn.BatchNorm2d(conv_d
        self.cn1 = CondNorm2d(feat_dim, conv_dim * mult)

        curr_dim = conv_dim * mult

        self.l2 = SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1))
        #layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        self.cn2 = CondNorm2d(feat_dim, int(curr_dim / 2))

        curr_dim = int(curr_dim / 2)

        self.l3 = SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1))
        #layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        self.cn3 = CondNorm2d(feat_dim, int(curr_dim / 2))
        
        curr_dim = int(curr_dim / 2)
        
        self.l4 = SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1))
        #layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
        self.cn4 = CondNorm2d(feat_dim, int(curr_dim / 2))
        
        curr_dim = int(curr_dim / 2)
        
        self.l5 = SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1))
        #layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
        self.cn5 = CondNorm2d(feat_dim, int(curr_dim / 2))
        
        curr_dim = int(curr_dim / 2)
        
        self.l6 = SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1))
        #layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
        self.cn6 = CondNorm2d(feat_dim, int(curr_dim / 2))
        
        curr_dim = int(curr_dim / 2)
        
        self.last = nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1)
        self.last_cn = CondNorm2d(feat_dim, 3)

        self.attn1 = Self_Attn( 256, 'relu')
        self.attn2 = Self_Attn( 128,  'relu')

    def forward(self, z, feature):
        z = z.view(z.size(0), z.size(1), 1, 1)
        #print("z: {}".format(z.size()))
        out = self.l1(z)
        #print("l1: {}".format(out.size()))
        out = F.relu(self.cn1(out, feature))
        out = self.l2(out)
        #print("l2: {}".format(out.size()))
        out = F.relu(self.cn2(out, feature))
        out = self.l3(out)
        #print("l3: {}".format(out.size()))
        out = F.relu(self.cn3(out, feature))
        out = self.l4(out)
        #print("l4: {}".format(out.size()))
        out = F.relu(self.cn4(out, feature))
        out,p1 = self.attn1(out)
        out = self.l5(out)
        #print("l5: {}".format(out.size()))
        out = F.relu(self.cn5(out, feature))
        out,p2 = self.attn2(out)
        out = self.l6(out)
        #print("l6: {}".format(out.size()))
        out = F.relu(self.cn6(out, feature))
        
        out = self.last(out)
        #print("last: {}".format(out.size()))
        out = F.tanh(self.last_cn(out, feature))

        return out, p1, p2


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64, feat_dim=512):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 256:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2,4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')
        
        self.cn = CondNorm2d(feat_dim, 512)

    def forward(self, x, feature):
        out = self.l1(x)
        #print("l1: {}".format(out.size()))
        out = self.l2(out)
        #print("l2: {}".format(out.size()))
        out = self.l3(out)
        #print("l3: {}".format(out.size()))
        out,p1 = self.attn1(out)
        #print("attn1: {}".format(out.size()))
        out=self.l4(out)
        #print("l4: {}".format(out.size()))
        out,p2 = self.attn2(out)
        #print("attn2: {}".format(out.size()))
        out = self.cn(out, feature)
        #print("cn: {}".format(out.size()))
        out = self.last(out)
        #print("last: {}".format(out.size()))

        return out.squeeze(), p1, p2
    
    
class Generator32(nn.Module):
    """Generator. 32by32"""

    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64, feat_dim=512):
        super(Generator32, self).__init__()
        self.imsize = image_size
        self.z_dim = z_dim
        layer1 = []
        layer2 = []
        layer3 = []
        last = []
        
        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num # 8
        
        self.l1 = SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4))
        #layer1.append(nn.BatchNorm2d(conv_d
        self.cn1 = CondNorm2d(feat_dim, conv_dim * mult)

        curr_dim = conv_dim * mult

        self.l2 = SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1))
        #layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        self.cn2 = CondNorm2d(feat_dim, int(curr_dim / 2))

        curr_dim = int(curr_dim / 2)

        self.l3 = SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1))
        #layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        self.cn3 = CondNorm2d(feat_dim, int(curr_dim / 2))
        
        curr_dim = int(curr_dim / 2)
        
        self.last = nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1)
        self.last_cn = CondNorm2d(feat_dim, 3)

        self.attn1 = Self_Attn( 128, 'relu')
        self.attn2 = Self_Attn( 64,  'relu')

    def forward(self, z, feature):
        z = z.view(z.size(0), z.size(1), 1, 1)
        #print("z: {}".format(z.size()))
        out = self.l1(z)
        #print("l1: {}".format(out.size()))
        out = F.relu(self.cn1(out, feature))
        out = self.l2(out)
        #print("l2: {}".format(out.size()))
        out = F.relu(self.cn2(out, feature))
        out,p1 = self.attn1(out)
        out = self.l3(out)
        #print("l3: {}".format(out.size()))
        out = F.relu(self.cn3(out, feature))
        out,p2 = self.attn2(out)
        out = self.last(out)
        #print("last: {}".format(out.size()))
        out = F.tanh(self.last_cn(out, feature))

        return out, p1, p2


class Discriminator32(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64, feat_dim=512):
        super(Discriminator32, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(128, 'relu')
        self.attn2 = Self_Attn(256, 'relu')
        
        self.cn = CondNorm2d(feat_dim, 256)

    def forward(self, x, feature):
        out = self.l1(x)
        out = self.l2(out)
        out,p1 = self.attn1(out)
        out = self.l3(out)
        out,p2 = self.attn2(out)
        out = self.cn(out, feature)
        out=self.last(out)

        return out.squeeze(), p1, p2