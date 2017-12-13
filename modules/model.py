import os
import scipy.io
import numpy as np
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.utils.model_zoo as model_zoo 
import torchvision.models as models # will likely need to install torchvision manually
import math


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def append_params(params, module, prefix):
    for child in module.children():
        for k,p in child._parameters.items():
            if p is None: continue
            
            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k
            
            if name not in params:
                params[name] = p
            else:
                raise RuntimeError("Duplicated param name: %s" % (name))


class LRN(nn.Module):
    def __init__(self):
        super(LRN, self).__init__()

    def forward(self, x):
        #
        # x: N x C x H x W
        pad = Variable(x.data.new(x.size(0), 1, 1, x.size(2), x.size(3)).zero_())
        x_sq = (x**2).unsqueeze(dim=1)
        x_tile = torch.cat((torch.cat((x_sq,pad,pad,pad,pad),2),
                            torch.cat((pad,x_sq,pad,pad,pad),2),
                            torch.cat((pad,pad,x_sq,pad,pad),2),
                            torch.cat((pad,pad,pad,x_sq,pad),2),
                            torch.cat((pad,pad,pad,pad,x_sq),2)),1)
        x_sumsq = x_tile.sum(dim=1).squeeze(dim=1)[:,2:-2,:,:]
        x = x / ((2.+0.0001*x_sumsq)**0.75)
        return x


# based on ResNet50 model, need to figure out what is wrong with the output dims
class MDNet(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(MDNet, self).__init__()
        self.K = K
        self.inplanes = 64
        self.layers = nn.Sequential(OrderedDict([
                    ('conv1', nn.Sequential(nn.Conv2d(3, 64, kernel_size=1, stride=2, padding=1, bias=False))),
                    ('bn1', nn.Sequential(nn.BatchNorm2d(64))),
                    ('relu', nn.Sequential(nn.ReLU(inplace=True))),
                    ('maxpool', nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))),
                    ('layer1', self._make_layer(Bottleneck, 64, 3)),
                    ('layer2', self._make_layer(Bottleneck, 128, 4, stride=2)),
                    ('layer3', self._make_layer(Bottleneck, 256, 6, stride=2)),
                    ('layer4', self._make_layer(Bottleneck, 512, 3, stride=2)),
                    ('avgpool', nn.Sequential(nn.AvgPool2d(4, stride=1))),
                    ('fc', nn.Sequential(nn.Linear(512 * BasicBlock.expansion, 1000)))            
        ]))
        '''
        self.layers = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(),
                                        LRN(),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                        nn.ReLU(),
                                        LRN(),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                        nn.ReLU())),
                ('fc4',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512 * 3 * 3, 512),
                                        nn.ReLU())),
                ('fc5',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512, 512),
                                        nn.ReLU()))]))
        '''
        
        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5), 
                                                     nn.Linear(512, 2)) for _ in range(K)])
        
        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError("Unkown model format: %s" % (model_path))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_%d'%(k))

    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False
 
    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params
    
    def forward(self, x, k=0, in_layer='conv1', out_layer='fc'):
        #
        # forward model from in_layer to out_layer

        run = False
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                print(name)
                x = module(x)
                if name == 'avgpool':
                    x = x.view(x.size(0),-1)
                if name == out_layer:
                    return x
    
    def load_model(self, model_path):
        states = torch.load(model_path)
        items = list(states.items())

        self.layers[0][0].weight.data = items[0][1] #conv1
        self.layers[1][0].weight.data = items[1][1] #bn1 weight
        self.layers[1][0].bias.data = items[2][1] #bn1 bias
        self.layers[1][0].running_mean.data = items[3][1]
        self.layers[1][0].running_var.data = items[4][1]

        self.layers[4][0].conv1.weight.data = items[5][1] # layer 1.0
        self.layers[4][0].bn1.weight.data = items[6][1]
        self.layers[4][0].bn1.bias.data = items[7][1]
        self.layers[4][0].bn1.running_mean.data = items[8][1]
        self.layers[4][0].bn1.running_var.data = items[9][1]
        self.layers[4][0].conv2.weight.data = items[10][1]
        self.layers[4][0].bn2.weight.data = items[11][1]
        self.layers[4][0].bn2.bias.data = items[12][1]
        self.layers[4][0].bn2.running_mean.data = items[13][1]
        self.layers[4][0].bn2.running_var.data = items[14][1]
        self.layers[4][0].conv3.weight.data = items[15][1]
        self.layers[4][0].bn3.weight.data = items[16][1]
        self.layers[4][0].bn3.bias.data = items[17][1]
        self.layers[4][0].bn3.running_mean.data = items[18][1]
        self.layers[4][0].bn3.running_var.data = items[19][1]

        self.layers[4][1].conv1.weight.data = items[25][1] # layer 1.1
        self.layers[4][1].bn1.weight.data = items[26][1]
        self.layers[4][1].bn1.bias.data = items[27][1]
        self.layers[4][1].bn1.running_mean.data = items[28][1]
        self.layers[4][1].bn1.running_var.data = items[29][1]
        self.layers[4][1].conv2.weight.data = items[30][1]
        self.layers[4][1].bn2.weight.data = items[31][1]
        self.layers[4][1].bn2.bias.data = items[32][1]
        self.layers[4][1].bn2.running_mean.data = items[33][1]
        self.layers[4][1].bn2.running_var.data = items[34][1]
        self.layers[4][1].conv3.weight.data = items[35][1]
        self.layers[4][1].bn3.weight.data = items[36][1]
        self.layers[4][1].bn3.bias.data = items[37][1]
        self.layers[4][1].bn3.running_mean.data = items[38][1]
        self.layers[4][1].bn3.running_var.data = items[39][1]

        self.layers[4][2].conv1.weight.data = items[40][1] # layer 1.2
        self.layers[4][2].bn1.weight.data = items[41][1]
        self.layers[4][2].bn1.bias.data = items[42][1]
        self.layers[4][2].bn1.running_mean.data = items[43][1]
        self.layers[4][2].bn1.running_var.data = items[44][1]
        self.layers[4][2].conv2.weight.data = items[45][1]
        self.layers[4][2].bn2.weight.data = items[46][1]
        self.layers[4][2].bn2.bias.data = items[47][1]
        self.layers[4][2].bn2.running_mean.data = items[48][1]
        self.layers[4][2].bn2.running_var.data = items[49][1]
        self.layers[4][2].conv3.weight.data = items[50][1]
        self.layers[4][2].bn3.weight.data = items[51][1]
        self.layers[4][2].bn3.bias.data = items[52][1]
        self.layers[4][2].bn3.running_mean.data = items[53][1]
        self.layers[4][2].bn3.running_var.data = items[54][1]
       
       #######

        self.layers[5][0].conv1.weight.data = items[55][1] # layer 2.0
        self.layers[5][0].bn1.weight.data = items[56][1]
        self.layers[5][0].bn1.bias.data = items[57][1]
        self.layers[5][0].bn1.running_mean.data = items[58][1]
        self.layers[5][0].bn1.running_var.data = items[59][1]
        self.layers[5][0].conv2.weight.data = items[60][1]
        self.layers[5][0].bn2.weight.data = items[61][1]
        self.layers[5][0].bn2.bias.data = items[62][1]
        self.layers[5][0].bn2.running_mean.data = items[63][1]
        self.layers[5][0].bn2.running_var.data = items[64][1]
        self.layers[5][0].conv3.weight.data = items[65][1]
        self.layers[5][0].bn3.weight.data = items[66][1]
        self.layers[5][0].bn3.bias.data = items[67][1]
        self.layers[5][0].bn3.running_mean.data = items[68][1]
        self.layers[5][0].bn3.running_var.data = items[69][1]

        self.layers[5][1].conv1.weight.data = items[75][1] # layer 2.1
        self.layers[5][1].bn1.weight.data = items[76][1]
        self.layers[5][1].bn1.bias.data = items[77][1]
        self.layers[5][1].bn1.running_mean.data = items[78][1]
        self.layers[5][1].bn1.running_var.data = items[79][1]
        self.layers[5][1].conv2.weight.data = items[80][1]
        self.layers[5][1].bn2.weight.data = items[81][1]
        self.layers[5][1].bn2.bias.data = items[82][1]
        self.layers[5][1].bn2.running_mean.data = items[83][1]
        self.layers[5][1].bn2.running_var.data = items[84][1]
        self.layers[5][1].conv3.weight.data = items[85][1]
        self.layers[5][1].bn3.weight.data = items[86][1]
        self.layers[5][1].bn3.bias.data = items[87][1]
        self.layers[5][1].bn3.running_mean.data = items[88][1]
        self.layers[5][1].bn3.running_var.data = items[89][1]

        self.layers[5][2].conv1.weight.data = items[90][1] # layer 2.2
        self.layers[5][2].bn1.weight.data = items[91][1]
        self.layers[5][2].bn1.bias.data = items[92][1]
        self.layers[5][2].bn1.running_mean.data = items[93][1]
        self.layers[5][2].bn1.running_var.data = items[94][1]
        self.layers[5][2].conv2.weight.data = items[95][1]
        self.layers[5][2].bn2.weight.data = items[96][1]
        self.layers[5][2].bn2.bias.data = items[97][1]
        self.layers[5][2].bn2.running_mean.data = items[98][1]
        self.layers[5][2].bn2.running_var.data = items[99][1]
        self.layers[5][2].conv3.weight.data = items[100][1]
        self.layers[5][2].bn3.weight.data = items[101][1]
        self.layers[5][2].bn3.bias.data = items[102][1]
        self.layers[5][2].bn3.running_mean.data = items[103][1]
        self.layers[5][2].bn3.running_var.data = items[104][1]

        self.layers[5][3].conv1.weight.data = items[105][1] # layer 2.3
        self.layers[5][3].bn1.weight.data = items[106][1]
        self.layers[5][3].bn1.bias.data = items[107][1]
        self.layers[5][3].bn1.running_mean.data = items[108][1]
        self.layers[5][3].bn1.running_var.data = items[109][1]
        self.layers[5][3].conv2.weight.data = items[110][1]
        self.layers[5][3].bn2.weight.data = items[111][1]
        self.layers[5][3].bn2.bias.data = items[112][1]
        self.layers[5][3].bn2.running_mean.data = items[113][1]
        self.layers[5][3].bn2.running_var.data = items[114][1]
        self.layers[5][3].conv3.weight.data = items[115][1]
        self.layers[5][3].bn3.weight.data = items[116][1]
        self.layers[5][3].bn3.bias.data = items[117][1]
        self.layers[5][3].bn3.running_mean.data = items[118][1]
        self.layers[5][3].bn3.running_var.data = items[119][1]
       
        #######

        self.layers[6][0].conv1.weight.data = items[120][1] # layer 3.0
        self.layers[6][0].bn1.weight.data = items[121][1]
        self.layers[6][0].bn1.bias.data = items[122][1]
        self.layers[6][0].bn1.running_mean.data = items[123][1]
        self.layers[6][0].bn1.running_var.data = items[124][1]
        self.layers[6][0].conv2.weight.data = items[125][1]
        self.layers[6][0].bn2.weight.data = items[126][1]
        self.layers[6][0].bn2.bias.data = items[127][1]
        self.layers[6][0].bn2.running_mean.data = items[128][1]
        self.layers[6][0].bn2.running_var.data = items[129][1]
        self.layers[6][0].conv3.weight.data = items[130][1]
        self.layers[6][0].bn3.weight.data = items[131][1]
        self.layers[6][0].bn3.bias.data = items[132][1]
        self.layers[6][0].bn3.running_mean.data = items[133][1]
        self.layers[6][0].bn3.running_var.data = items[134][1]

        self.layers[6][1].conv1.weight.data = items[140][1] # layer 3.1
        self.layers[6][1].bn1.weight.data = items[141][1]
        self.layers[6][1].bn1.bias.data = items[142][1]
        self.layers[6][1].bn1.running_mean.data = items[143][1]
        self.layers[6][1].bn1.running_var.data = items[144][1]
        self.layers[6][1].conv2.weight.data = items[145][1]
        self.layers[6][1].bn2.weight.data = items[146][1]
        self.layers[6][1].bn2.bias.data = items[147][1]
        self.layers[6][1].bn2.running_mean.data = items[148][1]
        self.layers[6][1].bn2.running_var.data = items[149][1]
        self.layers[6][1].conv3.weight.data = items[150][1]
        self.layers[6][1].bn3.weight.data = items[151][1]
        self.layers[6][1].bn3.bias.data = items[152][1]
        self.layers[6][1].bn3.running_mean.data = items[153][1]
        self.layers[6][1].bn3.running_var.data = items[154][1]

        self.layers[6][2].conv1.weight.data = items[155][1] # layer 3.2
        self.layers[6][2].bn1.weight.data = items[156][1]
        self.layers[6][2].bn1.bias.data = items[157][1]
        self.layers[6][2].bn1.running_mean.data = items[158][1]
        self.layers[6][2].bn1.running_var.data = items[159][1]
        self.layers[6][2].conv2.weight.data = items[160][1]
        self.layers[6][2].bn2.weight.data = items[161][1]
        self.layers[6][2].bn2.bias.data = items[162][1]
        self.layers[6][2].bn2.running_mean.data = items[163][1]
        self.layers[6][2].bn2.running_var.data = items[164][1]
        self.layers[6][2].conv3.weight.data = items[165][1]
        self.layers[6][2].bn3.weight.data = items[166][1]
        self.layers[6][2].bn3.bias.data = items[167][1]
        self.layers[6][2].bn3.running_mean.data = items[168][1]
        self.layers[6][2].bn3.running_var.data = items[169][1]

        self.layers[6][3].conv1.weight.data = items[170][1] # layer 3.3
        self.layers[6][3].bn1.weight.data = items[171][1]
        self.layers[6][3].bn1.bias.data = items[172][1]
        self.layers[6][3].bn1.running_mean.data = items[173][1]
        self.layers[6][3].bn1.running_var.data = items[174][1]
        self.layers[6][3].conv2.weight.data = items[175][1]
        self.layers[6][3].bn2.weight.data = items[176][1]
        self.layers[6][3].bn2.bias.data = items[177][1]
        self.layers[6][3].bn2.running_mean.data = items[178][1]
        self.layers[6][3].bn2.running_var.data = items[179][1]
        self.layers[6][3].conv3.weight.data = items[180][1]
        self.layers[6][3].bn3.weight.data = items[181][1]
        self.layers[6][3].bn3.bias.data = items[182][1]
        self.layers[6][3].bn3.running_mean.data = items[183][1]
        self.layers[6][3].bn3.running_var.data = items[184][1]

        self.layers[6][4].conv1.weight.data = items[185][1] # layer 3.4
        self.layers[6][4].bn1.weight.data = items[186][1]
        self.layers[6][4].bn1.bias.data = items[187][1]
        self.layers[6][4].bn1.running_mean.data = items[188][1]
        self.layers[6][4].bn1.running_var.data = items[189][1]
        self.layers[6][4].conv2.weight.data = items[190][1]
        self.layers[6][4].bn2.weight.data = items[191][1]
        self.layers[6][4].bn2.bias.data = items[192][1]
        self.layers[6][4].bn2.running_mean.data = items[193][1]
        self.layers[6][4].bn2.running_var.data = items[194][1]
        self.layers[6][4].conv3.weight.data = items[195][1]
        self.layers[6][4].bn3.weight.data = items[196][1]
        self.layers[6][4].bn3.bias.data = items[197][1]
        self.layers[6][4].bn3.running_mean.data = items[198][1]
        self.layers[6][4].bn3.running_var.data = items[199][1]

        self.layers[6][5].conv1.weight.data = items[200][1] # layer 3.5
        self.layers[6][5].bn1.weight.data = items[201][1]
        self.layers[6][5].bn1.bias.data = items[202][1]
        self.layers[6][5].bn1.running_mean.data = items[203][1]
        self.layers[6][5].bn1.running_var.data = items[204][1]
        self.layers[6][5].conv2.weight.data = items[205][1]
        self.layers[6][5].bn2.weight.data = items[206][1]
        self.layers[6][5].bn2.bias.data = items[207][1]
        self.layers[6][5].bn2.running_mean.data = items[208][1]
        self.layers[6][5].bn2.running_var.data = items[209][1]
        self.layers[6][5].conv3.weight.data = items[210][1]
        self.layers[6][5].bn3.weight.data = items[211][1]
        self.layers[6][5].bn3.bias.data = items[212][1]
        self.layers[6][5].bn3.running_mean.data = items[213][1]
        self.layers[6][5].bn3.running_var.data = items[214][1]

        #######

        self.layers[7][0].conv1.weight.data = items[215][1] # layer 4.0
        self.layers[7][0].bn1.weight.data = items[216][1]
        self.layers[7][0].bn1.bias.data = items[217][1]
        self.layers[7][0].bn1.running_mean.data = items[218][1]
        self.layers[7][0].bn1.running_var.data = items[219][1]
        self.layers[7][0].conv2.weight.data = items[220][1]
        self.layers[7][0].bn2.weight.data = items[221][1]
        self.layers[7][0].bn2.bias.data = items[222][1]
        self.layers[7][0].bn2.running_mean.data = items[223][1]
        self.layers[7][0].bn2.running_var.data = items[224][1]
        self.layers[7][0].conv3.weight.data = items[225][1]
        self.layers[7][0].bn3.weight.data = items[226][1]
        self.layers[7][0].bn3.bias.data = items[227][1]
        self.layers[7][0].bn3.running_mean.data = items[228][1]
        self.layers[7][0].bn3.running_var.data = items[229][1]

        self.layers[7][1].conv1.weight.data = items[235][1] # layer 4.1
        self.layers[7][1].bn1.weight.data = items[236][1]
        self.layers[7][1].bn1.bias.data = items[237][1]
        self.layers[7][1].bn1.running_mean.data = items[238][1]
        self.layers[7][1].bn1.running_var.data = items[239][1]
        self.layers[7][1].conv2.weight.data = items[240][1]
        self.layers[7][1].bn2.weight.data = items[241][1]
        self.layers[7][1].bn2.bias.data = items[242][1]
        self.layers[7][1].bn2.running_mean.data = items[243][1]
        self.layers[7][1].bn2.running_var.data = items[244][1]
        self.layers[7][1].conv3.weight.data = items[245][1]
        self.layers[7][1].bn3.weight.data = items[246][1]
        self.layers[7][1].bn3.bias.data = items[247][1]
        self.layers[7][1].bn3.running_mean.data = items[248][1]
        self.layers[7][1].bn3.running_var.data = items[249][1]

        self.layers[7][2].conv1.weight.data = items[250][1] # layer 4.2
        self.layers[7][2].bn1.weight.data = items[251][1]
        self.layers[7][2].bn1.bias.data = items[252][1]
        self.layers[7][2].bn1.running_mean.data = items[253][1]
        self.layers[7][2].bn1.running_var.data = items[254][1]
        self.layers[7][2].conv2.weight.data = items[255][1]
        self.layers[7][2].bn2.weight.data = items[256][1]
        self.layers[7][2].bn2.bias.data = items[257][1]
        self.layers[7][2].bn2.running_mean.data = items[258][1]
        self.layers[7][2].bn2.running_var.data = items[259][1]
        self.layers[7][2].conv3.weight.data = items[260][1]
        self.layers[7][2].bn3.weight.data = items[261][1]
        self.layers[7][2].bn3.bias.data = items[262][1]
        self.layers[7][2].bn3.running_mean.data = items[263][1]
        self.layers[7][2].bn3.running_var.data = items[264][1]

        self.layers[9][0].weight.data = items[265][1] #fc
        self.layers[9][0].bias.data = items[266][1]
    
    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat('../models/imagenet-vgg-m.mat') # hardcoding this for now so that it works
        mat_layers = list(mat['layers'])[0]
        
        # copy conv weights, this works for the imagenet-vgg-m.mat file that the repo mentions
        '''
        for i in range(3):
            weight, bias = mat_layers[i*4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3,2,0,1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:,0])
        '''
        # imagenet-resnet-50-dag.mat file stores the weight info in 'params' instead of 'weights'
        
        own_state = self.state_dict()

        new_state = model_zoo.load_url(model_urls['resnet50'])
        for name, param in new_state.items():
            n = name.split('.')[0]
        for i in range(10):
            print(self.layers[i][0])

        #print(new_params.keys())
        #self.load_state_dict(new_params, strict=False)
        self.state_dict().update(own_state)
       # print(self.state_dict())
        '''
        mat = scipy.io.loadmat(matfile)
        self.layers[0][0].weight.data = torch.from_numpy(np.transpose(mat['params'][0][0][1], (3, 2, 0, 1)))
        self.layers[0][0].bias.data = torch.from_numpy(mat['params'][0][1][1][:,0])
        print(self.layers[0][0].weight.data.size())

        self.layers[1][0].weight.data = torch.from_numpy(np.transpose(mat['params'][0][0][1], (3, 2, 0, 1)))
        self.layers[1][0].bias.data = torch.from_numpy(mat['params'][0][1][1][:,0])

        self.layers[2][0].weight.data = torch.from_numpy(np.transpose(mat['params'][0][0][1], (3, 2, 0, 1)))
        self.layers[2][0].bias.data = torch.from_numpy(mat['params'][0][1][1][:,0])
        '''
        '''
        #printing the layer names in ResNet
        mat = scipy.io.loadmat(matfile)
        print('Printing ResNet layers that are not being used')
        for m in mat['params'][0][2:]:
            print(m[0])
        '''

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()
 
    def forward(self, pos_score, neg_score):
        pos_loss = -F.log_softmax(pos_score)[:,1]
        neg_loss = -F.log_softmax(neg_score)[:,0]
        
        loss = pos_loss.sum() + neg_loss.sum()
        return loss


class Accuracy():
    def __call__(self, pos_score, neg_score):
        
        pos_correct = (pos_score[:,1] > pos_score[:,0]).sum().float()
        neg_correct = (neg_score[:,1] < neg_score[:,0]).sum().float()
        
        pos_acc = pos_correct / (pos_score.size(0) + 1e-8)
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)

        return pos_acc.data[0], neg_acc.data[0]


class Precision():
    def __call__(self, pos_score, neg_score):
        
        scores = torch.cat((pos_score[:,1], neg_score[:,1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0)+1e-8)
        
        return prec.data[0]