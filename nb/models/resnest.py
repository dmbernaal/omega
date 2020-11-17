import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from ..activations import Swish, Mila, Mish, BentID
from torch.nn.init import zeros_

__all__ = ['resnest50', 'mininest_ba', 'mininest_bn']

def get_activation_(act='relu', inplace=True):
    _activations_ = nn.ModuleDict([
        ['relu', nn.ReLU(inplace=inplace)],
        ['lrelu', nn.LeakyReLU(inplace=inplace)],
        ['swish', Swish()],
        ['bent_id', BentID()],
        ['mila', Mila()],
        ['mish', Mish()]
    ])
    return _activations_[act]

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super(rSoftMax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality
        
    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else: x = torch.sigmoid(x)
        return x
    
"""
Split Attention Conv2d
"""
class SplAtConv2d(nn.Module):
    def __init__(self, ni, nf, ks, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, bias=True, radix=2, reduction_factor=4, norm_layer=None, act='relu', **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        inter_channels = max(ni*radix//reduction_factor, 32)
        self.use_bn = norm_layer is not None
        self.radix = radix
        self.cardinality = groups
        self.channels = nf
        if self.use_bn:
            self.bn0 = nn.BatchNorm2d(nf*radix)
            self.bn1 = nn.BatchNorm2d(inter_channels)
            bias = False
        self.conv = nn.Conv2d(ni, nf*radix, kernel_size=ks, stride=stride, padding=padding, dilation=dilation, groups=groups*radix, bias=bias, **kwargs)
        self.act_fn = get_activation_(act)
        self.conv_fc1 = nn.Conv2d(nf, inter_channels, 1, groups=self.cardinality)
        self.conv_fc2 = nn.Conv2d(inter_channels, nf*radix, 1, groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, groups)
        
    def forward(self, x):
        x = self.conv(x)
        if self.use_bn: x = self.bn0(x)
        x = self.act_fn(x)
        
        batch, rchannel = x.shape[:2]
        
        if self.radix > 1:
            splitted = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splitted)
        else: gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.conv_fc1(gap)
        
        if self.use_bn: gap = self.bn1(gap)
        gap = self.act_fn(gap)
        
        atten = self.conv_fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        
        if self.radix > 1:
            attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([attn*split for (attn,split) in zip(attens, splitted)])
        else: out = atten * x
        
        return out.contiguous()
    
class GlobalAvgPool2d(nn.Module):
    """Global average pooling over the input's spatial dimensions"""
    def __init__(self): super(GlobalAvgPool2d, self).__init__()
    def forward(self, x): return F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
    
class Noop(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x
    
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, radix=1, cardinality=1, bottleneck_width=64, avd=False, avd_first=False, dilation=1, is_first=False, norm_layer=None, last_gamma=False, act='relu'):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width/64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first
        
        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1
            
        if radix >= 1:
            self.conv2 = SplAtConv2d(
                group_width, group_width, ks=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix, norm_layer=norm_layer, act=act)
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation, dilation=dilation,
                groups=cardinality, bias=False)
            self.bn2 = nn.BatchNorm2d(group_width)
            
        self.conv3 = nn.Conv2d(
            group_width, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        
        if last_gamma: zeros_(self.bn3.weight)
        
        self.act = get_activation_(act)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        if self.avd and self.avd_first: out = self.avd_layer(out)
            
        out = self.conv2(out)
        if self.radix==0:
            out = self.bn2(out)
            out = self.act(out)
        if self.avd and not self.avd_first: out = self.avd_layer(out)
            
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None: residual = self.downsample(x)
        out += residual
        out = self.act(out)
        
        return out
    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, radix=1, cardinality=1, bottleneck_width=64, avd=False, avd_first=False, dilation=1, is_first=False, norm_layer=None, last_gamma=False, act='relu'):
        super(BasicBlock, self).__init__()
        group_width = int(planes * (bottleneck_width/64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=3, bias=False, stride=stride, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        if self.avd: stride = 1
            
        if radix >= 1:
            self.conv2 = SplAtConv2d(
                group_width, group_width, ks=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix, norm_layer=norm_layer, act=act)
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation, dilation=dilation,
                groups=cardinality, bias=False)
            self.bn2 = nn.BatchNorm2d(group_width)
        
        if last_gamma: zeros_(self.bn2.weight)
        
        self.act = get_activation_(act)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
            
        out = self.conv2(out)
        if self.radix==0:
            out = self.bn2(out)
            
        if self.downsample is not None: residual = self.downsample(x)
        out += residual
        out = self.act(out)
        
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, radix=1, groups=1, bottleneck_width=64, c_in=3, num_classes=1000, dilated=False, dilation=1, deep_stem=False, stem_width=64, avg_down=False, avd=False, avd_first=False, final_drop=0.0, last_gamma=False, norm_layer=True, act='relu'):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        self.inplanes = stem_width*2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        self.norm_layer = norm_layer
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first
        
        super(ResNet, self).__init__()
        self.act_fn = get_activation_(act)
        conv_layer = nn.Conv2d
        conv_kwargs = {}
        if deep_stem:
            self.conv1 = nn.Sequential(
                conv_layer(c_in, stem_width, kernel_size=3, stride=2, padding=1, bias=False, **conv_kwargs),
                nn.BatchNorm2d(stem_width) if norm_layer else Noop(), self.act_fn,
                conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
                nn.BatchNorm2d(stem_width) if norm_layer else Noop(), self.act_fn,
                conv_layer(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs))
        else: self.conv1 = conv_layer(c_in, 64, kernel_size=7, stride=3, padding=3, bias=False, **conv_kwargs)
        self.bn1 = nn.BatchNorm2d(self.inplanes) if norm_layer else None
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False, act=act)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer, act=act)
        
        if dilated or dilation==4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer, act=act)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, norm_layer=norm_layer, act=act)
        elif dilation==2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=1, norm_layer=norm_layer, act=act)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, norm_layer=norm_layer, act=act)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer, act=act)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer, act=act)
        
        self.avgpool = GlobalAvgPool2d()
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. /n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, is_first=True, act='relu'):
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1: down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
                else: down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1, ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False))
            else: down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False))
            if self.norm_layer: down_layers.append(nn.BatchNorm2d(planes*block.expansion))
            downsample = nn.Sequential(*down_layers)
        
        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality, bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first, dilation=1, is_first=is_first, norm_layer=norm_layer,
                                last_gamma=self.last_gamma, act=act))
        elif dilation==4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality, bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first, dilation=2, is_first=is_first, norm_layer=norm_layer,
                                last_gamma=self.last_gamma, act=act))
        else: raise RuntimeError("=> unknown dilation size: {}".format(dilation))
        
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width, avd=self.avd,
                                avd_first=self.avd_first, dilation=dilation, norm_layer=norm_layer,
                                last_gamma=self.last_gamma, act=act))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        if self.bn1: x = self.bn1(x)
        x = self.act_fn(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.drop: x = self.drop(x)
        x = self.fc(x)
        
        return x
    
def resnest50(c_in=3, num_classes=1000, act='relu', **kwargs):
    layers = [3, 4, 5, 6]
    model = ResNet(Bottleneck, layers,
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, c_in=c_in, num_classes=num_classes, act=act, **kwargs)
    return model

def mininest_ba(c_in=3, num_classes=1000, act='relu', **kwargs):
    layers = [1, 1, 1, 1]
    model = ResNet(BasicBlock, layers,
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, c_in=c_in, num_classes=num_classes, act=act, **kwargs)
    return model

def mininest_bn(c_in=3, num_classes=1000, act='relu', **kwargs):
    layers = [1, 1, 1, 1]
    model = ResNet(Bottleneck, layers,
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, c_in=c_in, num_classes=num_classes, act=act, **kwargs)
    return model