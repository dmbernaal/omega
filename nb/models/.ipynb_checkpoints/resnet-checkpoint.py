import torch
import torch.nn as nn
import torch.nn.functional as F
from ..activations import Swish, Mila, Mish, BentID
from collections import OrderedDict

__all__ = ['mininet']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 conv with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 conv"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def create_norm_(inplanes, norm_='bn'):
    norm_type_ = nn.ModuleDict([
        ['bn', nn.BatchNorm2d(inplanes)],
        ['gn', nn.GroupNorm(1, inplanes)]
    ])
    return norm_type_[norm_]

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

def stem_block(in_c=3, inplanes=64, act='relu', norm='bn', ks=7, stride=2, padding=3):
    """
    Basic stem. Created at the beginning of the ResNet Arch
    """
    bias = False if norm=='bn' else True
    stem = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(in_c, inplanes, kernel_size=ks, stride=stride))
    ]))
    if norm is not None: stem.add_module(name='norm1', module=create_norm_(inplanes, norm))
    stem.add_module(name='act', module=get_activation_(act, inplace=True))
    stem.add_module(name='maxpool', module=nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    return stem

class Flatten(nn.Module):
    def __init__(self): super(Flatten, self).__init__()
    def forward(self, x): return torch.flatten(x, 1)
    
def create_head(planes, num_classes):
    """
    Basic head component. For classification or other various tasks
    """
    return nn.Sequential(OrderedDict([
        ['avgpool', nn.AdaptiveAvgPool2d((1,1))],
        ['flatten', Flatten()],
        ['fc', nn.Linear(planes, num_classes)]
    ]))

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, act='relu', norm='bn', stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64: raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1: raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm1 = None if norm is None else create_norm_(planes, norm)
        self.act = get_activation_(act, inplace=True)
        
        self.conv2 = conv3x3(planes, planes)
        self.norm2 = None if norm is None else create_norm_(planes, norm)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        if self.norm1: out = self.norm1(out)
        out = self.act(out)
        
        out = self.conv2(out)
        if self.norm2: out = self.norm2(out)
        if self.downsample: identity = self.downsample(x)
            
        out += identity
        out = self.act(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, act='relu', norm='bn', in_c=3, stem_ks=7, num_classes=1000, zero_init_residual=False, init_method='kaiming_normal', mode='fan_out', groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(ResNet, self).__init__()
        
        self._norm_layer = norm
        self._act_fn = act
        self.inplanes = 64
        self.dilation = 1
        
        if replace_stride_with_dilation is None: replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be a 3-element tuple')
            
        self.groups = groups
        self.base_width = width_per_group
        
        # Add more inits
        self._inits = {'kaiming_normal': nn.init.kaiming_normal_, 'kaiming_uniform': nn.init.kaiming_uniform_}
        
        # layers
        self.stem = stem_block(in_c=in_c, inplanes=self.inplanes, act=act, norm=norm, ks=stem_ks)
        self.body = nn.Sequential(OrderedDict([
            ['layer1', self._make_layer(block, 64, layers[0])],
            ['layer2', self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])],
            ['layer3', self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])],
            ['layer4', self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])]
        ]))
        self.head = create_head(512 * block.expansion, num_classes)
        
        # weight init
        self.init_model_(init_method, mode)
        if zero_init_residual and norm=='bn': self.zero_init_residual_()
            
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        act_fn = self._act_fn
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(OrderedDict([
                ['0', conv1x1(self.inplanes, planes * block.expansion, stride)]
            ]))
            if norm_layer:
                downsample.add_module(
                    name='1',
                    module=create_norm_(planes * block.expansion, norm_layer)
                )
        layers = []
        layers.append(block(self.inplanes, planes, act_fn, norm_layer, stride, downsample, self.groups, self.base_width, previous_dilation))
        
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, act_fn, norm_layer, groups=self.groups, base_width=self.base_width, dilation=self.dilation))
            
        return nn.Sequential(*layers)
    
    def init_model_(self, init_method='kaiming_normal', mode='fan_out'):
        init = self._inits[init_method]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self._act_fn == 'relu': init(m.weight, mode=mode, nonlinearity=self._act_fn)
                else: init(m.weight, mode=mode)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def zero_init_residual_(self):
        for m in self.body.modules():
            if isinstance(m, BasicBlock): nn.init.constant_(m.norm3.weigth, 0)
                
    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.head(x)
        return x
    
def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def mininet(in_c, num_classes=1000, **kwargs): return _resnet(BasicBlock, [1,1,1,1], in_c=in_c, num_classes=num_classes, **kwargs)