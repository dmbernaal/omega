import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from fastcore import foundation as fd
import math
from ..activations import Swish, Mila, Mish, BentID

delegates = fd.delegates

__all__ = ['xmininet', 'xsemininet']

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

"""
TODO: Expand with various norms: Group norm & Instance norm
"""
_norms_ = {'bn': nn.BatchNorm2d, 'gn': nn.GroupNorm}
def _get_norm(norm):
    try: return _norms_[norm] if norm is not None else None
    except Exception as e: raise ValueError(f'{e}\nWrong norm type use [bn, gn]')
        
"""
TODO: Expand with various activations: Follow FastAI for proper 
"""
_activations_ = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid}
def _get_act_cls(act_cls):
    try: return get_activation_(act_cls) if act_cls is not None else None
    except Exception as e: raise ValueError(f'{e}\nWrong activation')
        
# def _get_act_cls(act_cls):
#     try: return _activations_[act_cls] if act_cls is not None else None
#     except Exception as e: raise ValueError(f'{e}\nWrong activation')
        
def AvgPool(ks=2, stride=None, padding=0, ceil_mode=False):
    return nn.AvgPool2d(ks, stride=stride, padding=padding, ceil_mode=ceil_mode)

def MaxPool(ks=2, stride=None, padding=0, ceil_mode=False):
    return nn.MaxPool2d(ks, stride=stride, padding=padding)

def AdaptiveAvgPool(sz=1):
    return nn.AdaptiveAvgPool2d(sz)

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)
    
def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)
        
def SEBlock(expansion, ni, nf, reduction=16, stride=1, **kwargs):
    return ResBlock(expansion, ni, nf, stride=stride, reduction=reduction, nh1=nf*2, nh2=nf*expansion, **kwargs)

class SequentialEx(nn.Module):
    """
    Like nn.sequential but with ModuleList semantics sand can access module input
    """
    def __init__(self, *layers): 
        super(SequentialEx, self).__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        res = x
        for l in self.layers:
            res.orig = x
            nres = l(res)
            res.orig = None
            res = nres
        return res
    
    def __getitem__(self, i): return self.layers[i]
    def append(self, l): return self.layers.append(l)
    def extend(self, l): return self.layers.extend(l)
    def insert(self, i, l): return self.layers.insert(i,l)
    
class ProdLayer(nn.Module):
    def __init__(self): 
        super(ProdLayer, self).__init__()
        pass
    def forward(self, x): return x * x.orig
    
def SEModule(ch, reduction, act_cls='relu'):
    nf = math.ceil(ch//reduction/8)*8
    return SequentialEx(
        nn.AdaptiveAvgPool2d(1),
        ConvLayer(ch, nf, ks=1, norm_type=None, act_cls=act_cls),
        ConvLayer(nf, ch, ks=1, norm_type=None, act_cls='sigmoid'),
        ProdLayer()
    )

def _conv1d_spect(ni, no, ks=1, stride=1, padding=0, bias=False):
    """
    Create and init a conv1d layer with spectral normalization
    """
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return spectral_norm(conv)

class SimpleSelfAttention(nn.Module):
    def __init__(self, n_in, ks=1, sym=False):
        super(SimpleSelfAttention, self).__init__()
        self.sym, self.n_in = sym, n_in
        self.conv = _conv1d_spect(n_in, n_in, ks, padding=ks//2, bias=False)
        self.gamma = nn.Parameter(torch.tensor([0.]))
        
    def forward(self, x):
        if self.sym:
            c = self.conv.weight.view(self.n_in, self.n_in)
            c = (c + c.t())/2
            self.conv.weight = c.view(self.n_in, self.n_in, 1)
        
        size = x.size()
        x = x.view(*size[:2], -1)
        
        convx = self.conv(x)
        xxT = torch.bmm(x, x.permute(0,2,1).contiguous())
        o = torch.bmm(xxt, convx)
        o = self.gamma * o + x
        return o.view(*size).contiguous()
    
class ConvLayer(nn.Sequential):
    """
    Creates a sequence of Conv, Act, Norm
    """
    @delegates(nn.Conv2d)
    def __init__(self, ni, nf, ks=3, stride=1, padding=None, bias=None, norm_type='bn', bn_1st=True, act_cls='relu', init='auto', xtra=None, bias_std=0.01, **kwargs):
        if padding is None: padding = ((ks-1)//2)
        norm = _get_norm(norm_type)
        bias = None if not (not norm) else bias
        conv = nn.Conv2d(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding, **kwargs)
        act = _get_act_cls(act_cls)
        layers = [conv]
        act_bn = []
        if act is not None: act_bn.append(act())
        if norm: act_bn.append(norm(nf))
        if bn_1st: act_bn.reverse()
        layers+=act_bn
        if xtra: layers.append(xtra)
        super().__init__(*layers)
        
class ResBlock(nn.Module):
    """
    Resnet block from ni to nh with stride
    """
    @delegates(ConvLayer.__init__)
    def __init__(self, expansion, ni, nf, stride=1, groups=1, reduction=None, nh1=None, nh2=None, dw=False, g2=1, sa=False, sym=False, norm_type='bn', act_cls='relu', ks=3, pool_first=True, **kwargs):
        super(ResBlock, self).__init__()
        norm1 = norm2 = norm_type
        pool = AvgPool
        if nh2 is None: nh2 = nf
        if nh1 is None: nh1 = nh2
        nf, ni = nf*expansion, ni*expansion
        k0 = dict(norm_type=norm1, act_cls=act_cls, **kwargs)
        k1 = dict(norm_type=norm2, act_cls=None, **kwargs)
        conv_path = [
            ConvLayer(ni, nh2, ks, stride=stride, **k0),
            ConvLayer(nh2, nf, ks, **k1)
        ] if expansion == 1 else [
            ConvLayer(ni, nh1, 1, **k0),
            ConvLayer(nh1, nh2, ks, stride=stride, **k0),
            ConvLayer(nh2, nf, 1, **k1)]
        if reduction: conv_path.append(SEModule(nf, reduction=reduction, act_cls=act_cls))
        if sa: conv_path.append(SimpleSelfAttention(nf, ks=1, sym=sym))
        self.conv_path = nn.Sequential(*conv_path)
        id_path = []
        if ni!=nf: id_path.append(ConvLayer(ni, nf, 1, act_cls=None, **kwargs))
        if stride!=1: id_path.insert((1,0)[pool_first], pool(stride, ceil_mode=True))
        self.id_path = nn.Sequential(*id_path)
        self.act = _get_act_cls(act_cls)(inplace=True) if act_cls=='relu' else _get_act_cls(act_cls)()
        
    def forward(self, x): return self.act(self.conv_path(x) + self.id_path(x))
    
class XResNet(nn.Sequential):
    @delegates(ResBlock)
    def __init__(self, block, expansion, layers, p=0.0, c_in=3, n_out=1000, stem_szs=(32, 32, 64), widen=1.0, sa=False, act_cls='relu', ks=3, stride=2, **kwargs):
        self.block, self.expansion, self.act_cls, self.ks = block, expansion, act_cls, ks
        if ks%2==0: raise Exception('Kernel size has to be odd')
        stem_szs = [c_in, *stem_szs]
        stem = [ConvLayer(stem_szs[i], stem_szs[i+1], ks=ks, stride=stride if i==0 else 1, act_cls=act_cls)
                for i in range(3)]
        
        block_szs = [int(o*widen) for o in [64,128,256,512] +[256]*(len(layers)-4)]
        block_szs = [64//expansion] + block_szs
        blocks = self._make_blocks(layers, block_szs, sa, stride, **kwargs)
        
        super().__init__(
            *stem, MaxPool(ks=ks, stride=stride, padding=ks//2),
            *blocks,
            AdaptiveAvgPool(sz=1), Flatten(), nn.Dropout(p),
            nn.Linear(block_szs[-1]*expansion, n_out),
        )
        init_cnn(self)
        
    def _make_blocks(self, layers, block_szs, sa, stride, **kwargs):
        return [self._make_layer(ni=block_szs[i], nf=block_szs[i+1], blocks=l,
                                 stride=1 if i==0 else stride, sa=sa and i==len(layers)-4, **kwargs)
                                 for i,l in enumerate(layers)]
    
    def _make_layer(self, ni, nf, blocks, stride, sa, **kwargs):
        return nn.Sequential(
            *[self.block(self.expansion, ni if i==0 else nf, nf, stride=stride if i==0 else 1,
                         sa=sa and i==(blocks-1), act_cls=self.act_cls, ks=self.ks, **kwargs)
                         for i in range(blocks)])
    
def _xresnet(expansion, layers, **kwargs):
    res = XResNet(ResBlock, expansion, layers, **kwargs)
    return res
    
def _xseresnet(expansion, layers, **kwargs):
    res = XResNet(SEBlock, expansion, layers, **kwargs)
    return res

def xmininet(n_in, n_out=1000, **kwargs):
    layers = [1, 1, 1, 1]
    return _xresnet(1, layers, c_in=n_in, n_out=n_out, **kwargs)

def xsemininet(n_in, n_out=1000, **kwargs):
    layers = [1, 1, 1, 1]
    return _xseresnet(1, layers, c_in=n_in, n_out=n_out, **kwargs)