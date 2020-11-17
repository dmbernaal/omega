import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Swish', 'Mila', 'Mish', 'BentID']

class Swish(nn.Module):
    def __init__(self, device='cuda:0'):
        super().__init__()
        self.device=torch.device(device)
        self.beta = nn.Parameter(torch.randn(1))
        self.beta.requires_grad = True

    def forward(self, x):
        x = x.to('cpu')
        t = torch.sigmoid(self.beta * x)
        return (x * t).to(self.device)

class Mila(nn.Module):
    def __init__(self, beta=-0.25):
        super().__init__()
        self.beta = beta
    def forward(self, x): return x * torch.tanh(F.softplus(x + self.beta))
    
class Mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return mish(input)
    
class BentID(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, input): return input + ((torch.sqrt(torch.pow(input, 2) + 1) - 1) / 2)
    
@torch.jit.script
def mish(input):
    return input * torch.tanh(F.softplus(input))