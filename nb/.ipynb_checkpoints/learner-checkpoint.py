import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch_lr_finder import LRFinder

__all__ = ['Learner', 'model_config']

def accuracy(out, yb):
    preds = torch.argmax(out.detach(), 1)
    correct = (yb==preds).sum().item()
    return correct

def model_config(data):
    import numpy as np
    xb,_ = next(iter(data.train_dl))
    c_in = xb.shape[1]
    num_classes = len(np.unique(data.train_ds.y[:,0]))
    return c_in, num_classes

def update_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg['lr'] = lr
        
def update_optimizer(optimizer, **params):
    for p in params:
        for pg in optimizer.param_groups:
            pg[p] = params[p]
        
def get_lr(optimizer):
    for pg in optimizer.param_groups: return pg['lr']
    
class Learner:
    def __init__(self, data, model, opt_fn=None, loss_fn=None, metrics=None, lr=None, wd=None, device='cuda:0', **kwargs):
        self.data, self.model = data, model
        self.lr = 1e-4 if lr is None else lr
        self.wd = 0. if wd is None else wd
        self.opt_fn = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd, **kwargs) if opt_fn is None else opt_fn
        self.loss_fn = nn.CrossEntropyLoss() if loss_fn is None else loss_fn
        self.metric_fn = accuracy if metrics is None else metrics
        self.device = torch.device(device)
        
    def save_model(self, path2save): torch.save(self.model.state_dict(), str(path2save))
    def load_model(self, path2weights): self.model.load_state_dict(torch.load(path2weights))
    def lr_finder(self, end_lr=100, num_iter=100):
        lr_finder = LRFinder(self.model, self.opt_fn, self.loss_fn, device=self.device)
        lr_finder.range_test(self.data.train_dl, end_lr=end_lr, num_iter=num_iter)
        lr_finder.plot()
        lr_finder.reset()
        
    def update_lr(self, lr): update_lr(self.opt_fn, lr)
        
    def update_opt(self, **params): update_optimizer(self.opt_fn, **params)
    
    def freeze(self, m=None):
        """
        TODO: change up model topology to be cosistant with: stem | body | head
        This will allow for us to freeze individual parts or comb of parts instead of freezing the entire model
        """
        m = self.model if m is None else m
        for pg in m.parameters():
            pg.requires_grad_(False)
            
    def unfreeze(self, m=None):
        """
        TODO: same as above
        """
        m = self.model if m is None else m
        for pg in m.parameters():
            pg.requires_grad_(True)
            
    def predict(self, x, confidence=0.5):
        """TODO: build predict"""
        pass
    
    def show_results(self, ds_type='valid', nrows=5, **kwargs):
        """
        TODO: Build show results.
        
        This method should grab a time window plot via price trend with upper/lower bound, ground truth and predicted labels. 
        """
        pass