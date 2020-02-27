from torch import nn
import torch.nn.functional as F

def nll_loss():
    return F.nll_loss()

def cross_entropy_loss():
    return nn.CrossEntropyLoss()