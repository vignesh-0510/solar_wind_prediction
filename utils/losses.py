import torch
import torch.nn as nn
from neuralop import LpLoss

class L2OperatorLoss(nn.Module):
    """
    L2 operator norm loss for DeepONet-style operator learning.
    
    Input:
        pred   : (B, N) or (B, N, C)
        target : (B, N) or (B, N, C)
    Output:
        scalar loss
    """
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        # pointwise squared error
        diff = pred - target
        sq_error = diff.pow(2)

        # integrate over spatial domain
        loss_per_function = sq_error.mean(dim=1)  # (B,)

        if self.reduction == "mean":
            return loss_per_function.mean()
        elif self.reduction == "sum":
            return loss_per_function.sum()
        else:
            return loss_per_function

def fetch_loss_function(loss_fn_str):
    if loss_fn_str == "l2":
        return LpLoss(d=2, p=2)
    elif loss_fn_str == "l2l1":
        return L1L2Loss(d=2)
    elif loss_fn_str == "mse":
        return nn.MSELoss()
    else:
        raise ValueError('loss should be either "l2" or "l2l1"')
