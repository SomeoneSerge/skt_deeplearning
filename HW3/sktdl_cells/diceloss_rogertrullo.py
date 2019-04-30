# Source: https://github.com/rogertrullo/pytorch/blob/rogertrullo-dice_loss/torch/nn/functional.py#L708


import numpy as np
import torch
import torch.nn.functional as F


def dice_loss(input, target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    cpu = torch.device('cpu')
    uniques=np.unique(target.to(cpu).numpy())
    assert set(list(uniques)) <= set([0,1]), "target must only contain zeros and ones"

    probs = F.softmax(input)
    num = probs*target # b, c, h,w--p*g
    num = torch.sum(num, dim=3) #b, c, h
    num = torch.sum(num, dim=2)
    

    den1 = probs*probs # -- p^2
    den1 = torch.sum(den1, dim=3) # b,c,h
    den1 = torch.sum(den1, dim=2)
    

    den2 = target*target # -- g^2
    den2 = torch.sum(den2, dim=3) # b,c,h
    den2 = torch.sum(den2, dim=2) # b,c
    

    dice = 2*(num/(den1+den2))
    dice_eso = dice[:, 1:] # we ignore bg dice val, and take the fg

    dice_total = -1 * torch.sum(dice_eso)/dice_eso.size(0) # divide by batch_sz

    return dice_total
