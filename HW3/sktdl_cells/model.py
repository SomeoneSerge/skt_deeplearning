import torch
from torch import nn
from pytorch_unet.unet import Unet

class UnetAsExtractor(nn.Module):
    def __init__(self):
        super(UnetAsExtractor, self).__init__()
        self.inconv = torch.nn.Conv2d(3, 3, kernel_size=5, padding=2)
        self.unet = UNet(3, 1)
        self.outconv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)
    def forward(self, input):
            out = input
            out = self.inconv(out)
            out = self.unet(out)
            out = self.outconv(out)
            return out
