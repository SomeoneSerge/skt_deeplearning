import torch
import numpy as np
import math


def bn_relu_conv(
        in_features, out_features, kernel_size,
        stride=1,
        drop_rate=0.,
        padding=1,
        bias=False):
    layers = []
    layers += [
            torch.nn.BatchNorm2d(in_features),
            torch.nn.ReLU(),
    ]
    if drop_rate > 0:
        layers += [torch.nn.Dropout(drop_rate)]
    layers += [
            torch.nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                ),
    ]
    return tuple(layers)

def conv_bn_relu(
        in_features, out_features, kernel_size,
        stride=1,
        drop_rate=0.,
        padding=1,
        bias=False):
    layers = []
    layers += [
            torch.nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                ),
    ]
    if drop_rate > 0:
        layers += [torch.nn.Dropout(drop_rate)]
    layers += [
            torch.nn.BatchNorm2d(out_features),
            torch.nn.ReLU(),
    ]
    return tuple(layers)


class Immersion(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Immersion, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)*2
        self.register_buffer('weight', torch.ones(out_channels, in_channels, *kernel_size))
        self.weight.div_(in_channels*kernel_size[0]*kernel_size[1])
        self.weight.requires_grad_(False)
        self.stride = stride
        self.padding = padding
    def forward(self, input):
        assert not hasattr(self.weight, 'grad') or self.weight.grad is None
        return torch.conv2d(input, self.weight, stride=self.stride, padding=self.padding)

class ResBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, make_conv, stride, drop_rate):
        """`B(?, ?)` from `https://arxiv.org/pdf/1605.07146.pdf`"""
        super(ResBlock, self).__init__()
        res_layers = tuple()
        res_layers += make_conv(in_features, out_features,
                kernel_size=3,
                stride=stride,
                drop_rate=0.,
                bias=False)
        res_layers += make_conv(
                out_features, out_features,
                kernel_size=3,
                stride=1,
                padding=1,
                drop_rate=drop_rate,
                bias=False)
        self.residual = torch.nn.Sequential(*res_layers)
        if in_features != out_features:
            self.shortcut = Immersion(
                    in_features, out_features,
                    kernel_size=1,
                    stride=stride,
                    padding=0)
        else:
            self.shortcut = None
    def forward(self, input):
        out = self.residual.forward(input)
        if self.shortcut:
            out += self.shortcut.forward(input)
        else:
            out += input
        return out

class Flatten(torch.nn.Module):
    def forward(self, input):
        return torch.flatten(input, start_dim=1)

def make_wideresnet(
        n_classes, depth,
        make_conv,
        apooling_cls,
        resblock_strides,
        widen_factor=3, drop_rate=.2):
    assert (depth - 4) % 6 == 0
    print('widen={}, depth={}'.format(widen_factor, depth))
    n = (depth - 4) // 6
    layers = [ torch.nn.Conv2d(3, 16, 1) ]
    channels = [16] + [2**(3 + k) * widen_factor for k in [1, 2, 3]]
    in_channels = channels[:-1]
    out_channels = channels[1:]
    for in_channels, out_channels, stride in zip(in_channels, out_channels, resblock_strides):
        layers += [ ResBlock(in_channels, out_channels, make_conv, stride, drop_rate) ]
        for i in range(1, n):
            layers += [ ResBlock(out_channels, out_channels, make_conv, 1, drop_rate) ]
    layers += [
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            apooling_cls(1),
            Flatten(),
            torch.nn.Linear(
                channels[-1],
                n_classes),
            ]
    return torch.nn.Sequential(*layers)
