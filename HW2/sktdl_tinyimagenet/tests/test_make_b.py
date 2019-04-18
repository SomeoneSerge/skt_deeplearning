import pytest
import torch
from sktdl_tinyimagenet import model


def test_ResBlock():
    net = model.ResBlock(3, 2, 2, .2)
    input = torch.randn(10, 3, 28, 28)
    output = net.forward(input)
    reached_without_exceptions = True
    assert reached_without_exceptions

def test_ResBlock():
    net = model.make_wideresnet(n_classes=200, layers_per_stage=3, widen_factor=3, drop_rate=.2)
    input = torch.randn(2, 3, 64, 64)
    output = net.forward(input)
    assert output.shape[0] == 2
    assert output.shape[1] == 200
    assert output.dim() == 2
    reached_without_exceptions = True
    assert reached_without_exceptions
