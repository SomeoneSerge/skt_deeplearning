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
    net = model.make_wideresnet(
            n_classes=200,
            make_conv=model.bn_relu_conv,
            depth=22,
            widen_factor=3,
            drop_rate=.2,
            apooling_cls=torch.nn.AdaptiveMaxPool2d,
            apooling_output_size=(10,10),
            append_logsoftmax=True)
    input = torch.randn(2, 3, 64, 64)
    output = net.forward(input)
    assert output.shape[0] == 2
    assert output.shape[1] == 200
    assert output.dim() == 2
    reached_without_exceptions = True
    assert reached_without_exceptions


def test_ImmersionHasNoParams():
    m = model.Immersion(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    assert len(list(m.parameters())) == 0
