import torch

from patchdfs.models import (
    CifarClassifierHead,
    CifarResNetBackbone,
    CifarSelectorHead,
    MLP,
)


def test_cifar_resnet_shapes():
    bb = CifarResNetBackbone()
    x = torch.randn(2, 3, 32, 32)
    f = bb(x)
    assert f.shape == (2, 512, 4, 4)
    assert CifarClassifierHead()(f).shape == (2, 10)
    logits = CifarSelectorHead()(f)
    assert logits.shape == (2, 1, 8, 8), logits.shape


def test_mlp_shapes():
    net = MLP(784 * 2, 512, 10)
    assert net(torch.randn(4, 784 * 2)).shape == (4, 10)
    value = MLP(784 * 2, 512, 784)
    assert value(torch.randn(4, 784 * 2)).shape == (4, 784)
