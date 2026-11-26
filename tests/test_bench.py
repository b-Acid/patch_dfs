import torch

from patchdfs.bench.flops import conv_linear_macs, count_flops


def test_flop_counter_matches_linear_macs():
    lin = torch.nn.Linear(64, 32)
    x = torch.randn(8, 64)
    flops, _ = count_flops(lambda: lin(x))
    assert abs(flops - 2 * conv_linear_macs(lin, x)) < 1e-3 * flops


def test_conv_macs_cifar_backbone():
    from patchdfs.models import CifarResNetBackbone

    bb = CifarResNetBackbone().eval()
    macs = conv_linear_macs(bb, torch.randn(1, 3, 32, 32))
    assert 2.0e8 < macs < 3.0e8, macs  # about 2.5e8 MACs
