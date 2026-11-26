import math

import torch

from patchdfs.common import ConcreteSelector, MaskLayer, MaskLayer2d, generate_uniform_mask, make_onehot


def test_make_onehot_and_no_repeat_logic():
    logits = torch.tensor([[0.1, 0.9, 0.0], [5.0, 1.0, 1.0]])
    m = make_onehot(logits)
    assert torch.equal(m, torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))
    # Authors' no-repeat trick: subtract a large constant on already-selected entries.
    m2 = torch.max(m, make_onehot(logits - 1e6 * m))
    assert m2.sum(dim=1).tolist() == [2, 2]


def test_uniform_mask_cardinality_is_uniform():
    g = torch.Generator().manual_seed(0)
    m = generate_uniform_mask(200000, 20, generator=g)
    counts = torch.bincount(m.sum(dim=1).long(), minlength=21) / 200000
    assert abs(counts[0].item() - 1 / 21) < 0.005
    assert abs(counts[20].item() - 1 / 21) < 0.005


def test_mask_layer_1d_append():
    layer = MaskLayer(append=True, mask_size=4)
    x = torch.ones(2, 4)
    m = torch.tensor([[1.0, 0, 0, 0], [0, 0, 1.0, 1.0]])
    out = layer(x, m)
    assert out.shape == (2, 8)
    assert torch.equal(out[:, :4], m)
    assert torch.equal(out[:, 4:], m)


def test_mask_layer_2d_zeroes_patches():
    layer = MaskLayer2d(append=False, mask_width=8, patch_size=4)
    x = torch.ones(2, 3, 32, 32)
    m = torch.zeros(2, 64)
    m[:, 0] = 1  # top-left 4x4 patch
    m[:, 63] = 1  # bottom-right 4x4 patch
    out = layer(x, m)
    assert out.shape == x.shape
    assert out[:, :, :4, :4].sum() == 2 * 3 * 16
    assert out[:, :, -4:, -4:].sum() == 2 * 3 * 16
    assert out.sum() == 2 * 3 * 32
    # 4d mask input is accepted too
    out4 = layer(x, m.reshape(2, 1, 8, 8))
    assert torch.equal(out4, out)


def test_concrete_selector_is_one_hot_at_low_temperature():
    torch.manual_seed(0)
    sel = ConcreteSelector()
    logits = torch.randn(64, 10)
    soft = sel(logits, temp=0.01)
    assert soft.shape == (64, 10)
    assert torch.allclose(soft.sum(dim=1), torch.ones(64), atol=1e-4)
    assert (soft.max(dim=1).values > 0.99).float().mean() > 0.95
    det = sel(logits, temp=1.0, deterministic=True)
    assert torch.allclose(det.sum(dim=1), torch.ones(64), atol=1e-5)
    assert math.isfinite(det.sum().item())
