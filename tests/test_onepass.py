import pytest
import torch

from patchdfs.common import MaskLayer2d
from patchdfs.dime.model import DIMENet, PredictorViT, ValueNetworkViT
from patchdfs.sparse.onepass import OnePassViT, cls_attention


def _backbone():
    import timm

    torch.manual_seed(0)
    return timm.create_model("vit_small_patch16_224", pretrained=False).eval()


def test_cls_attention_is_a_distribution_over_patches():
    bb = _backbone()
    x = torch.randn(2, 197, bb.embed_dim)
    s = cls_attention(bb.blocks[0], x)
    assert s.shape == (2, 196) and (s >= 0).all()
    # the CLS token also attends to itself, so slightly less than all mass goes to the patches
    assert ((s.sum(1) > 0.9) & (s.sum(1) < 1.0)).all()


def test_full_budget_reproduces_dense_vit():
    bb = _backbone()
    net = OnePassViT(bb, glance_layers=2, fuse_dropped=False).eval()
    dense = DIMENet(MaskLayer2d(False, 14, 16), PredictorViT(bb, 10), ValueNetworkViT(bb), backbone=bb).eval()
    dense.predictor.fc.load_state_dict(net.fc.state_dict())  # same classifier weights
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        ours, idx, score = net(x, 196)
        ref = dense.predict(x, torch.ones(2, 196))
    assert torch.allclose(ours, ref, atol=1e-3), (ours - ref).abs().max()
    assert idx.shape == (2, 196) and score.shape == (2, 196)


def test_partial_budget_shapes_and_uniqueness():
    bb = _backbone()
    for fuse in (False, True):
        net = OnePassViT(bb, glance_layers=2, fuse_dropped=fuse).eval()
        assert net.token_count(10) == 1 + 10 + int(fuse)
        x = torch.randn(3, 3, 224, 224)
        with torch.no_grad():
            logits, idx, score = net(x, 10)
        assert logits.shape == (3, 10)
        assert idx.shape == (3, 10) and (idx < 196).all()
        assert all(len(set(row.tolist())) == 10 for row in idx)
        # the kept patches are exactly the top-10 by CLS attention
        top = score.topk(10, dim=1).indices
        assert torch.equal(idx.sort(1).values, top.sort(1).values)


def test_invalid_glance_layers_rejected():
    bb = _backbone()
    with pytest.raises(ValueError):
        OnePassViT(bb, glance_layers=0)
    with pytest.raises(ValueError):
        OnePassViT(bb, glance_layers=len(bb.blocks))
