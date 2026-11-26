import torch

from patchdfs.common import MaskLayer2d
from patchdfs.dime.model import DIMENet, PredictorViT, ValueNetworkViT
from patchdfs.sparse.token_vit import TokenDroppedViT, block_forward


def _backbone():
    import timm

    torch.manual_seed(0)
    return timm.create_model("vit_small_patch16_224", pretrained=False).eval()


def test_block_forward_matches_timm_block():
    bb = _backbone()
    x = torch.randn(2, 197, bb.embed_dim)
    with torch.no_grad():
        ref = bb.blocks[0](x)
        ours = block_forward(bb.blocks[0], x, None)
    assert torch.allclose(ref, ours, atol=1e-5)


def test_patch_embedding_matches_dense_conv():
    bb = _backbone()
    net = TokenDroppedViT(bb)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        ours = net._patch_tokens(x)
        ref = bb.patch_embed(x)  # (B, 196, D)
    assert torch.allclose(ours, ref, atol=1e-4)


def test_full_mask_reproduces_dense_vit_and_partial_masks_run():
    bb = _backbone()
    net = TokenDroppedViT(bb).eval()
    dense = DIMENet(MaskLayer2d(False, 14, 16), PredictorViT(bb, 10), ValueNetworkViT(bb), backbone=bb).eval()
    dense.predictor.fc.load_state_dict(net.fc.state_dict())  # same classifier weights
    x = torch.randn(2, 3, 224, 224)
    full = torch.ones(2, 196)
    with torch.no_grad():
        p_full, _ = net.both(x, full)
        p_dense = dense.predict(x, full)
    assert torch.allclose(p_full, p_dense, atol=1e-3), (p_full - p_dense).abs().max()
    for frac in (0.0, 0.05, 0.5):
        m = (torch.rand(2, 196) < frac).float()
        m[0, :3] = 1  # make the two rows differ in length
        p, v = net.both(x, m)
        assert p.shape == (2, 10) and v.shape == (2, 196) and torch.isfinite(v).all()


def test_padded_tokens_do_not_affect_outputs():
    bb = _backbone()
    net = TokenDroppedViT(bb).eval()
    x = torch.randn(3, 3, 224, 224)
    m = torch.zeros(3, 196)
    m[:, :5] = 1
    m[2, 5:40] = 1  # third row much longer -> rows 0,1 are padded when batched together
    with torch.no_grad():
        p_batch, v_batch = net.both(x, m)
        p_single, v_single = net.both(x[:1], m[:1])
    assert torch.allclose(p_batch[0], p_single[0], atol=1e-3)
    assert torch.allclose(v_batch[0], v_single[0], atol=1e-3)