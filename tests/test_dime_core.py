import torch
from torch.utils.data import DataLoader, TensorDataset

from patchdfs.dime import build_dime
from patchdfs.dime.cmi import fit_dime, greedy_rollout, infer, train_epoch


def _tiny_loaders():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 784, generator=g)
    y = torch.randint(0, 10, (64,), generator=g)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=16), DataLoader(ds, batch_size=16)


def test_build_dime_mlp_ties_first_layer():
    net = build_dime({"kind": "mlp", "hidden": 32, "dropout": 0.3, "append_mask": True}, 784, 10)
    assert net.predictor[0] is net.value_network[0]
    m = torch.zeros(3, 784)
    p, v = net.both(torch.randn(3, 784), m)
    assert p.shape == (3, 10) and v.shape == (3, 784)
    assert net.mask_size == 784


def test_fused_vit_matches_two_passes():
    import timm

    from patchdfs.common import MaskLayer2d
    from patchdfs.dime.model import DIMENet, PredictorViT, ValueNetworkViT

    torch.manual_seed(0)
    backbone = timm.create_model("vit_small_patch16_224", pretrained=False)
    net = DIMENet(MaskLayer2d(False, 14, 16), PredictorViT(backbone, 10), ValueNetworkViT(backbone), backbone=backbone)
    net.eval()
    x = torch.randn(2, 3, 224, 224)
    m = torch.zeros(2, 196)
    m[:, 3] = 1
    with torch.no_grad():
        p1, v1 = net.both(x, m)
        p2, v2 = net.predict(x, m), net.value_logits(x, m)
    assert torch.allclose(p1, p2, atol=1e-5) and torch.allclose(v1, v2, atol=1e-5)


def test_train_step_rollout_and_infer_on_cpu():
    train, val = _tiny_loaders()
    net = build_dime({"kind": "mlp", "hidden": 16, "dropout": 0.1, "append_mask": True}, 784, 10)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    out = train_epoch(net, train, opt, eps=0.05, max_features=3, device=torch.device("cpu"), amp=False, fused=False)
    assert out["pred_loss"] > 0 and out["value_loss"] >= 0
    va = greedy_rollout(net, val, 3, torch.device("cpu"), amp=False, fused=False)
    assert len(va["per_step_acc"]) == 4
    r = infer(net, val, torch.device("cpu"), budget=2)
    assert (r["n_features"] == 2).all()
    # the budget-k prediction must be exactly the greedy rollout's step-k prediction
    va = greedy_rollout(net, val, 3, torch.device("cpu"), amp=False, fused=False)
    for k in (1, 2):
        rk = infer(net, val, torch.device("cpu"), budget=k)
        assert abs(va["per_step_acc"][k] - float((rk["pred"].argmax(1) == rk["y"]).mean())) < 1e-9
    r = infer(net, val, torch.device("cpu"), lam=10.0)  # nothing is worth 10 nats -> zero features
    assert r["n_features"].max() == 0
    info = fit_dime(net, train, val, max_features=2, lr=1e-3, min_lr=1e-6, eps0=0.5, eps_decay=0.2, eps_steps=2, patience=0, max_epochs=3, device=torch.device("cpu"))
    assert 1 <= info["epochs"] <= 3