import torch
from torch.utils.data import DataLoader, TensorDataset

from patchdfs.gdfs import build_gdfs
from patchdfs.gdfs.evaluate import greedy_select
from patchdfs.gdfs.fit import validate
from patchdfs.gdfs.pretrain import pretrain_predictor


def _tiny_loaders():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 784, generator=g)
    y = torch.randint(0, 10, (64,), generator=g)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=16), DataLoader(ds, batch_size=16)


def test_build_gdfs_shapes():
    net = build_gdfs({"kind": "mlp", "hidden": 32, "dropout": 0.3, "append_mask": True}, 784, 10)
    x = torch.randn(3, 784)
    m = torch.zeros(3, 784)
    assert net.select_logits(x, m).shape == (3, 784)
    assert net.predict(x, m).shape == (3, 10)
    assert net.mask_size == 784


def test_greedy_select_no_repeats_and_records_every_step():
    net = build_gdfs({"kind": "mlp", "hidden": 32, "dropout": 0.0, "append_mask": True}, 784, 10)
    net.eval()
    x = torch.randn(5, 784)
    sel, pred = greedy_select(net, x, k_max=7)
    assert sel.shape == (5, 7) and pred.shape == (5, 7, 10)
    for i in range(5):
        assert len(set(sel[i].tolist())) == 7


def test_pretrain_and_validate_run_on_cpu():
    train, val = _tiny_loaders()
    net = build_gdfs({"kind": "mlp", "hidden": 16, "dropout": 0.1, "append_mask": True}, 784, 10)
    hist = pretrain_predictor(
        net.predictor_module(), lambda x, m: net.predict(x, m), net.mask_size, train, val, lr=1e-3, nepochs=2, patience=1, device=torch.device("cpu")
    )
    assert len(hist) == 2 and all(h["val_loss"] > 0 for h in hist)
    soft, hard = validate(net, val, temp=0.5, max_features=3, device=torch.device("cpu"), amp=False)
    assert 0.0 <= soft <= 1.0 and 0.0 <= hard <= 1.0