import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from patchdfs.gdfs.baselines import ConcreteMask2d, StaticMaskLayer2d, cae_selected_inds, center_mask, fit_cae_selector, inds_to_mask, random_mask


def test_static_mask_shapes():
    m = center_mask(8, 3, 4)
    assert m.sum() == 12 and m.shape == (8, 8)
    layer = StaticMaskLayer2d(torch.from_numpy(m), patch_size=4)
    x = torch.ones(2, 3, 32, 32)
    out = layer(x)
    assert out.shape == x.shape and out.sum() == 2 * 3 * 12 * 16
    assert layer.mask_size == 64
    r = random_mask(8, 5, np.random.RandomState(0))
    assert r.sum() == 5
    assert (inds_to_mask(np.arange(3)).sum()) == 3


def test_cae_selector_small_run():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(16, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (16,), generator=g)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=8)
    sel = ConcreteMask2d(width=8, patch_size=4, num_select=3)
    out = sel(x, temp=0.5)
    assert out.shape == x.shape
    model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(4), torch.nn.Flatten(), torch.nn.Linear(48, 10))
    info = fit_cae_selector(model, sel, loader, loader, nepochs=1, patience=0, temp_steps=2, start_temp=1.0, end_temp=0.1, device=torch.device("cpu"), amp_dtype=None)
    assert len(info["history"]) == 2
    inds = cae_selected_inds(sel, 3)
    assert len(inds) == 3
