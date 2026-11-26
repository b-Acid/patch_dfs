import numpy as np

from patchdfs.sparse.conv_bound import active_fraction, sparse_bound


def test_active_fraction_extremes():
    full = np.tile(np.arange(64), (2, 1))
    assert active_fraction(full, 32, 1.0).mean() == 1.0
    one = np.array([[0]])  # top-left patch only
    f = active_fraction(one, 32, 1.0)[0]
    # 4x4 patch dilated by 1 pixel -> 6x6 = 36 of 1024 sites
    assert abs(f - 36 / 1024) < 1e-9
    f2 = active_fraction(one, 16, 4.0)[0]  # 2x2 sites at 16x16 dilated by 2 sites each side, clipped at the corner -> 4x4
    assert abs(f2 - 16 / 256) < 1e-9


def test_sparse_bound_random_selections():
    rng = np.random.RandomState(0)
    selections = np.stack([rng.permutation(64) for _ in range(50)])
    res = sparse_bound(selections, [1, 10, 64], n_max=50)
    assert res["per_budget"][64]["gdfs_inference_saving"] == 0.0
    assert res["per_budget"][1]["selector_pass_saving"] > res["per_budget"][10]["selector_pass_saving"] > 0
