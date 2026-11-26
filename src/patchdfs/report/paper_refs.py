"""Numbers read off the papers' figures and tables, for comparison only.

GDFS MNIST: Table 3 (exact, single run). GDFS CIFAR-10: Figure 3 (mean of 5 runs; readings
accurate to about +/-0.02). DIME Imagenette: Figure 4 (ViT-S/16; +/-0.02). DIME MNIST: text
("over 90% with ~10/784 features on average, penalised")."""

GDFS_MNIST_GREEDY = {5: 0.695, 10: 0.875, 15: 0.926, 20: 0.950, 25: 0.960, 30: 0.966, 40: 0.973, 50: 0.975}

GDFS_CIFAR = {
    "greedy": {1: 0.27, 2: 0.36, 3: 0.44, 4: 0.48, 5: 0.55, 6: 0.60, 7: 0.64, 8: 0.68, 9: 0.72, 10: 0.74, 15: 0.81, 20: 0.84, 25: 0.85, 30: 0.86},
    "cae": {5: 0.50, 10: 0.72, 15: 0.78, 20: 0.82, 30: 0.84},
    "random": {5: 0.42, 10: 0.63, 15: 0.72, 20: 0.76, 30: 0.80},
    "center": {5: 0.43, 10: 0.63, 15: 0.72, 20: 0.77, 30: 0.80},
}

DIME_IMAGENETTE_VIT = {
    "penalized": {2: 0.57, 4: 0.75, 6: 0.85, 8: 0.91, 10: 0.94, 12: 0.96, 15: 0.97},
    "budget": {2: 0.37, 4: 0.50, 6: 0.60, 8: 0.68, 10: 0.75, 12: 0.80, 15: 0.84},
    "cae": {2.5: 0.37, 5: 0.45, 7.5: 0.50, 10: 0.55, 12.5: 0.60},
}

DIME_MNIST_PENALIZED = {10: 0.90}  # "~10 features -> over 90%" (text)
