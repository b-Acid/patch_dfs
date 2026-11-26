"""CIFAR ResNet used by GDFS (experiments/cifar/resnet.py), with the selector head fixed to
output an 8x8 logit grid as described in the paper.

Backbone: 3x3 stem (64) and one BasicBlock per stage (64 @32x32, 128 @16, 256 @8, 512 @4).
Classifier head: GAP + linear. Selector head: one upsampling residual block (512->256, 4->8)
followed by a 1x1 conv, giving 8x8 logits matching the 8x8 patch grid.

Note: the released code stacks two upsampling blocks (16x16 logits) while its mask grid is 8x8,
which cannot run; we follow the paper's 8x8 description instead (recorded in RESULTS)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class UpsamplingBlock(nn.Module):
    """Residual block with 2x transposed-conv upsampling (authors' code)."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(in_planes, planes, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


def CifarResNetBackbone() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Sequential(BasicBlock(64, 64, stride=1)),
        nn.Sequential(BasicBlock(64, 128, stride=2)),
        nn.Sequential(BasicBlock(128, 256, stride=2)),
        nn.Sequential(BasicBlock(256, 512, stride=2)),
    )


def CifarClassifierHead(num_classes: int = 10) -> nn.Sequential:
    return nn.Sequential(nn.AvgPool2d(4), nn.Flatten(), nn.Linear(512, num_classes))


def CifarSelectorHead() -> nn.Sequential:
    """4x4 x 512 backbone output -> 8x8 logits (one per 4x4 patch)."""
    return nn.Sequential(UpsamplingBlock(512, 256), nn.Conv2d(256, 1, kernel_size=1))
