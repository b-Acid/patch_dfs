"""Whole-step CUDA graph capture for the launch-bound MLP training loops.

The GDFS / DIME training step is a loop of max_features (50) tiny MLP forward/backward passes, i.e.
several thousand small kernel launches per batch. On a shared GPU every launch waits for a time
slice, so these jobs are slowed down several-fold by other processes. Capturing the whole step
(including the 50 backward calls and the Adam step) in one CUDA graph makes it a single launch.

Semantics are the authors' (same losses, same Concrete / eps-greedy sampling); only the random
streams differ. Re-captured whenever the temperature or the learning rate changes.
"""
from __future__ import annotations

import torch


class GraphedStep:
    """Capture fn() (a full training step on static inputs) in a CUDA graph and replay it.

    fn(x, y) must: zero grads with set_to_none=False, run forward/backward, call opt.step(),
    and return a detached loss tensor. No .item() / CPU sync inside fn.
    """

    def __init__(self, fn, x: torch.Tensor, y: torch.Tensor, *, warmup: int = 3):
        self.fn = fn
        self.x = x.clone()
        self.y = y.clone()
        self.graph = None
        self.loss = None
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self.loss = fn(self.x, self.y)
        torch.cuda.current_stream().wait_stream(s)
        self.capture()

    def capture(self):
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss = self.fn(self.x, self.y)
        torch.cuda.synchronize()

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.x.copy_(x, non_blocking=True)
        self.y.copy_(y, non_blocking=True)
        self.graph.replay()
        return self.loss
