"""Run directories and JSON helpers. Heavy artefacts stay on the remote machine."""
from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_default)
    os.replace(tmp, path)


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _default(o: Any) -> Any:
    if isinstance(o, np.integer | np.floating | np.bool_):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"not serializable: {type(o)}")


class RunLogger:
    """Append one JSON line per logged event; write a final summary.json."""

    def __init__(self, run_dir: str, config: dict | None = None):
        self.run_dir = ensure_dir(run_dir)
        self.path = os.path.join(self.run_dir, "log.jsonl")
        self.t0 = time.time()
        if config is not None:
            save_json(os.path.join(self.run_dir, "config.json"), config)

    def log(self, event: str, **fields: Any) -> None:
        rec = {"event": event, "t": round(time.time() - self.t0, 1), **fields}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=_default) + "\n")

    def summary(self, obj: Any) -> str:
        path = os.path.join(self.run_dir, "summary.json")
        save_json(path, obj)
        return path