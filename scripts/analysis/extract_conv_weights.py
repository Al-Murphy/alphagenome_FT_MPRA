#!/usr/bin/env python3
"""Save the first conv layer's raw kernel (~(15,4,768)) for the frozen, fly- and
human-fine-tuned encoders, for the weight-space paired first-conv filter comparison.

GPU node (loads the three encoders via encoder_common). Output:
  results/filter_qdist/{pretrained,fly_ft,human_ft}_conv1_w.npy
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from encoder_common import REPO, get_device, load_encoders  # noqa: E402

OUT = REPO / "results/filter_qdist"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"JAX device: {device}\n")
    _base, enc = load_encoders(device, include_human=True)
    for name in ("pretrained", "fly_ft", "human_ft"):
        params = enc[name][0]
        w = None
        for _k, mod in params.items():                 # DNA-reading conv: in=4 channels
            if not isinstance(mod, dict):
                continue
            for _pk, arr in mod.items():
                a = np.asarray(arr)
                if a.ndim == 3 and a.shape[1] == 4 and a.shape[2] == 768:
                    w = a.astype(np.float32)
        if w is None:
            print(f"  {name}: first-conv kernel NOT found"); continue
        np.save(OUT / f"{name}_conv1_w.npy", w)
        print(f"  {name}: conv1 kernel {w.shape} -> {name}_conv1_w.npy")


if __name__ == "__main__":
    main()
