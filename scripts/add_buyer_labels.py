#!/usr/bin/env python3
"""Per-buyer fulfilment labels for the retailer surrogate data.

    python scripts/add_buyer_labels.py data/surrogate/retailer data/surrogate/retailer_buyer --files L_rho0.1_s11_202001.pkl ...

Trained only on the graph-level worst fulfilment, the model learned "the largest buyer is
the bottleneck" and ranked fronts backwards where it is not. Each graph gets
con.y_buyer: every buyer's fulfilment averaged over the scenarios, so each buyer's score
can be supervised directly. W is read back from edge feature 0 (features.py, row-major
gen x con), the instance is rebuilt as gen_retailer_surrogate_data.py built it, and the
recomputed indicators must equal the stored y, or the file is refused.
"""
import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.greenwheel.data import retailer2021 as rt  # noqa: E402
from src.greenwheel.surrogate import retailer_graph as rg  # noqa: E402

def label(src: Path, dst: Path):
    m_ = re.fullmatch(r"([SML])_rho([\d.]+)_s(\d+)_(\d{4})(\d{2})\.pkl", src.name)
    arch, rho, seed, year, month = m_.group(1), float(m_.group(2)), int(m_.group(3)), int(m_.group(4)), int(m_.group(5))
    mp = rt.RetailerMonth(rt.Retailer(arch, rho, seed), month, year=year)
    graphs = pickle.load(open(src, "rb"))
    worst_err = 0.0
    for g in graphs:
        m, n = g["gen"].x.shape[0], g["con"].x.shape[0]
        if (m, n) != (mp.m, mp.n):
            raise ValueError(f"{src.name}: graph is {m}x{n}, rebuilt instance is {mp.m}x{mp.n}")
        W = g["gen", "supplies", "con"].edge_attr[:, 0].double().numpy().reshape(m, n)
        r = mp.evaluate(W)
        y = rg.indicators(r)
        err = np.abs(y - g.y.double().numpy()) / np.maximum(np.abs(g.y.double().numpy()), 1e-9)
        worst_err = max(worst_err, float(err.max()))
        g["con"].y_buyer = torch.tensor(r["fulfil"].mean(axis=0), dtype=torch.float32)
    # y was stored as float32, so agreement is to float32 precision, not exact.
    if worst_err > 1e-5:
        raise ValueError(f"{src.name}: recomputed indicators differ from stored y (max rel err {worst_err:.2e})")
    with open(dst / src.name, "wb") as f:
        pickle.dump(graphs, f)
    print(f"{src.name}: {len(graphs)} graphs, max rel err {worst_err:.1e}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--files", nargs="+", required=True)
    a = ap.parse_args()
    dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in a.files:
        if not (dst / f).exists():
            label(Path(a.src) / f, dst)

if __name__ == "__main__":
    main()
