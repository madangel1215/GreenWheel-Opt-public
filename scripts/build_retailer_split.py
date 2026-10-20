#!/usr/bin/env python3
"""Split per-instance retailer graphs (gen_retailer_surrogate_data.py) into train.pkl / val.pkl.

    python scripts/build_retailer_split.py data/surrogate/retailer --val-seeds 13

Split by instance, never by sample: every graph of one (archetype, rho, seed, month)
goes to the same side. Only pre-2021 instances are accepted; 2021 is the held-out
replay, and a 2021 file here is an error, not something to skip.
"""
import argparse
import pickle
import re
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--val-seeds", type=int, nargs="+", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    train, val, n_tr, n_va = [], [], 0, 0
    for f in sorted(d.glob("*_rho*_s*_*.pkl")):
        m = re.fullmatch(r"([SML])_rho([\d.]+)_s(\d+)_(\d{4})(\d{2})\.pkl", f.name)
        if not m:
            continue
        if int(m.group(4)) >= 2021:
            raise ValueError(f"{f.name}: 2021 is the replay and may not enter training")
        graphs = pickle.load(open(f, "rb"))
        if int(m.group(3)) in a.val_seeds:
            val += graphs
            n_va += 1
        else:
            train += graphs
            n_tr += 1
    pickle.dump(train, open(d / "train.pkl", "wb"))
    pickle.dump(val, open(d / "val.pkl", "wb"))
    print(f"train {n_tr} instances / {len(train)} graphs, val {n_va} instances / {len(val)} graphs")

if __name__ == "__main__":
    main()
