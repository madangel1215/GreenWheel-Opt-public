#!/usr/bin/env python3
"""Sites and buyers of the three archetype retailers in the T-REC 2025 records.

    python scripts/build_trec_archetypes.py

Reads data/raw/trec/direct_supply_2025.json, the 2025 table of T-REC direct-supply and
wheeling certificate transactions (see README), and writes
data/raw/trec/trec_archetypes_2025.tsv, which src/greenwheel/data/retailer2021.py reads.
One output row per (retailer, kind, key): kind "site" is one generating case (case_id)
with its technology and the MWh it transferred through that retailer in 2025; kind
"buyer" is one buyer with the MWh it bought from that retailer. 移轉量 is MWh (column
header of the T-REC table). Site rows give the site sizes, buyer rows the contract volumes.
"""
import collections
import json
import re
from pathlib import Path

SRC = Path("data/raw/trec/direct_supply_2025.json")
OUT = Path("data/raw/trec/trec_archetypes_2025.tsv")
ARCHETYPES = {"S": "台汽電綠能", "M": "富威電力", "L": "天能綠電"}

def seller(r):
    return [p for p in re.sub(r"<[^>]+>", "|", r["seller_name"]).split("|") if p.strip()][0]

def main():
    data = json.load(open(SRC))["data"]
    out = ["archetype\tretailer\tkind\tkey\ttech\tmwh"]
    for arch, key in ARCHETYPES.items():
        rows = [r for r in data if key in seller(r)]
        names = {seller(r) for r in rows}
        if len(names) != 1:
            raise ValueError(f"{key}: expected one retailer, got {names}")
        sites, techs, buyers = collections.Counter(), {}, collections.Counter()
        for r in rows:
            mwh = float(r["power"].replace(",", ""))
            sites[r["case_id"]] += mwh
            techs[r["case_id"]] = r["energy"]
            buyers[r["buyer_name"]] += mwh
        tot = sum(sites.values())
        mix = collections.Counter()
        for c, v in sites.items():
            mix[techs[c]] += v / tot
        print(f"{arch} {names.pop()}: sites {len(sites)}  buyers {len(buyers)}  {tot:,.0f} MWh  "
              f"mix {dict((k, round(v, 3)) for k, v in mix.items())}")
        out += [f"{arch}\t{key}\tsite\t{c}\t{techs[c]}\t{v}" for c, v in sorted(sites.items())]
        out += [f"{arch}\t{key}\tbuyer\t{b}\t\t{v}" for b, v in buyers.most_common()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
