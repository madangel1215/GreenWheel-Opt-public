"""GNN inputs and targets for the real-scenario retailer month.

The published architecture and feature extractors are reused unchanged: a retailer month
is turned into a WheelingInstance whose profiles are the scenario means, which is all
that is known when Pmi is decided. Station groups are the generator nodes, buyers the
consumer nodes. The three targets are the multi-scenario indicators and occupy
the slots the training code calls profit / surplus / shortfall, all to be maximised:

    y = [mean profit, mean worst contract fulfilment, CVaR20 of profit]
"""

import numpy as np

from src.greenwheel.config import ConsumerSite, GeneratorPlant, MarketParams, WheelingInstance
from src.greenwheel.data import replay2021 as rp
from src.greenwheel.data import retailer2021 as rt
from src.greenwheel.surrogate.graph import instance_to_heterodata

CVAR_SHARE = 0.2

def indicators(r):
    """The three indicators from RetailerMonth.evaluate output."""
    p = r["profit"]
    tail = np.sort(p)[: max(1, int(np.ceil(CVAR_SHARE * len(p))))]
    return np.array([p.mean(), r["worst"].mean(), tail.mean()])

def mean_instance(mp: rt.RetailerMonth) -> WheelingInstance:
    ret = mp.ret
    G = mp.R.mean(axis=0) * mp.scale            # (m, T) kWh per 15 min, month-scaled like the targets
    D = mp.D.mean(axis=0) * mp.scale
    # T-REC gives no kW, so capacity is the peak 15-min output in kW; it only feeds the
    # log-capacity and capacity-factor features.
    gens = [GeneratorPlant(id=i, capacity_kw=float(G[i].max() * 4), technology=ret.group_tech[i],
                           ppa_price=rp.PPA_PRICE, generation=G[i]) for i in range(mp.m)]
    supply = G.sum()
    cons = [ConsumerSite(id=j, total_monthly_kwh=float(D[j].sum()),
                         contract_kwh=float(ret.contract_share[j] * supply), re100_target=0.0,
                         retail_price=float(ret.price[j]), demand=D[j], consumer_type="industrial")
            for j in range(mp.n)]
    market = MarketParams(surplus_price=0.0, wheeling_fee=rt.FEE, re100_penalty_weight=0.0)
    return WheelingInstance(generators=gens, consumers=cons, market=market, n_intervals=G.shape[1])

def to_graph(inst: WheelingInstance, W: np.ndarray, y: np.ndarray | None = None):
    return instance_to_heterodata(inst, W, targets=y, temporal=True)
