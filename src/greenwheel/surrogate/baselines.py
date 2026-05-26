"""Non-GNN surrogate baselines for fair comparison.

This module provides two non-graph surrogate baselines:
  * classical surrogates (RandomForest, XGBoost), to show the gain comes
    from the GNN rather than merely from having a surrogate;
  * a stronger non-graph neural baseline (DeepSets / set encoding) than
    the flat MLP.

All baselines consume the SAME phase7 dataset (list of PyG ``HeteroData``)
and are scored with the SAME metric schema as the GNN
(:func:`src.greenwheel.surrogate.validation.evaluate_model`), so the numbers
drop straight into the comparison table.

Design notes
------------
* RandomForest / XGBoost need a fixed-length input, so they are trained
  **per scale** (the flat dim is ``8m + 11n + 6 + mn``). This is the same
  limitation that handicaps the MLP - we document it rather than hide it.
* DeepSets is permutation-invariant over the generator and consumer sets and
  is trained **jointly across scales** (like the GNN). Allocation information
  is folded into each node via per-node aggregation of its incident edge
  features - a legitimate set encoding with no pairwise message passing.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kendalltau, spearmanr

# ---------------------------------------------------------------------------
# Shared metrics (mirrors validation.evaluate_model lines 82-105 exactly)
# ---------------------------------------------------------------------------
OBJ_NAMES = ["profit", "surplus", "shortfall"]

def compute_metrics(pred_raw: np.ndarray, true_raw: np.ndarray) -> dict:
    """Per-objective metrics on raw (denormalized) values.

    Args:
        pred_raw, true_raw: arrays of shape (N, 3).

    Returns dict identical in schema to the GNN evaluation.
    """
    metrics: dict = {}
    for i, name in enumerate(OBJ_NAMES):
        y_true = true_raw[:, i]
        y_pred = pred_raw[:, i]

        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        sp, sp_p = spearmanr(y_true, y_pred)
        kt, kt_p = kendalltau(y_true, y_pred)

        metrics[name] = {
            "r2": float(r2),
            "rmse": rmse,
            "mae": mae,
            "spearman": float(sp),
            "spearman_p": float(sp_p),
            "kendall_tau": float(kt),
            "kendall_p": float(kt_p),
            "n_samples": int(len(y_true)),
        }
    return metrics

# ---------------------------------------------------------------------------
# Feature extraction from HeteroData
# ---------------------------------------------------------------------------
def _scale_key(d) -> tuple[int, int]:
    return int(d["gen"].num_nodes), int(d["con"].num_nodes)

def group_by_scale(data_list: list) -> dict[tuple[int, int], list]:
    """Group HeteroData samples by (m, n) so per-scale models can train."""
    groups: dict[tuple[int, int], list] = {}
    for d in data_list:
        groups.setdefault(_scale_key(d), []).append(d)
    return groups

def flatten_sample(d) -> np.ndarray:
    """Flat feature vector for tree baselines:
    [gen(8m), con(11n), global(6), edge_attr(3mn)].

    Gives the trees ALL three edge features (w_ij, price_spread,
    gen_demand_ratio) - the same edge information the GNN and DeepSets receive -
    so the only thing differing across baselines is the model's inductive bias,
    not the input information. (The legacy MLP used only the w_ij column, which
    we note in the paper as a fairness caveat.) Raw features are fine: tree
    ensembles are invariant to monotone feature scaling.
    """
    gen = d["gen"].x.reshape(-1).numpy()
    con = d["con"].x.reshape(-1).numpy()
    g = d.global_feat.reshape(-1).numpy()
    edges = d["gen", "supplies", "con"].edge_attr.reshape(-1).numpy()  # 3mn
    return np.concatenate([gen, con, g, edges]).astype(np.float32)

def stack_flat(data_list: list) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, Y) for a single-scale group. Y is raw targets (N, 3)."""
    X = np.stack([flatten_sample(d) for d in data_list])
    Y = np.stack([d.y.numpy() for d in data_list]).astype(np.float32)
    return X, Y

# ---------------------------------------------------------------------------
# Classical surrogates: RandomForest / XGBoost (per scale)
# ---------------------------------------------------------------------------
def train_classical_per_scale(
    train_data: list,
    test_data: list,
    kind: str = "rf",
    n_estimators: int = 400,
    max_depth: int | None = None,
    n_jobs: int = -1,
    random_state: int = 0,
) -> dict:
    """Train one model per scale, predict each scale's test subset, then pool.

    Args:
        kind: "rf" (RandomForest) or "xgb" (XGBoost).

    Returns:
        {"aggregate": metrics_on_pooled_predictions,
         "per_scale": {"5x10": metrics, ...},
         "params": {...}}
    """
    train_groups = group_by_scale(train_data)
    test_groups = group_by_scale(test_data)

    pooled_pred, pooled_true = [], []
    per_scale: dict[str, dict] = {}

    for scale, tr in sorted(train_groups.items()):
        te = test_groups.get(scale, [])
        if not te:
            continue
        Xtr, Ytr = stack_flat(tr)
        Xte, Yte = stack_flat(te)

        model = _make_classical(kind, n_estimators, max_depth, n_jobs, random_state)
        model.fit(Xtr, Ytr)
        pred = model.predict(Xte).astype(np.float32)

        pooled_pred.append(pred)
        pooled_true.append(Yte)
        m, n = scale
        per_scale[f"{m}x{n}"] = compute_metrics(pred, Yte)

    pred_all = np.concatenate(pooled_pred)
    true_all = np.concatenate(pooled_true)
    agg = compute_metrics(pred_all, true_all)
    agg.pop("predictions", None)

    return {
        "model_type": kind,
        "aggregate": agg,
        "per_scale": per_scale,
        "params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "note": "one model per scale (flat dim depends on m,n)",
        },
    }

def _make_classical(kind, n_estimators, max_depth, n_jobs, random_state):
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if kind == "xgb":
        import xgboost as xgb
        from sklearn.multioutput import MultiOutputRegressor

        base = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth or 6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=n_jobs,
            random_state=random_state,
            tree_method="hist",
        )
        return MultiOutputRegressor(base)
    raise ValueError(f"unknown classical kind: {kind}")

# ---------------------------------------------------------------------------
# DeepSets neural baseline (cross-scale, permutation invariant)
# ---------------------------------------------------------------------------
def _node_edge_aggregates(d) -> tuple[np.ndarray, np.ndarray]:
    """Per-node aggregation of incident edge features (no message passing).

    edge_attr is (m*n, 3) row-major over (gen i, con j). For each generator we
    pool its n outgoing edges; for each consumer its m incoming edges.
    Aggregation = [mean, sum, max] over the 3 edge features -> 9 extra dims.
    Returns (gen_aggr (m,9), con_aggr (n,9)).
    """
    m = int(d["gen"].num_nodes)
    n = int(d["con"].num_nodes)
    ea = d["gen", "supplies", "con"].edge_attr.numpy().reshape(m, n, 3)

    def agg(a, axis):
        return np.concatenate(
            [a.mean(axis=axis), a.sum(axis=axis), a.max(axis=axis)], axis=-1
        )

    gen_aggr = agg(ea, 1)  # (m, 9): pool over consumers
    con_aggr = agg(ea, 0)  # (n, 9): pool over generators
    return gen_aggr.astype(np.float32), con_aggr.astype(np.float32)

def build_deepsets_arrays(data_list: list) -> list[dict]:
    """Per-sample dict of augmented node sets + global + raw targets."""
    out = []
    for d in data_list:
        gen_aggr, con_aggr = _node_edge_aggregates(d)
        gen_x = np.concatenate([d["gen"].x.numpy(), gen_aggr], axis=1)  # (m, 17)
        con_x = np.concatenate([d["con"].x.numpy(), con_aggr], axis=1)  # (n, 20)
        out.append({
            "gen": gen_x.astype(np.float32),
            "con": con_x.astype(np.float32),
            "global": d.global_feat.reshape(-1).numpy().astype(np.float32),
            "y": d.y.numpy().astype(np.float32),
        })
    return out

class DeepSetsNorm:
    """Z-score stats for DeepSets (its own dims differ from the GNN's)."""

    def __init__(self, samples: list[dict]):
        gen = np.concatenate([s["gen"] for s in samples])
        con = np.concatenate([s["con"] for s in samples])
        glob = np.stack([s["global"] for s in samples])
        y = np.stack([s["y"] for s in samples])
        self.gen_m, self.gen_s = gen.mean(0), _safe_std(gen)
        self.con_m, self.con_s = con.mean(0), _safe_std(con)
        self.g_m, self.g_s = glob.mean(0), _safe_std(glob)
        self.y_m, self.y_s = y.mean(0), _safe_std(y)

    def apply(self, s: dict) -> dict:
        return {
            "gen": (s["gen"] - self.gen_m) / self.gen_s,
            "con": (s["con"] - self.con_m) / self.con_s,
            "global": (s["global"] - self.g_m) / self.g_s,
            "y": (s["y"] - self.y_m) / self.y_s,
        }

    def denorm_y(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * self.y_s + self.y_m

def _safe_std(a: np.ndarray) -> np.ndarray:
    s = a.std(0)
    s[s < 1e-6] = 1.0
    return s

class WheelingDeepSets(nn.Module):
    """phi(node) -> {mean,sum} pool -> rho. No graph message passing."""

    def __init__(self, gen_in=17, con_in=20, global_dim=6, hidden=128, dropout=0.1):
        super().__init__()
        self.phi_gen = _mlp(gen_in, hidden, dropout)
        self.phi_con = _mlp(con_in, hidden, dropout)
        self.rho = nn.Sequential(
            nn.Linear(hidden * 4 + global_dim, hidden),
            nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, gen, con, gen_mask, con_mask, glob):
        # gen: (B, M, gen_in), gen_mask: (B, M) 1=real
        hg = self.phi_gen(gen) * gen_mask.unsqueeze(-1)
        hc = self.phi_con(con) * con_mask.unsqueeze(-1)
        gen_n = gen_mask.sum(1, keepdim=True).clamp(min=1)
        con_n = con_mask.sum(1, keepdim=True).clamp(min=1)
        g_pool = torch.cat([hg.sum(1) / gen_n, hg.sum(1)], dim=-1)   # mean, sum
        c_pool = torch.cat([hc.sum(1) / con_n, hc.sum(1)], dim=-1)
        z = torch.cat([g_pool, c_pool, glob], dim=-1)
        return self.rho(z)

def _mlp(in_dim, hidden, dropout):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.ReLU(),
    )

def _collate(batch: list[dict], device):
    """Dynamic padding to the batch's max M, N with masks."""
    M = max(s["gen"].shape[0] for s in batch)
    N = max(s["con"].shape[0] for s in batch)
    B = len(batch)
    gi = batch[0]["gen"].shape[1]
    ci = batch[0]["con"].shape[1]
    gen = np.zeros((B, M, gi), np.float32)
    con = np.zeros((B, N, ci), np.float32)
    gm = np.zeros((B, M), np.float32)
    cm = np.zeros((B, N), np.float32)
    glob = np.zeros((B, batch[0]["global"].shape[0]), np.float32)
    y = np.zeros((B, 3), np.float32)
    for b, s in enumerate(batch):
        m, n = s["gen"].shape[0], s["con"].shape[0]
        gen[b, :m] = s["gen"]; con[b, :n] = s["con"]
        gm[b, :m] = 1.0; cm[b, :n] = 1.0
        glob[b] = s["global"]; y[b] = s["y"]
    t = lambda a: torch.tensor(a, device=device)
    return t(gen), t(con), t(gm), t(cm), t(glob), t(y)

def train_deepsets(
    train_data: list,
    val_data: list,
    test_data: list,
    epochs: int = 300,
    patience: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "auto",
    verbose: bool = True,
) -> dict:
    """Train DeepSets jointly across scales; evaluate on test (raw scale)."""
    dev = _device(device)
    tr = build_deepsets_arrays(train_data)
    va = build_deepsets_arrays(val_data)
    te = build_deepsets_arrays(test_data)

    norm = DeepSetsNorm(tr)
    trn = [norm.apply(s) for s in tr]
    van = [norm.apply(s) for s in va]
    ten = [norm.apply(s) for s in te]

    model = WheelingDeepSets(
        gen_in=tr[0]["gen"].shape[1], con_in=tr[0]["con"].shape[1],
        global_dim=tr[0]["global"].shape[0],
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    loss_fn = nn.MSELoss()

    def batches(data, shuffle):
        idx = np.random.permutation(len(data)) if shuffle else np.arange(len(data))
        for i in range(0, len(data), batch_size):
            yield [data[j] for j in idx[i:i + batch_size]]

    best_val, best_state, bad = float("inf"), None, 0
    history = []
    for ep in range(epochs):
        model.train()
        for b in batches(trn, True):
            gen, con, gm, cm, glob, y = _collate(b, dev)
            opt.zero_grad()
            loss = loss_fn(model(gen, con, gm, cm, glob), y)
            loss.backward(); opt.step()
        sched.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for b in batches(van, False):
                gen, con, gm, cm, glob, y = _collate(b, dev)
                vl += loss_fn(model(gen, con, gm, cm, glob), y).item() * len(b)
        vl /= len(van)
        history.append({"epoch": ep, "val_loss": vl})
        if vl < best_val - 1e-6:
            best_val, bad = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 20 == 0:
            print(f"  [deepsets] epoch {ep:3d} val_loss={vl:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    # Evaluate on test (denormalized)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for b in batches(ten, False):
            gen, con, gm, cm, glob, y = _collate(b, dev)
            preds.append(model(gen, con, gm, cm, glob).cpu().numpy())
            trues.append(y.cpu().numpy())
    pred_norm = np.concatenate(preds)
    true_norm = np.concatenate(trues)
    pred_raw = norm.denorm_y(pred_norm)
    true_raw = norm.denorm_y(true_norm)
    metrics = compute_metrics(pred_raw, true_raw)
    metrics.pop("predictions", None)

    return {
        "model_type": "deepsets",
        "aggregate": metrics,
        "best_val_loss": best_val,
        "epochs_run": len(history),
        "history": history,
        "n_params": sum(p.numel() for p in model.parameters()),
    }

def _device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)
