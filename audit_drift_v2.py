#!/usr/bin/env python3
"""Rebuild of the drift-adaptation experiment (Table 7 / Fig. F3) and the
single-shot rate-estimator check, on [[72,12,6]].

Noise. Each data qubit i has log-rate offset x_i(t) following a discrete
Ornstein-Uhlenbeck (AR(1)) process with stationary standard deviation SIGMA
and per-shot innovation standard deviation `vol`:
    x_i(t+1) = rho x_i(t) + vol * xi,   rho = sqrt(1 - vol^2 / SIGMA^2),
so vol = 0 is a static heterogeneous field. The rate is
p_i(t) = P * exp(x_i(t) - SIGMA^2/2) (mean P), capped at 0.5, and each shot
draws Z flips with p_i * eta/(eta+1) and X flips with p_i/(eta+1).

Decoders. Serial min-sum BP-OSD-CS-10 (scaling 0.8, 100 iterations) on both
CSS components; they differ only in the per-qubit prior used at time t:
  oracle  true p_i(t)
  stale   p_i(0), a calibration taken at the start of the run
  mean    uniform P
  empir   per-qubit frequency of the decoder's own Z-error estimates over
          the previous W shots, shrunk toward P with K pseudo-shots
  gnn     a TannerGNN that regresses x_i(t) from the violated-check
          frequencies of the previous W syndromes
Shots t < W are warm-up and not scored.

The single-shot check trains the same estimator with W = 1 on a field that is
redrawn every shot and compares it with the uniform mean prior.

Usage:
    python audit_drift_v2.py train          # train both estimators
    python audit_drift_v2.py drift          # Table 7 / Fig. F3
    python audit_drift_v2.py single         # single-shot estimator check
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from codes.code_registry import get_code_params
from codes.codes_q import create_bivariate_bicycle_codes
from gnn_pipeline.decoding_failure import css_failures_from_errors
from gnn_pipeline.gnn_model import TannerGNN
from gnn_pipeline.tanner_graph import build_tanner_graph

OUT = ROOT / "results" / "drift_v2"
P, ETA, SIGMA, W, K = 0.04, 20.0, 1.0, 32, 16
VOLS = (0.0, 0.1, 0.35, 0.5)
SHOTS = 6000
BPOSD = dict(bp_method="ms", ms_scaling_factor=0.8, schedule="serial", max_iter=100,
             osd_method="osd_cs", osd_order=10)

css, _, _ = create_bivariate_bicycle_codes(**get_code_params("72_12_6"))
HX, HZ, LX, LZ = (np.asarray(getattr(css, a), dtype=np.uint8) for a in ("hx", "hz", "lx", "lz"))
N, MX, MZ = HX.shape[1], HX.shape[0], HZ.shape[0]
NT, EI, ET = (torch.from_numpy(a) for a in build_tanner_graph(HX, HZ))


# ------------------------------------------------------------------ noise
def field(vol, T, rng):
    """(T, N) log-rate offsets x_i(t)."""
    x = np.empty((T, N))
    x[0] = SIGMA * rng.standard_normal(N)
    rho = math.sqrt(max(1.0 - vol ** 2 / SIGMA ** 2, 0.0))
    for t in range(1, T):
        x[t] = rho * x[t - 1] + vol * rng.standard_normal(N)
    return x


def rates(x):
    return np.minimum(P * np.exp(x - SIGMA ** 2 / 2), 0.5)


def errors(p, rng):
    z = (rng.random(p.shape) < p * ETA / (ETA + 1)).astype(np.uint8)
    x = (rng.random(p.shape) < p / (ETA + 1)).astype(np.uint8)
    return z, x


# ------------------------------------------------------------------ estimator
def graph(fx, fz):
    """Node features: variable [0,1,0,0]; X-check [f,0,1,0]; Z-check [f,0,0,1]."""
    f = torch.zeros(N + MX + MZ, 4)
    f[:N, 1] = 1.0
    f[N:N + MX, 0] = torch.from_numpy(fx.astype(np.float32)); f[N:N + MX, 2] = 1.0
    f[N + MX:, 0] = torch.from_numpy(fz.astype(np.float32)); f[N + MX:, 3] = 1.0
    return f


def new_model():
    return TannerGNN(node_feat_dim=4, hidden_dim=32, num_mp_layers=3, dropout=0.0, use_film=False,
                     use_attention=False, use_residual=True, use_layer_norm=True)


def window_features(z, x):
    """Violated-check frequencies over a window of shots: z, x are (w, N)."""
    return ((z @ HX.T) % 2).mean(0), ((x @ HZ.T) % 2).mean(0)


def train_estimator(w, n_train, seed, epochs=30):
    """Regress the current x_i(t) from the previous w syndromes."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    data = []
    for k in range(n_train):
        if w == 1:
            xs = SIGMA * rng.standard_normal((1, N))              # fresh field, one shot
            target = xs[0]
            z, xx = errors(rates(xs), rng)
        else:
            vol = VOLS[k % len(VOLS)]
            xs = field(vol, w + 1, rng)                          # w past shots + current
            target = xs[-1]
            z, xx = errors(rates(xs[:-1]), rng)
        fx, fz = window_features(z, xx)
        data.append(Data(x=graph(fx, fz), edge_index=EI, edge_type=ET, node_type=NT,
                         y=torch.from_numpy(target.astype(np.float32))))
    n_val = n_train // 10
    tr = DataLoader(data[n_val:], batch_size=64, shuffle=True)
    va = DataLoader(data[:n_val], batch_size=512)
    model = new_model()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best, best_state, hist = math.inf, None, []
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        for b in tr:
            opt.zero_grad()
            loss = torch.mean((model(b).view(-1) - b.y) ** 2)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * b.num_graphs
        model.eval()
        with torch.no_grad():
            vl = sum(torch.sum((model(b).view(-1) - b.y) ** 2).item() for b in va) / (n_val * N)
        hist.append(dict(epoch=ep, train_mse=tot / (n_train - n_val), val_mse=vl))
        print(f"W={w} ep {ep:2d} train {hist[-1]['train_mse']:.4f} val {vl:.4f}", flush=True)
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, hist


def predict(model, feats):
    model.eval()
    loader = DataLoader([Data(x=graph(fx, fz), edge_index=EI, edge_type=ET, node_type=NT)
                         for fx, fz in feats], batch_size=512)
    with torch.no_grad():
        return torch.cat([model(b).view(b.num_graphs, N) for b in loader]).numpy().astype(np.float64)


# ------------------------------------------------------------------ decoding
class Decoder:
    def __init__(self):
        from ldpc import BpOsdDecoder
        self.dz = BpOsdDecoder(HX, error_rate=P * ETA / (ETA + 1), **BPOSD)
        self.dx = BpOsdDecoder(HZ, error_rate=P / (ETA + 1), **BPOSD)

    def __call__(self, sz, sx, p):
        p = np.clip(p, 1e-6, 0.5)
        self.dz.update_channel_probs(p * ETA / (ETA + 1))
        self.dx.update_channel_probs(p / (ETA + 1))
        return self.dz.decode(sz), self.dx.decode(sx)


def mcnemar(a, b):
    n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); d = n10 + n01
    chi2 = (abs(n10 - n01) - 1) ** 2 / d if d else 0.0
    return dict(n10=n10, n01=n01, p=math.erfc(math.sqrt(chi2 / 2)) if d else 1.0)


def wilson(k, n, z=1.96):
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [max(c - h, 0.0), c + h]


def run_drift():
    model = new_model(); model.load_state_dict(torch.load(OUT / "estimator_W32.pt"))
    out = dict(p=P, eta=ETA, sigma=SIGMA, W=W, K=K, shots=SHOTS, decoder=BPOSD, vols={})
    arrays = {}
    for vi, vol in enumerate(VOLS):
        t0 = time.time()
        rng = np.random.default_rng(9000 + vi)
        T = SHOTS + W
        xs = field(vol, T, rng)
        pt = rates(xs)
        z, x = errors(pt, rng)
        sz, sx = (z @ HX.T) % 2, (x @ HZ.T) % 2
        # GNN estimate for shot t uses syndromes t-W .. t-1
        feats = [window_features(z[t - W:t], x[t - W:t]) for t in range(W, T)]
        p_gnn = rates(predict(model, feats))
        decs = {k: Decoder() for k in ("oracle", "stale", "mean", "empir", "gnn")}
        est = {k: (np.zeros((T, N), np.uint8), np.zeros((T, N), np.uint8)) for k in decs}
        for t in range(T):
            pri = {"oracle": pt[t], "stale": pt[0], "mean": np.full(N, P)}
            if t >= W:
                c = est["empir"][0][t - W:t].sum(0)
                pz_hat = (c + K * P * ETA / (ETA + 1)) / (W + K)
                pri["empir"] = pz_hat * (ETA + 1) / ETA
                pri["gnn"] = p_gnn[t - W]
            else:
                pri["empir"] = pri["gnn"] = np.full(N, P)
            for k, d in decs.items():
                est[k][0][t], est[k][1][t] = d(sz[t], sx[t], pri[k])
        f = {k: css_failures_from_errors(e[0][W:], e[1][W:], z[W:], x[W:], HX, HZ, LX, LZ).failure
             for k, e in est.items()}
        rec = {k: dict(failures=int(v.sum()), ler=float(v.mean()), ci=wilson(int(v.sum()), len(v)))
               for k, v in f.items()}
        rec["mcnemar_gnn_vs_stale"] = mcnemar(f["stale"], f["gnn"])      # n10: stale fails only
        rec["mcnemar_gnn_vs_empir"] = mcnemar(f["empir"], f["gnn"])      # n10: empir fails only
        rec["mcnemar_empir_vs_stale"] = mcnemar(f["stale"], f["empir"])
        rec["gnn_rate_rmse_log"] = float(np.sqrt(np.mean((np.log(p_gnn) - np.log(pt[W:])) ** 2)))
        out["vols"][str(vol)] = rec
        arrays.update({f"vol{vol}_{k}": v for k, v in f.items()})
        print(f"vol={vol}: " + " ".join(f"{k}={rec[k]['ler']:.4f}" for k in f)
              + f" | gnn vs stale {rec['mcnemar_gnn_vs_stale']} | gnn vs empir {rec['mcnemar_gnn_vs_empir']}"
              + f" ({time.time() - t0:.0f}s)", flush=True)
    (OUT / "drift.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(OUT / "drift_failures.npz", **arrays)


def run_single(shots=20000):
    model = new_model(); model.load_state_dict(torch.load(OUT / "estimator_W1.pt"))
    rng = np.random.default_rng(9100)
    xs = SIGMA * rng.standard_normal((shots, N))
    pt = rates(xs)
    z, x = errors(pt, rng)
    sz, sx = (z @ HX.T) % 2, (x @ HZ.T) % 2
    p_gnn = rates(predict(model, [window_features(z[t:t + 1], x[t:t + 1]) for t in range(shots)]))
    decs = {k: Decoder() for k in ("mean", "gnn", "oracle")}
    est = {k: (np.zeros_like(z), np.zeros_like(x)) for k in decs}
    for t in range(shots):
        pri = {"mean": np.full(N, P), "gnn": p_gnn[t], "oracle": pt[t]}
        for k, d in decs.items():
            est[k][0][t], est[k][1][t] = d(sz[t], sx[t], pri[k])
    f = {k: css_failures_from_errors(e[0], e[1], z, x, HX, HZ, LX, LZ).failure for k, e in est.items()}
    out = dict(p=P, sigma=SIGMA, shots=shots,
               **{k: dict(failures=int(v.sum()), ler=float(v.mean()), ci=wilson(int(v.sum()), shots))
                  for k, v in f.items()},
               mcnemar_gnn_vs_mean=mcnemar(f["mean"], f["gnn"]))   # n10: mean fails only
    (OUT / "single_shot.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(OUT / "single_shot_failures.npz", **f)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1]
    if what == "train":
        for w, n, seed in ((W, 20000, 1), (1, 20000, 2)):
            m, h = train_estimator(w, n, seed)
            torch.save(m.state_dict(), OUT / f"estimator_W{w}.pt")
            (OUT / f"estimator_W{w}_history.json").write_text(json.dumps(h, indent=2))
    elif what == "drift":
        run_drift()
    elif what == "single":
        run_single()
