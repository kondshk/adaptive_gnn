#!/usr/bin/env python3
"""Drift experiment (Sec. 5.6, Table 8, Fig. 7) on [[72,12,6]], fully specified.

Noise. Each data qubit i has a log-rate offset x_i(t) that follows a discrete
Ornstein-Uhlenbeck (AR(1)) process, independently across qubits:
    x_i(0)   ~ N(0, SIGMA^2)                      (stationary start)
    x_i(t+1) = rho * x_i(t) + vol * xi_i(t),     xi ~ N(0, 1),
    rho      = sqrt(1 - vol^2 / SIGMA^2),
so the stationary standard deviation is SIGMA for every volatility, `vol` is
the per-shot innovation standard deviation of the log-rate, and vol = 0 is a
static heterogeneous field. The correlation time is -1/ln(rho) shots
(about 200, 15 and 7 shots for vol = 0.1, 0.35, 0.5). The rate is
    p_i(t) = min(0.5, P * exp(x_i(t) - SIGMA^2/2)),
whose mean over the stationary distribution is P. Each shot draws a Z flip on
qubit i with probability p_i(t)*eta/(eta+1) and, independently, an X flip with
probability p_i(t)/(eta+1).

Decoders. Serial min-sum BP-OSD-CS-10 (scaling 0.8, 100 iterations, ldpc) on
each CSS component separately, with per-qubit priors p_z = q*eta/(eta+1) and
p_x = q/(eta+1), where q is the rate estimate used at shot t:
  oracle  q_i = p_i(t), the true current rates
  stale   q_i = p_i(0), the true rates at the start of the stream (a perfect
          calibration taken once and never updated)
  mean    q_i = P, the uniform mean rate
  empir   per-qubit frequency of this decoder's own Z-error estimates over the
          previous W shots, shrunk toward P with K pseudo-shots:
          q_i = (sum_{s=t-W}^{t-1} zhat_i(s) + K*P*eta/(eta+1)) / (W + K) * (eta+1)/eta
  gnn     a TannerGNN rate estimator (five training seeds): it regresses x_i(t)
          from the violated-check frequencies of the previous W syndromes and
          sets q_i = min(0.5, P * exp(xhat_i - SIGMA^2/2)). It sees syndromes
          only, never decoder outputs or true rates.
Every decoder sees the current syndrome. Shots t < W are warm-up (empir and gnn
use q = P there) and are not scored.

GNN estimator training (per seed s = 1..5). 20,000 windows; window k uses
vol = VOLS[k mod 4], a stationary field of length W + 1, errors drawn from the
first W rates, input = violated-check frequencies over those W shots, target =
x_i at the next step. Model: TannerGNN as in Table 2 (hidden 32, 3 layers,
GRU + residual + layer norm, no FiLM), variable-node input [0, 1, 0, 0],
check-node input [frequency, type one-hot]. MSE loss, AdamW (lr 2e-3,
weight decay 1e-4), batch 64, gradient clipping 1.0, 30 epochs; the first 10%
of windows are a validation set and the epoch with the lowest validation MSE
is kept. Evaluation streams use different random seeds.

Evaluation. For each vol, STREAMS independent streams of W + SCORED shots.
Per-shot failures use gnn_pipeline.decoding_failure. Reported: LER with Wilson
95% intervals, paired McNemar tests on the pooled shots, and a sign test over
streams (shots within one stream share the same rate field, so the pooled
McNemar p-values are optimistic; the stream-level sign test is the robust one).

Usage:
    python audit_drift.py train      # five estimator checkpoints -> results/drift/
    python audit_drift.py eval       # Table 8 / Fig. 7 -> results/drift/drift.json
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time
from multiprocessing import Pool

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

OUT = ROOT / "results" / "drift"
P, ETA, SIGMA, W, K = 0.04, 20.0, 1.0, 32, 16
VOLS = (0.0, 0.1, 0.35, 0.5)
STREAMS, SCORED = 20, 2500
SEEDS = (1, 2, 3, 4, 5)
N_TRAIN, EPOCHS = 20000, 30
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


def train_estimator(seed):
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    data = []
    for k in range(N_TRAIN):
        xs = field(VOLS[k % len(VOLS)], W + 1, rng)
        z, xx = errors(rates(xs[:-1]), rng)
        fx, fz = window_features(z, xx)
        data.append(Data(x=graph(fx, fz), edge_index=EI, edge_type=ET, node_type=NT,
                         y=torch.from_numpy(xs[-1].astype(np.float32))))
    n_val = N_TRAIN // 10
    tr = DataLoader(data[n_val:], batch_size=64, shuffle=True)
    va = DataLoader(data[:n_val], batch_size=512)
    model = new_model()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best, best_ep, best_state, hist = math.inf, 0, None, []
    for ep in range(1, EPOCHS + 1):
        model.train(); tot = 0.0
        for b in tr:
            opt.zero_grad()
            loss = torch.mean((model(b).view(-1) - b.y) ** 2)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * b.num_graphs
        model.eval()
        with torch.no_grad():
            vl = sum(torch.sum((model(b).view(-1) - b.y) ** 2).item() for b in va) / (n_val * N)
        hist.append(dict(epoch=ep, train_mse=tot / (N_TRAIN - n_val), val_mse=vl))
        print(f"seed {seed} ep {ep:2d} train {hist[-1]['train_mse']:.4f} val {vl:.4f}", flush=True)
        if vl < best:
            best, best_ep, best_state = vl, ep, {k: v.clone() for k, v in model.state_dict().items()}
    torch.save(best_state, OUT / f"estimator_seed{seed}.pt")
    (OUT / f"estimator_seed{seed}_history.json").write_text(
        json.dumps(dict(seed=seed, best_epoch=best_ep, best_val_mse=best, history=hist), indent=2))
    return seed, best_ep, best


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

    def __call__(self, sz, sx, q):
        q = np.clip(q, 1e-6, 0.5)
        self.dz.update_channel_probs(q * ETA / (ETA + 1))
        self.dx.update_channel_probs(q / (ETA + 1))
        return self.dz.decode(sz), self.dx.decode(sx)


def stream(vi, si):
    """Sample one stream; return its rates, errors, syndromes and GNN window features."""
    rng = np.random.default_rng(9000 + 100 * vi + si)
    T = W + SCORED
    pt = rates(field(VOLS[vi], T, rng))
    z, x = errors(pt, rng)
    return pt, z, x


def decode_stream(args):
    vi, si, q_gnn = args          # q_gnn: (n_seeds, SCORED, N) GNN rate estimates for shots W..
    pt, z, x = stream(vi, si)
    T = W + SCORED
    sz, sx = (z @ HX.T) % 2, (x @ HZ.T) % 2
    names = ["oracle", "stale", "mean", "empir"] + [f"gnn_s{s}" for s in SEEDS]
    decs = {k: Decoder() for k in names}
    est = {k: (np.zeros((T, N), np.uint8), np.zeros((T, N), np.uint8)) for k in names}
    for t in range(T):
        pri = {"oracle": pt[t], "stale": pt[0], "mean": np.full(N, P)}
        if t >= W:
            c = est["empir"][0][t - W:t].sum(0)
            pri["empir"] = (c + K * P * ETA / (ETA + 1)) / (W + K) * (ETA + 1) / ETA
            for j, s in enumerate(SEEDS):
                pri[f"gnn_s{s}"] = q_gnn[j, t - W]
        else:
            pri["empir"] = np.full(N, P)
            for s in SEEDS:
                pri[f"gnn_s{s}"] = np.full(N, P)
        for k, d in decs.items():
            est[k][0][t], est[k][1][t] = d(sz[t], sx[t], pri[k])
    return vi, si, {k: css_failures_from_errors(e[0][W:], e[1][W:], z[W:], x[W:], HX, HZ, LX, LZ).failure
                    for k, e in est.items()}


def mcnemar(a, b):
    """a, b: failure vectors of the reference and the compared decoder; n10 = only a fails."""
    n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); d = n10 + n01
    chi2 = (abs(n10 - n01) - 1) ** 2 / d if d else 0.0
    return dict(n10=n10, n01=n01, p=math.erfc(math.sqrt(chi2 / 2)) if d else 1.0)


def sign_test(wins, losses):
    """Two-sided exact binomial sign test, ties dropped."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilson(k, n, z=1.96):
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [max(c - h, 0.0), c + h]


def run_eval(workers=4):
    torch.set_num_threads(workers)
    models = []
    for s in SEEDS:
        m = new_model(); m.load_state_dict(torch.load(OUT / f"estimator_seed{s}.pt")); models.append(m)
    jobs = []
    for vi in range(len(VOLS)):
        for si in range(STREAMS):
            pt, z, x = stream(vi, si)
            feats = [window_features(z[t - W:t], x[t - W:t]) for t in range(W, W + SCORED)]
            q = np.stack([rates(predict(m, feats)) for m in models])
            jobs.append((vi, si, q.astype(np.float32)))
        print(f"vol={VOLS[vi]}: GNN estimates done", flush=True)
    torch.set_num_threads(1)
    with Pool(workers) as pool:
        res = pool.map(decode_stream, jobs)
    gnn_keys = [f"gnn_s{s}" for s in SEEDS]
    out = dict(p=P, eta=ETA, sigma=SIGMA, W=W, K=K, vols=list(VOLS), streams=STREAMS,
               scored_shots_per_stream=SCORED, decoder=BPOSD, seeds=list(SEEDS), results={})
    arrays = {}
    for vi, vol in enumerate(VOLS):
        per = sorted((r for r in res if r[0] == vi), key=lambda r: r[1])
        f = {k: np.concatenate([r[2][k] for r in per]) for k in per[0][2]}
        cnt = {k: [int(r[2][k].sum()) for r in per] for k in per[0][2]}
        rec = {k: dict(failures=int(v.sum()), shots=len(v), ler=float(v.mean()), ci=wilson(int(v.sum()), len(v)),
                       per_stream_failures=cnt[k]) for k, v in f.items()}
        gl = np.array([rec[k]["ler"] for k in gnn_keys])
        rec["gnn_seed_mean_ler"] = float(gl.mean()); rec["gnn_seed_min_ler"] = float(gl.min())
        rec["gnn_seed_max_ler"] = float(gl.max())
        comp = {}
        for ref in ("stale", "empir", "mean"):
            for k in gnn_keys:
                comp[f"{k}_vs_{ref}"] = mcnemar(f[ref], f[k])
        comp["empir_vs_stale"] = mcnemar(f["stale"], f["empir"])
        comp["empir_vs_mean"] = mcnemar(f["mean"], f["empir"])
        comp["oracle_vs_mean"] = mcnemar(f["mean"], f["oracle"])
        comp["oracle_vs_stale"] = mcnemar(f["stale"], f["oracle"])
        rec["mcnemar"] = comp      # n10: only the reference (second name) fails
        # stream-level sign tests: seed-averaged GNN failures vs each reference
        g = np.mean([cnt[k] for k in gnn_keys], axis=0)
        st = {}
        for ref in ("stale", "empir", "mean"):
            wins = int((g < np.array(cnt[ref])).sum()); losses = int((g > np.array(cnt[ref])).sum())
            st[f"gnn_vs_{ref}"] = dict(gnn_better=wins, gnn_worse=losses, p=sign_test(wins, losses))
        e, s_ = np.array(cnt["empir"]), np.array(cnt["stale"])
        st["empir_vs_stale"] = dict(empir_better=int((e < s_).sum()), empir_worse=int((e > s_).sum()),
                                    p=sign_test(int((e < s_).sum()), int((e > s_).sum())))
        rec["sign_tests"] = st
        out["results"][str(vol)] = rec
        arrays.update({f"vol{vol}_{k}": v for k, v in f.items()})
        print(f"vol={vol}: " + " ".join(f"{k}={rec[k]['ler']:.4f}" for k in ("oracle", "stale", "mean", "empir"))
              + f" gnn={rec['gnn_seed_mean_ler']:.4f} [{rec['gnn_seed_min_ler']:.4f},{rec['gnn_seed_max_ler']:.4f}]"
              + f" | sign {st}", flush=True)
    (OUT / "drift.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(OUT / "drift_failures.npz", **arrays)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1]
    if what == "train":
        with Pool(min(4, len(SEEDS))) as pool:
            for seed, ep, mse in pool.imap_unordered(train_estimator, SEEDS):
                print(f"seed {seed}: best epoch {ep}, val MSE {mse:.4f}", flush=True)
    elif what == "eval":
        run_eval()
