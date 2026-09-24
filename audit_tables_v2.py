#!/usr/bin/env python3
"""Regenerate the paper's classical-decoder tables with stored paired outcomes.

Every experiment draws fresh code-capacity data with its own RNG seed, decodes
all compared decoders on the *same* shots, and saves the per-shot failure
vectors next to a JSON summary in results/tables_v2/, so every table entry
(and every McNemar count) can be recomputed from raw outcomes.

Noise: independent Z and X flips per data qubit with p_Z = p*eta/(eta+1),
p_X = p/(eta+1), eta = 20 (no explicit Y). Decoding is separate-CSS: Z errors
are decoded on H_X with prior p_Z and X errors on H_Z with prior p_X. Failures
use gnn_pipeline.decoding_failure (syndrome mismatch or logical flip, OR over
the two components).

Subcommands:
  bposd_configs   Table 3  (288 code, p=0.04, BP / BP-OSD configurations)
  decomposition   Fig. F1  (288 code, p=0.04, flooding -> serial -> +OSD)
  gnn_vs_bposd    Table 5 / Fig. F4 (72 code, headline_v2 GNN + BP-OSD)
  oracle          Table 6 / Fig. F2 (per-qubit log-normal rates, 3 codes)
  invariance      Sec. invariance: uniform-prior and eta bit-identity checks
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from codes.code_registry import get_code_params
from codes.codes_q import create_bivariate_bicycle_codes
from gnn_pipeline.decoding_failure import css_failures_from_errors

OUT = ROOT / "results" / "tables_v2"
ETA = 20.0

# Decoder configurations (ldpc 2.x keyword arguments).
FLOODING_MS = dict(bp_method="ms", ms_scaling_factor=0.8, schedule="parallel", max_iter=100)
SERIAL_MS = dict(bp_method="ms", ms_scaling_factor=0.8, schedule="serial", max_iter=100)
PARALLEL_PS30 = dict(bp_method="ps", schedule="parallel", max_iter=30)
OSD_CS10 = dict(osd_method="osd_cs", osd_order=10)


# ------------------------------------------------------------------ helpers
_CODES = {}


def code(name):
    if name not in _CODES:
        css, _, _ = create_bivariate_bicycle_codes(**get_code_params(name))
        _CODES[name] = tuple(np.asarray(getattr(css, a), dtype=np.uint8) for a in ("hx", "hz", "lx", "lz"))
    return _CODES[name]


def rates(p):
    return p * ETA / (ETA + 1), p / (ETA + 1)


def sample(name, p, shots, seed, qubit_rates=None):
    """Code-capacity errors; qubit_rates (shots, n) overrides the uniform p."""
    hx, hz, _, _ = code(name)
    n = hx.shape[1]
    rng = np.random.default_rng(seed)
    pq = np.full((shots, n), p) if qubit_rates is None else qubit_rates
    pz, px = rates(pq)
    z = (rng.random((shots, n)) < pz).astype(np.uint8)
    x = (rng.random((shots, n)) < px).astype(np.uint8)
    return z, x


def decode_all(name, z, x, cfg, osd, priors=None, p=None, same_prior=False):
    """Decode both CSS components with ldpc.

    priors (shots, n) are per-shot total rates, split into p_Z / p_X. With
    same_prior=True the given rates are used unchanged for both components.
    """
    from ldpc import BpDecoder, BpOsdDecoder
    hx, hz, lx, lz = code(name)
    kw = dict(cfg, **(OSD_CS10 if osd else {}))
    Dec = BpOsdDecoder if osd else BpDecoder
    pz0, px0 = rates(p if p is not None else 0.01)
    dz = Dec(hx, error_rate=pz0, **kw)
    dx = Dec(hz, error_rate=px0, **kw)
    sz, sx = (z @ hx.T) % 2, (x @ hz.T) % 2
    zh = np.empty_like(z); xh = np.empty_like(x)
    for i in range(len(z)):
        if priors is not None:
            pz, px = (priors[i], priors[i]) if same_prior else rates(priors[i])
            dz.update_channel_probs(np.clip(pz, 1e-9, 0.5))
            dx.update_channel_probs(np.clip(px, 1e-9, 0.5))
        zh[i] = dz.decode(sz[i]); xh[i] = dx.decode(sx[i])
    return css_failures_from_errors(zh, xh, z, x, hx, hz, lx, lz)


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 2
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [max(c - h, 0.0), c + h]


def mcnemar(a, b):
    """a, b: failure vectors. n10 = a fails only, n01 = b fails only."""
    n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); d = n10 + n01
    chi2 = (abs(n10 - n01) - 1) ** 2 / d if d else 0.0
    return dict(n10=n10, n01=n01, chi2=chi2, p=math.erfc(math.sqrt(chi2 / 2)) if d else 1.0)


def entry(f):
    k = int(f.sum())
    return dict(failures=k, shots=len(f), ler=k / len(f), ci=wilson(k, len(f)))


def chunked(fn, jobs, workers):
    with Pool(workers) as pool:
        return pool.map(fn, jobs)


def save(name, summary, arrays):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(OUT / f"{name}_failures.npz", **arrays)
    print(json.dumps(summary, indent=2)[:3000])


# ------------------------------------------------------ Table 3 and Fig. F1
CONFIGS = {
    "bp_parallel_ps30": (PARALLEL_PS30, False),
    "bp_serial_ms100": (SERIAL_MS, False),
    "bposd_parallel_ps30": (PARALLEL_PS30, True),
    "bposd_serial_ms100": (SERIAL_MS, True),
    "bp_flooding_ms100": (FLOODING_MS, False),
}


def _config_chunk(args):
    code_name, p, shots, seed, keys = args
    z, x = sample(code_name, p, shots, seed)
    return {k: decode_all(code_name, z, x, *CONFIGS[k], p=p).failure for k in keys}


def run_configs(tag, keys, shots, seed, workers, code_name="288_12_18", p=0.04):
    per = -(-shots // workers)
    parts = chunked(_config_chunk, [(code_name, p, per, seed + j, keys) for j in range(workers)], workers)
    f = {k: np.concatenate([r[k] for r in parts])[:shots] for k in keys}
    summ = dict(code=code_name, p=p, eta=ETA, shots=shots, data_seed=seed, chunks=workers,
                configs={k: dict(CONFIGS[k][0], osd=OSD_CS10 if CONFIGS[k][1] else None) for k in keys},
                results={k: entry(v) for k, v in f.items()},
                mcnemar={f"{a}_vs_{b}": mcnemar(f[a], f[b]) for i, a in enumerate(keys) for b in keys[i + 1:]})
    save(tag, summ, f)


# ------------------------------------------------------ Table 5 / Fig. F4
def _gnn_chunk(args):
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from gnn_pipeline.gnn_model import TannerGNN
    from gnn_pipeline.tanner_graph import build_tanner_graph
    torch.set_num_threads(1)
    p, shots, seed, ckpts = args
    hx, hz, lx, lz = code("72_12_6")
    n, mx, mz = hx.shape[1], hx.shape[0], hz.shape[0]
    z, x = sample("72_12_6", p, shots, seed)
    sz, sx = (z @ hx.T) % 2, (x @ hz.T) % 2
    # Headline convention: one prior LLR log((1-p)/p) for both components.
    llr = math.log((1 - p) / p)
    out = {"bposd": decode_all("72_12_6", z, x, SERIAL_MS, True, priors=np.full(z.shape, p), p=p, same_prior=True).failure}
    nt, ei, et = (torch.from_numpy(a) for a in build_tanner_graph(hx, hz))
    graphs = []
    for i in range(shots):
        f = torch.zeros(n + mx + mz, 4)
        f[:n, 0] = llr; f[:n, 1] = 1.0
        f[n:n + mx, 0] = torch.from_numpy(sx[i].astype(np.float32)); f[n:n + mx, 2] = 1.0
        f[n + mx:, 0] = torch.from_numpy(sz[i].astype(np.float32)); f[n + mx:, 3] = 1.0
        graphs.append(Data(x=f, edge_index=ei, edge_type=et, node_type=nt))
    loader = DataLoader(graphs, batch_size=512, shuffle=False)
    for ck in ckpts:
        gnn = TannerGNN(node_feat_dim=4, hidden_dim=32, num_mp_layers=3, dropout=0.0, use_film=False,
                        use_attention=False, use_residual=True, use_layer_norm=True)
        gnn.load_state_dict(torch.load(ck, map_location="cpu")); gnn.eval()
        with torch.no_grad():
            delta = torch.cat([gnn(b).view(b.num_graphs, n) for b in loader]).numpy().astype(np.float64)
        # Corrected per-qubit rate from the corrected LLR, used for both
        # components as in the headline evaluation.
        q = 1.0 / (1.0 + np.exp(llr + delta))
        out[pathlib.Path(ck).stem] = decode_all("72_12_6", z, x, SERIAL_MS, True, priors=q, p=p,
                                                same_prior=True).failure
    return out


def run_gnn_vs_bposd(ps, shots, workers):
    ckpts = sorted(str(c) for c in (ROOT / "results" / "headline_v2").glob("seed*_best.pt"))
    summ = dict(code="72_12_6", eta=ETA, shots=shots, bposd=dict(SERIAL_MS, osd=OSD_CS10),
                prior="uniform LLR log((1-p)/p) on both components (headline convention)",
                checkpoints=[pathlib.Path(c).name for c in ckpts], points={})
    arrays = {}
    per = -(-shots // workers)
    for i, p in enumerate(ps):
        t0 = time.time()
        parts = chunked(_gnn_chunk, [(p, per, 4000 + 100 * i + j, ckpts) for j in range(workers)], workers)
        f = {k: np.concatenate([r[k] for r in parts])[:shots] for k in parts[0]}
        seeds = [k for k in f if k != "bposd"]
        rec = dict(data_seeds=[4000 + 100 * i + j for j in range(workers)], bposd=entry(f["bposd"]),
                   gnn={k: dict(entry(f[k]), mcnemar_bposd_vs_gnn=mcnemar(f["bposd"], f[k])) for k in seeds})
        summ["points"][str(p)] = rec
        arrays.update({f"p{p}_{k}": v for k, v in f.items()})
        g = [rec["gnn"][k]["failures"] for k in seeds]
        print(f"p={p}: BP-OSD {rec['bposd']['failures']} GNN+OSD {g} ({time.time() - t0:.0f}s)", flush=True)
    save("gnn_vs_bposd", summ, arrays)


# ------------------------------------------------------ Table 6 / Fig. F2
def _oracle_chunk(args):
    code_name, p, sigma, shots, seed, anchor = args
    n = code(code_name)[0].shape[1]
    rng = np.random.default_rng(seed + 7_777_777)
    median = p if anchor == "median" else p * math.exp(-sigma ** 2 / 2)
    pq = np.minimum(median * np.exp(sigma * rng.standard_normal((shots, n))), 0.5)  # fresh per shot
    z, x = sample(code_name, p, shots, seed, qubit_rates=pq)
    mean_p = median * math.exp(sigma ** 2 / 2)
    return dict(mean=decode_all(code_name, z, x, SERIAL_MS, True, p=mean_p).failure,
                median=decode_all(code_name, z, x, SERIAL_MS, True, p=median).failure,
                oracle=decode_all(code_name, z, x, SERIAL_MS, True, priors=pq, p=p).failure)


def run_oracle(points, sigmas, shots, workers, anchor="median"):
    """anchor='median': p is the median per-qubit rate (mean grows with sigma);
    anchor='mean': p is the mean rate, held fixed across sigma."""
    summ = dict(eta=ETA, shots=shots, decoder=dict(SERIAL_MS, osd=OSD_CS10), anchor=anchor,
                noise="p_q = p_median * exp(sigma * N(0,1)), fresh per shot, capped at 0.5; "
                      "p_median = p (anchor median) or p * exp(-sigma^2/2) (anchor mean)",
                mean_prior="uniform p_median * exp(sigma^2/2)", cells={})
    arrays = {}
    per = -(-shots // workers)
    for ci, (code_name, p) in enumerate(points):
        for si, s in enumerate(sigmas):
            t0 = time.time()
            seeds = [5_000_000 + 100_000 * ci + 1000 * si + j for j in range(workers)]
            parts = chunked(_oracle_chunk, [(code_name, p, s, per, sd, anchor) for sd in seeds], workers)
            f = {k: np.concatenate([r[k] for r in parts])[:shots] for k in ("mean", "median", "oracle")}
            rec = dict(code=code_name, p=p, anchor=anchor, sigma=s, data_seeds=seeds,
                       **{k: entry(v) for k, v in f.items()},
                       mcnemar_mean_vs_oracle=mcnemar(f["mean"], f["oracle"]),
                       mcnemar_median_vs_oracle=mcnemar(f["median"], f["oracle"]))
            for ref in ("mean", "median"):
                kr, ko = rec[ref]["failures"], rec["oracle"]["failures"]
                rec[f"gap_vs_{ref}"] = 1 - ko / kr if kr else float("nan")
            key = f"{code_name}_p{p}_s{s}"
            summ["cells"][key] = rec
            arrays.update({f"{key}_{k}": v for k, v in f.items()})
            print(f"{key}: mean {rec['mean']['failures']} median {rec['median']['failures']} "
                  f"oracle {rec['oracle']['failures']} gap {rec['gap_vs_mean']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    save("oracle" if anchor == "median" else f"oracle_{anchor}", summ, arrays)


# ------------------------------------------------------ invariance checks
def run_invariance():
    import torch
    from gnn_pipeline.bp_decoder import MinSumBPDecoder
    from ldpc import BpDecoder
    res = {}
    for name, p0, p1, shots in (("288_12_18", 0.015, 0.0286, 2000), ("72_12_6", 0.04, 0.02, 2000)):
        hx, hz, lx, lz = code(name)
        z, x = sample(name, 0.04, shots, 6000 + len(res))
        s = (z @ hx.T) % 2
        rec = {}
        # Repo's torch min-sum (flooding, alpha 0.8, 10 iterations, messages
        # clamped to +-20, which breaks exact scaling invariance), scalar LLR
        d = MinSumBPDecoder(hx, max_iter=10, alpha=0.8)
        outs = []
        for p in (p0, p1):
            llr = torch.full((shots, hx.shape[1]), math.log((1 - p) / p))
            outs.append(d(torch.from_numpy(s.astype(np.float32)), llr)[1].numpy())
        rec["torch_minsum_flooding10"] = int((outs[0] == outs[1]).all(1).sum())
        for label, kw in (("ldpc_ms_serial100", SERIAL_MS), ("ldpc_ms_flooding100", FLOODING_MS),
                          ("ldpc_ps_serial100", dict(SERIAL_MS, bp_method="ps"))):
            outs = []
            for p in (p0, p1):
                dd = BpDecoder(hx, error_rate=p, **kw)
                outs.append(np.stack([dd.decode(v) for v in s]))
            rec[label] = int((outs[0] == outs[1]).all(1).sum())
        res[f"{name}_p{p0}_vs_p{p1}"] = dict(shots=shots, identical_shots=rec)
    # eta sweep on 72 at p = 0.03: priors change with eta, data fixed
    hx, hz, lx, lz = code("72_12_6")
    z, x = sample("72_12_6", 0.03, 5000, 6100)
    sweep = {}
    for eta in (1, 2, 5, 10, 20, 50, 100):
        pz, px = 0.03 * eta / (eta + 1), 0.03 / (eta + 1)
        dz = BpDecoder(hx, error_rate=pz, **SERIAL_MS); dx = BpDecoder(hz, error_rate=px, **SERIAL_MS)
        zh = np.stack([dz.decode(v) for v in (z @ hx.T) % 2]); xh = np.stack([dx.decode(v) for v in (x @ hz.T) % 2])
        sweep[str(eta)] = dict(failures=int(css_failures_from_errors(zh, xh, z, x, hx, hz, lx, lz).failure.sum()),
                               z_sha1=hashlib.sha1(zh.tobytes()).hexdigest()[:12],
                               x_sha1=hashlib.sha1(xh.tobytes()).hexdigest()[:12])
    res["eta_sweep_72_p0.03_true_eta20"] = dict(shots=5000, decoder="ldpc_ms_serial100", by_assumed_eta=sweep)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "invariance.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["bposd_configs", "decomposition", "gnn_vs_bposd", "oracle", "invariance"])
    ap.add_argument("--shots", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--anchor", choices=["median", "mean"], default="median",
                    help="oracle: whether p is the median or the mean per-qubit rate")
    a = ap.parse_args()
    if a.what == "bposd_configs":
        run_configs("bposd_configs", ["bp_parallel_ps30", "bp_serial_ms100", "bposd_parallel_ps30",
                                      "bposd_serial_ms100"], a.shots or 100_000, 7000, a.workers)
    elif a.what == "decomposition":
        run_configs("decomposition", ["bp_flooding_ms100", "bp_serial_ms100", "bposd_serial_ms100"],
                    a.shots or 100_000, 7100, a.workers)
    elif a.what == "gnn_vs_bposd":
        run_gnn_vs_bposd([0.02, 0.03, 0.04, 0.05], a.shots or 20_000, a.workers)
    elif a.what == "oracle":
        run_oracle([("72_12_6", 0.04), ("144_12_12", 0.06), ("288_12_18", 0.07),
                    ("144_12_12", 0.04), ("288_12_18", 0.04)],
                   [0.25, 0.5, 1.0], a.shots or 50_000, a.workers, a.anchor)
    else:
        run_invariance()
