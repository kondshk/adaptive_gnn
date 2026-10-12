#!/usr/bin/env python3
"""Which BP failures does the GNN repair? (Sec. 5.4, R1 of the review.)

On the headline test sets, flooding min-sum BP (scaling 0.8, messages clamped
to +-20, exactly as gnn_pipeline.bp_decoder.MinSumBPDecoder) is traced for
100 iterations on the Z component, with and without the GNN correction (five
headline checkpoints). The X component is left out: BP has no X-component
failures on these sets (results/components/).

Per shot and decoder, from the hard decisions zhat(t), t = 1..100:
  converged-wrong   H_x zhat(10) = s_x but zhat(10) + e_z is a logical operator
  transient         syndrome not reproduced at t = 10 but reproduced at some t < 10
  slow              not reproduced at any t <= 10, reproduced at some 10 < t <= 100
  oscillating       never reproduced in 100 iterations, and zhat(t) changes in at
                    least 10 of the last 50 iterations
  stuck             never reproduced in 100 iterations, otherwise
A failure of the paper's 10-iteration decoder is any of the first four classes
plus 'stuck' (the decoder returns zhat(10)); per-shot early stopping (return
the first zhat(t), t <= 10, that reproduces the syndrome) is also scored.

Symmetric-stabilizer indicator. For the true Z error e_z, S = 1 if some
Z-stabilizer generator g (a weight-6 row of H_z) overlaps e_z on exactly three
qubits, so that e_z and e_z + g have equal weight and the same syndrome: two
equally likely, logically equivalent errors between which BP has no reason to
prefer either (cf. symmetric stabilizer trapping sets, Raveendran & Vasic,
Quantum 5, 562 (2021)).

Usage:
    python audit_mechanism.py          # writes results/mechanism/<split>.json
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import audit_headline_v2 as H
from audit_failure_sets import load_gnn
from gnn_pipeline.bp_decoder import MinSumBPDecoder

OUT = ROOT / "results" / "mechanism"
SEEDS = range(5)
T_MAX, T_RUN = 100, 10


def trace(dec: MinSumBPDecoder, syndrome: torch.Tensor, llr: torch.Tensor, iters: int) -> np.ndarray:
    """Hard decisions after each iteration, (iters, B, n), same updates as dec.forward."""
    B = syndrome.shape[0]
    s_sign = 1.0 - 2.0 * syndrome
    ctv = torch.zeros(B, dec.num_edges)
    out = np.empty((iters, B, dec.num_vars), np.uint8)
    for it in range(iters):
        var_total = (ctv[:, dec.var_adj] * dec.var_adj_mask.unsqueeze(0)).sum(dim=2)
        vtc = (llr[:, dec.var_idx] + var_total[:, dec.var_idx] - ctv).clamp(-dec.clamp_llr, dec.clamp_llr)
        g = vtc[:, dec.check_adj]
        signs = torch.sign(g)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        mask = dec.check_adj_mask.unsqueeze(0)
        sm = signs * mask + (1.0 - mask)
        total_sign = s_sign.unsqueeze(2) * sm.prod(dim=2, keepdim=True)
        excl_sign = total_sign * torch.where(sm.abs() > 0.5, sm, torch.ones_like(sm))
        a = g.abs() + (1.0 - mask) * 1e6
        topk, _ = a.topk(2, dim=2, largest=False)
        m1, m2 = topk[:, :, 0:1], topk[:, :, 1:2]
        excl_min = torch.where((a - m1).abs() < 1e-9, m2.expand_as(a), m1.expand_as(a))
        new = (dec.alpha * excl_sign * excl_min) * mask
        ctv = torch.zeros_like(ctv)
        ctv.scatter_(1, dec.check_adj.unsqueeze(0).expand(B, -1, -1).reshape(B, -1), new.reshape(B, -1))
        var_total = (ctv[:, dec.var_adj] * dec.var_adj_mask.unsqueeze(0)).sum(dim=2)
        out[it] = ((llr + var_total) < 0).numpy()
    return out


def classify(hd, hx, lx, ze, sx):
    """hd: (T_MAX, B, n). Returns per-shot class labels and early-stopping failure."""
    sat = np.stack([(((h @ hx.T) & 1) == sx).all(1) for h in hd])            # (T_MAX, B)
    flip10 = (((hd[T_RUN - 1] ^ ze) @ lx.T) & 1).any(1)
    sat10 = sat[T_RUN - 1]
    any_le10 = sat[:T_RUN].any(0)
    any_le100 = sat.any(0)
    changes = (hd[-50:][1:] != hd[-50:][:-1]).any(2).sum(0)
    lab = np.full(hd.shape[1], "ok", dtype=object)
    lab[sat10 & flip10] = "converged_wrong"
    lab[~sat10 & any_le10] = "transient"
    lab[~any_le10 & any_le100] = "slow"
    never = ~any_le100
    lab[never & (changes >= 10)] = "oscillating"
    lab[never & (changes < 10)] = "stuck"
    # per-shot early stopping within 10 iterations
    first = np.where(any_le10, sat[:T_RUN].argmax(0), T_RUN - 1)
    est = hd[first, np.arange(hd.shape[1])]
    early_fail = ~any_le10 | (((est ^ ze) @ lx.T) & 1).any(1)
    return lab, early_fail


def summarize(mask, lab, sym):
    n = int(mask.sum())
    d = dict(n=n, symmetric=int((mask & sym).sum()), symmetric_frac=float((mask & sym).sum() / n) if n else float("nan"))
    for c in ("converged_wrong", "transient", "slow", "oscillating", "stuck"):
        d[c] = int((mask & (lab == c)).sum())
    return d


def run(split):
    h = H.Harness()
    s = h.splits[split]
    hx, hz, lx, lz = h.code
    ze = s.ze.astype(np.uint8)
    sx = (ze @ hx.T) & 1
    sym = ((ze.astype(np.int64) @ hz.T.astype(np.int64)) == 3).any(1)     # some weight-6 Z stabilizer half-covered
    dec = MinSumBPDecoder(hx, max_iter=T_MAX, alpha=H.BP_ALPHA)
    syn = torch.from_numpy(s.syn[:, :h.mx])
    llr0 = torch.from_numpy(np.outer(s.llr, np.ones(h.n, np.float32)))
    with torch.no_grad():
        hd_bp = trace(dec, syn, llr0, T_MAX)
    lab_bp, early_bp = classify(hd_bp, hx, lx, ze, sx)
    fail_bp = lab_bp != "ok"
    stored = np.load(ROOT / "results" / "components" / f"{split}_failures.npz")["bp10_z"]
    assert np.array_equal(fail_bp, stored), "trace does not reproduce the stored BP10 Z-component failures"
    res = dict(split=split, shots=s.shots, p=float(s.pv[0]),
               symmetric_all=dict(n=s.shots, symmetric=int(sym.sum()), symmetric_frac=float(sym.mean())),
               bp10=dict(failures=int(fail_bp.sum()), early_stopping_failures=int(early_bp.sum()),
                         classes=summarize(fail_bp, lab_bp, sym),
                         successes=summarize(~fail_bp, lab_bp, sym)),
               gnn={})
    loader = DataLoader(h.graphs(s), batch_size=512, shuffle=False)
    for seed in SEEDS:
        corr = h.gnn_corrections(load_gnn(seed), loader)
        with torch.no_grad():
            hd_g = trace(dec, syn, llr0 + corr, T_MAX)
        lab_g, early_g = classify(hd_g, hx, lx, ze, sx)
        fail_g = lab_g != "ok"
        fixed, broke = fail_bp & ~fail_g, ~fail_bp & fail_g
        res["gnn"][f"s{seed}"] = dict(
            failures=int(fail_g.sum()), early_stopping_failures=int(early_g.sum()),
            fixed=summarize(fixed, lab_bp, sym),              # classes as BP saw them
            not_fixed=summarize(fail_bp & fail_g, lab_bp, sym),
            broke=summarize(broke, lab_g, sym),               # classes as GNN-BP saw them
            fixed_early_stopping=int((early_bp & ~early_g).sum()),
            broke_early_stopping=int((~early_bp & early_g).sum()))
        print(f"{split} seed {seed}: GNN-BP {int(fail_g.sum())} fails (early stop {int(early_g.sum())}); "
              f"fixed {res['gnn'][f's{seed}']['fixed']}", flush=True)
    # failure rates with and without a symmetric configuration, for every decoder of Table 6
    fs = np.load(ROOT / "results" / "failure_sets" / f"{split}_failures.npz")
    res["by_symmetric"] = {
        k: dict(with_symmetric=int((fs[k] & sym).sum()), without_symmetric=int((fs[k] & ~sym).sum()),
                rate_with=float(fs[k][sym].mean()), rate_without=float(fs[k][~sym].mean()))
        for k in ["bp10", "bp100", "bposd"] + [f"gnn_bp10_s{i}" for i in SEEDS] + [f"gnn_bposd_s{i}" for i in SEEDS]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{split}.json").write_text(json.dumps(res, indent=2))
    for k, v in res["by_symmetric"].items():
        print(f"  {k:14s} fail rate with symmetric {v['rate_with']:.3f} ({v['with_symmetric']}), "
              f"without {v['rate_without']:.4f} ({v['without_symmetric']})")
    print(f"{split} BP10: {res['bp10']['failures']} fails, early stopping {res['bp10']['early_stopping_failures']}")
    print("  BP failures:", res["bp10"]["classes"])
    print("  BP successes:", res["bp10"]["successes"])
    print("  all shots symmetric:", res["symmetric_all"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    for split in sys.argv[1:] or ["test_p04", "test_p06"]:
        run(split)
