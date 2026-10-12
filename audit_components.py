#!/usr/bin/env python3
"""X- and Z-component failure rates for the headline experiment (Table 4).

The headline failure criterion combines the two CSS components by OR. This
script reports each component separately on the headline test sets:

  Z component: Z errors e_z, decoded on H_x from the X-check syndrome s_x;
               failure if H_x zhat != s_x, or L_x (zhat + e_z) != 0.
  X component: X errors e_x, decoded on H_z from the Z-check syndrome s_z;
               failure if H_z xhat != s_z, or L_z (xhat + e_x) != 0.

Decoders (as in Table 4): 10- and 100-iteration flooding min-sum BP with the
prior LLR l = ln((1-p)/p) on both components; serial min-sum BP-OSD-CS-10 with
priors p_Z, p_X; and GNN-BP with the five headline checkpoints, where the GNN
correction Delta is added to l on both components (the paper's setting) or on
the Z component only (the component the GNN is trained on). GNN + serial BP-OSD
(Tables 5-6) is evaluated the same two ways: the corrected rate
sigma(-(l + Delta)) as the prior of both components, or of the Z component only
(the X component then keeps the prior of l).

Usage:
    python audit_components.py          # writes results/components/<split>.json
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
from audit_failure_sets import bposd, load_gnn
from gnn_pipeline.bp_decoder import MinSumBPDecoder
from gnn_pipeline.decoding_failure import component_failures

OUT = ROOT / "results" / "components"
SEEDS = range(5)


def torch_bp(h, s, iters, corr_z=None, corr_x=None):
    hx, hz, _, _ = h.code
    dz = MinSumBPDecoder(hx, max_iter=iters, alpha=H.BP_ALPHA)
    dx = MinSumBPDecoder(hz, max_iter=iters, alpha=H.BP_ALPHA)
    llr = torch.from_numpy(np.outer(s.llr, np.ones(h.n, np.float32)))
    lz = llr + corr_z if corr_z is not None else llr
    lx = llr + corr_x if corr_x is not None else llr
    with torch.no_grad():
        z = dz(torch.from_numpy(s.syn[:, :h.mx]), lz)[1].numpy()
        x = dx(torch.from_numpy(s.syn[:, h.mx:]), lx)[1].numpy()
    return z.round().astype(np.uint8), x.round().astype(np.uint8)


def components(h, s, z_hat, x_hat):
    hx, hz, lx, lz = h.code
    ze, xe = s.ze.astype(np.uint8), s.xe.astype(np.uint8)
    zp = component_failures(z_hat, (ze @ hx.T) & 1, hx, lx, (ze @ lx.T) & 1)
    xp = component_failures(x_hat, (xe @ hz.T) & 1, hz, lz, (xe @ lz.T) & 1)
    return zp, xp


def mcnemar(base, new):
    n10 = int((base & ~new).sum()); n01 = int((~base & new).sum()); d = n10 + n01
    chi2 = (abs(n10 - n01) - 1) ** 2 / d if d else 0.0
    return dict(n10=n10, n01=n01, p=math.erfc(math.sqrt(chi2 / 2)) if d else 1.0)


def run(split):
    h = H.Harness()
    s = h.splits[split]
    est = {"bp10": torch_bp(h, s, 10), "bp100": torch_bp(h, s, 100), "bposd": bposd(h, s)}
    loader = DataLoader(h.graphs(s), batch_size=512, shuffle=False)
    for seed in SEEDS:
        corr = h.gnn_corrections(load_gnn(seed), loader)
        est[f"gnn_bp10_s{seed}"] = torch_bp(h, s, 10, corr, corr)
        est[f"gnn_bp10_zonly_s{seed}"] = torch_bp(h, s, 10, corr, None)
        # GNN + serial BP-OSD: corrected prior on both components (Tables 5-6) or on Z only
        q = 1.0 / (1.0 + np.exp(s.llr[:, None].astype(np.float64) + corr.numpy().astype(np.float64)))
        q0 = np.broadcast_to(1.0 / (1.0 + np.exp(s.llr[:, None].astype(np.float64))), q.shape)
        est[f"gnn_bposd_s{seed}"] = bposd(h, s, q, q)
        est[f"gnn_bposd_zonly_s{seed}"] = bposd(h, s, q, q0)
    comp = {k: components(h, s, *v) for k, v in est.items()}
    fails = {}
    rec = {}
    for k, (zp, xp) in comp.items():
        z, x = zp.failure, xp.failure
        fails[f"{k}_z"], fails[f"{k}_x"], fails[f"{k}_any"] = z, x, z | x
        rec[k] = dict(z_failures=int(z.sum()), x_failures=int(x.sum()), any_failures=int((z | x).sum()),
                      both=int((z & x).sum()),
                      z_syndrome_mismatch=int(zp.syndrome_mismatch.sum()),
                      x_syndrome_mismatch=int(xp.syndrome_mismatch.sum()),
                      z_ler=float(z.mean()), x_ler=float(x.mean()), any_ler=float((z | x).mean()))
    paired = {}
    for seed in SEEDS:
        for var in ("gnn_bp10", "gnn_bp10_zonly"):
            k = f"{var}_s{seed}"
            paired[f"{k}_vs_bp10"] = {c: mcnemar(fails[f"bp10_{c}"], fails[f"{k}_{c}"]) for c in ("z", "x", "any")}
        for var in ("gnn_bposd", "gnn_bposd_zonly"):
            k = f"{var}_s{seed}"
            paired[f"{k}_vs_bposd"] = {c: mcnemar(fails[f"bposd_{c}"], fails[f"{k}_{c}"]) for c in ("z", "x", "any")}
        paired[f"gnn_bp10_zonly_s{seed}_vs_gnn_bp10_s{seed}"] = {
            c: mcnemar(fails[f"gnn_bp10_s{seed}_{c}"], fails[f"gnn_bp10_zonly_s{seed}_{c}"]) for c in ("x", "any")}
    res = dict(split=split, shots=s.shots, p=float(s.pv[0]),
               note="z: Z errors decoded on H_x; x: X errors decoded on H_z; any: OR (the paper's criterion)",
               decoders=rec, paired=paired)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{split}.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(OUT / f"{split}_failures.npz", **fails)
    for k, v in rec.items():
        print(f"{split} {k:22s} Z {v['z_failures']:5d}  X {v['x_failures']:4d}  any {v['any_failures']:5d}")


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_num_threads(2)
    for split in sys.argv[1:] or ["test_p04", "test_p06"]:
        run(split)
