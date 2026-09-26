#!/usr/bin/env python3
"""Paired failure-set analysis on the headline [[72,12,6]] test sets.

Decodes the headline_v2 test sets (p = 0.04, 20k shots; p = 0.06, 10k shots)
with every decoder compared in the paper and analyses *which* shots fail:

  bp10        flooding min-sum BP, 10 iterations (the GNN's training decoder)
  bp100       the same, 100 iterations
  gnn_bp10    GNN corrections + bp10, five training seeds
  bposd       serial min-sum BP-OSD-CS-10 (scaling 0.8, 100 iterations)
  gnn_bposd   GNN corrections + bposd, five training seeds

Outputs results/failure_sets/<split>.json with pairwise overlaps (counts,
Jaccard, conditional failure rates), failure categories, failure rate by
error weight, and a weight comparison between each failed estimate and the
true error; plus the per-shot failure vectors.
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
from gnn_pipeline.bp_decoder import MinSumBPDecoder
from gnn_pipeline.decoding_failure import css_failures_from_errors
from gnn_pipeline.gnn_model import TannerGNN

OUT = ROOT / "results" / "failure_sets"
SEEDS = range(5)
SERIAL_MS_OSD = dict(bp_method="ms", ms_scaling_factor=0.8, schedule="serial", max_iter=100,
                     osd_method="osd_cs", osd_order=10)
T = 2  # [[72,12,6]]: every error of weight <= (d-1)/2 = 2 is correctable


def torch_bp(h, s: H.Split, iters, corr=None):
    hx, hz, _, _ = h.code
    dz, dx = MinSumBPDecoder(hx, max_iter=iters, alpha=H.BP_ALPHA), MinSumBPDecoder(hz, max_iter=iters, alpha=H.BP_ALPHA)
    llr = torch.from_numpy(np.outer(s.llr, np.ones(h.n, np.float32)))
    if corr is not None:
        llr = llr + corr
    with torch.no_grad():
        z = dz(torch.from_numpy(s.syn[:, :h.mx]), llr)[1].numpy()
        x = dx(torch.from_numpy(s.syn[:, h.mx:]), llr)[1].numpy()
    return z.round().astype(np.uint8), x.round().astype(np.uint8)


def bposd(h, s: H.Split, rate_z=None, rate_x=None):
    """rate_z/x: per-shot per-qubit priors; default separate-CSS p_Z, p_X."""
    from ldpc import BpOsdDecoder
    hx, hz, _, _ = h.code
    p = float(s.pv[0])
    pz, px = p * H.ETA / (H.ETA + 1), p / (H.ETA + 1)
    dz, dx = BpOsdDecoder(hx, error_rate=pz, **SERIAL_MS_OSD), BpOsdDecoder(hz, error_rate=px, **SERIAL_MS_OSD)
    syn = s.syn.astype(np.uint8)
    z = np.empty((s.shots, h.n), np.uint8); x = np.empty_like(z)
    for i in range(s.shots):
        if rate_z is not None:
            dz.update_channel_probs(np.clip(rate_z[i], 1e-9, 0.5))
            dx.update_channel_probs(np.clip(rate_x[i], 1e-9, 0.5))
        z[i] = dz.decode(syn[i, :h.mx]); x[i] = dx.decode(syn[i, h.mx:])
    return z, x


def load_gnn(seed):
    g = TannerGNN(node_feat_dim=4, hidden_dim=32, num_mp_layers=3, dropout=0.0, use_film=False,
                  use_attention=False, use_residual=True, use_layer_norm=True)
    g.load_state_dict(torch.load(H.OUT_DIR / f"seed{seed}_best.pt", map_location="cpu"))
    return g


def overlap(a, b):
    both = int((a & b).sum()); union = int((a | b).sum())
    return dict(a=int(a.sum()), b=int(b.sum()), both=both, a_only=int((a & ~b).sum()),
                b_only=int((~a & b).sum()), jaccard=both / union if union else float("nan"),
                p_b_given_a=both / int(a.sum()) if a.any() else float("nan"),
                p_a_given_b=both / int(b.sum()) if b.any() else float("nan"))


def describe(rep, z_hat, x_hat, s: H.Split):
    """Failure categories plus estimate-vs-true weight for logical flips."""
    f = rep.failure
    wz_true, wx_true = s.ze.sum(1), s.xe.sum(1)
    w_true = wz_true + wx_true
    w_hat = z_hat.sum(1) + x_hat.sum(1)
    flip = rep.logical_flip
    return dict(
        failures=int(f.sum()),
        syndrome_mismatch=int(rep.syndrome_mismatch.sum()),
        logical_flip=int(flip.sum()),
        # logical flips where the decoder's estimate is no heavier than the true
        # error: a minimum-weight decoder would make the same kind of mistake
        flip_estimate_not_heavier=int((flip & (w_hat <= w_true)).sum()),
        flip_estimate_heavier=int((flip & (w_hat > w_true)).sum()),
        failures_with_weight_le_t=int((f & (w_true <= T)).sum()),
        mean_true_weight_of_failures=float(w_true[f].mean()) if f.any() else float("nan"),
    )


def by_weight(fails, s: H.Split, wmax=10):
    w = (s.ze.sum(1) + s.xe.sum(1)).astype(int)
    out = {}
    for k in range(wmax + 1):
        m = (w == k) if k < wmax else (w >= wmax)
        out[str(k) if k < wmax else f">={wmax}"] = dict(
            shots=int(m.sum()), **{name: int((f & m).sum()) for name, f in fails.items()})
    return out


def run(split):
    h = H.Harness()
    s = h.splits[split]
    hx, hz, lx, lz = h.code
    ze, xe = s.ze.astype(np.uint8), s.xe.astype(np.uint8)

    def rep_of(z_hat, x_hat):
        return css_failures_from_errors(z_hat, x_hat, ze, xe, hx, hz, lx, lz)

    est = {"bp10": torch_bp(h, s, 10), "bp100": torch_bp(h, s, 100), "bposd": bposd(h, s)}
    loader = DataLoader(h.graphs(s), batch_size=512, shuffle=False)
    llr = s.llr[:, None].astype(np.float64)
    for seed in SEEDS:
        corr = h.gnn_corrections(load_gnn(seed), loader)
        est[f"gnn_bp10_s{seed}"] = torch_bp(h, s, 10, corr)
        q = 1.0 / (1.0 + np.exp(llr + corr.numpy().astype(np.float64)))  # same prior both components
        est[f"gnn_bposd_s{seed}"] = bposd(h, s, q, q)
        print(f"{split}: seed {seed} decoded", flush=True)

    reps = {k: rep_of(*v) for k, v in est.items()}
    fails = {k: r.failure for k, r in reps.items()}
    gnn_bp = [f"gnn_bp10_s{i}" for i in SEEDS]
    gnn_osd = [f"gnn_bposd_s{i}" for i in SEEDS]

    # Shots the GNN fixes for bp10, and where they sit relative to BP-OSD.
    fixes = {}
    for k in gnn_bp:
        fixed = fails["bp10"] & ~fails[k]
        broke = ~fails["bp10"] & fails[k]
        fixes[k] = dict(fixed=int(fixed.sum()), broke=int(broke.sum()),
                        fixed_that_bposd_fails=int((fixed & fails["bposd"]).sum()),
                        fixed_that_bp100_fails=int((fixed & fails["bp100"]).sum()),
                        fixed_syndrome_mismatch_in_bp10=int((fixed & reps["bp10"].syndrome_mismatch).sum()))
    # What GNN+OSD changes relative to BP-OSD.
    osd_changes = {}
    for k, kb in zip(gnn_osd, gnn_bp):
        newly = fails[k] & ~fails["bposd"]
        rescued = ~fails[k] & fails["bposd"]
        osd_changes[k] = dict(newly_failed=int(newly.sum()), rescued=int(rescued.sum()),
                              newly_failed_where_gnn_bp10_also_fails=int((newly & fails[kb]).sum()),
                              newly_failed_where_bp10_fails=int((newly & fails["bp10"]).sum()))

    base = ["bp10", "bp100", "bposd"]
    pairs = [("bp10", "bposd"), ("bp10", "bp100"), ("bp100", "bposd")]
    pairs += [(k, "bposd") for k in gnn_bp] + [("bp10", k) for k in gnn_bp] + [("bposd", k) for k in gnn_osd]
    res = dict(
        split=split, shots=s.shots, p=float(s.pv[0]), t_correctable=T,
        decoders=dict(bp10="flooding min-sum, alpha 0.8, 10 it (repo torch)",
                      bp100="flooding min-sum, alpha 0.8, 100 it (repo torch)",
                      bposd="ldpc serial min-sum 0.8, 100 it, OSD-CS-10, priors p_Z, p_X",
                      gnn="headline_v2 checkpoints; GNN+OSD uses the corrected LLR on both components"),
        describe={k: describe(reps[k], *est[k], s) for k in est},
        overlaps={f"{a}|{b}": overlap(fails[a], fails[b]) for a, b in pairs},
        gnn_fixes=fixes, gnn_osd_changes=osd_changes,
        by_weight=by_weight({k: fails[k] for k in base + ["gnn_bp10_s1", "gnn_bposd_s1"]}, s),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{split}.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(OUT / f"{split}_failures.npz", **fails)
    for k in base + ["gnn_bp10_s1", "gnn_bposd_s1"]:
        print(k, res["describe"][k])
    for k, v in res["overlaps"].items():
        if "s0" in k or "_s" not in k:
            print(k, v)
    print("fixes", fixes["gnn_bp10_s1"], "osd", osd_changes["gnn_bposd_s1"])


if __name__ == "__main__":
    torch.manual_seed(0)
    for split in sys.argv[1:] or ["test_p04", "test_p06"]:
        run(split)
