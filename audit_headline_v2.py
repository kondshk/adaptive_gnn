#!/usr/bin/env python3
"""Headline re-run: GNN+flooding BP vs flooding BP on [[72,12,6]], leakage-free.

Differences from audit_confound1.py (the script behind the original 15.4%):
  * train / validation / test are generated with distinct RNG seeds, so no
    test shot shares a random stream with a training shot;
  * the checkpoint is selected on a separate validation set, and the test set
    is scored once, on that checkpoint;
  * several training seeds (model init + minibatch order) are run;
  * failures use gnn_pipeline.decoding_failure (syndrome mismatch counts).

Everything else (model, loss, optimiser, LLRs, BP settings) is unchanged so the
numbers are directly comparable with the original headline.

Usage:
    python audit_headline_v2.py --make_data
    python audit_headline_v2.py --baselines
    python audit_headline_v2.py --seed 0 [--seed 1 ...]
    python audit_headline_v2.py --summary
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from codes.code_registry import get_code_params
from codes.codes_q import create_bivariate_bicycle_codes
from gnn_pipeline.bp_decoder import MinSumBPDecoder
from gnn_pipeline.decoding_failure import css_failures_from_errors
from gnn_pipeline.generate_codecap import generate_code_capacity_data
from gnn_pipeline.gnn_model import TannerGNN
from gnn_pipeline.loss_functions import focal_loss
from gnn_pipeline.tanner_graph import build_tanner_graph

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "headline_v2"
OUT_DIR = ROOT / "results" / "headline_v2"

ETA = 20.0
# name -> (p_base, drift_amp, shots, data seed). Training files use the same
# sine drift as the original big72_train_p02/p03 (amp 0.015, period 500), so
# training p spans [0.005, 0.045]; p = 0.04 is inside that range, p = 0.06 is not.
SPLITS = {
    "train_p02": (0.02, 0.015, 4000, 1001),
    "train_p03": (0.03, 0.015, 4000, 1002),
    "val_p04":   (0.04, 0.0,   4000, 2001),
    "test_p04":  (0.04, 0.0,  20000, 3001),
    "test_p06":  (0.06, 0.0,  10000, 3002),
}
TEST_SPLITS = ("test_p04", "test_p06")

EPOCHS = 20
BATCH = 64
LR = 2e-3
WEIGHT_DECAY = 1e-4
BP_ITERS = 10
BP_ALPHA = 0.8


def build_code():
    css, _, _ = create_bivariate_bicycle_codes(**get_code_params("72_12_6"))
    return tuple(np.asarray(getattr(css, a), dtype=np.uint8) for a in ("hx", "hz", "lx", "lz"))


def make_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    hx, hz, lx, lz = build_code()
    for name, (p, amp, shots, seed) in SPLITS.items():
        res = generate_code_capacity_data(
            hx, hz, lx, lz, shots=shots, p_base=p, eta=ETA,
            drift_model="sine" if amp > 0 else "none", drift_amp=amp,
            drift_period=500, seed=seed,
        )
        meta = dict(code="72_12_6", p=p, eta=ETA, drift_amp=amp, drift_period=500,
                    shots=shots, seed=seed, generator="gnn_pipeline.generate_codecap")
        np.savez_compressed(DATA_DIR / f"{name}.npz", **res, hx=hx, hz=hz, lx=lx, lz=lz,
                            meta=json.dumps(meta).encode())
        pv = res["p_values"]
        print(f"{name}: {shots} shots, seed {seed}, p in [{pv.min():.3f}, {pv.max():.3f}]")
    overlap_report()


def _patterns(d):
    return {r.tobytes() for r in np.concatenate([d["z_errors"], d["x_errors"]], 1).astype(np.uint8)}


def overlap_report():
    """Exact error-pattern collisions between training and evaluation shots.

    With independent seeds, collisions come only from low-weight patterns that
    i.i.d. noise produces repeatedly; they are not leakage.
    """
    train = set()
    for name in ("train_p02", "train_p03"):
        train |= _patterns(np.load(DATA_DIR / f"{name}.npz"))
    for name in ("val_p04",) + TEST_SPLITS:
        d = np.load(DATA_DIR / f"{name}.npz")
        pats = np.concatenate([d["z_errors"], d["x_errors"]], 1).astype(np.uint8)
        hit = np.array([r.tobytes() in train for r in pats])
        w = pats.sum(1)
        by_w = {int(k): int(hit[w == k].sum()) for k in range(0, 5)}
        print(f"  {name}: {hit.sum()}/{len(hit)} patterns seen in training; "
              f"by weight {by_w}; weight>=5: {int(hit[w >= 5].sum())}")


class Split:
    def __init__(self, name, code):
        hx, hz, lx, lz = code
        raw = np.load(DATA_DIR / f"{name}.npz")
        self.mx = hx.shape[0]
        self.syn = raw["syndromes"].astype(np.float32)
        self.pv = raw["p_values"].astype(np.float32)
        self.llr = np.log((1 - self.pv) / self.pv).astype(np.float32)
        self.ze = raw["z_errors"].astype(np.float32)
        self.xe = raw["x_errors"].astype(np.float32)
        self.shots = len(self.pv)


class Harness:
    def __init__(self):
        self.code = build_code()
        hx, hz, lx, lz = self.code
        self.n, self.mx, self.mz = hx.shape[1], hx.shape[0], hz.shape[0]
        nt, ei, et = build_tanner_graph(hx, hz)
        self.topo = (torch.from_numpy(nt), torch.from_numpy(ei), torch.from_numpy(et))
        self.dec_z = MinSumBPDecoder(hx, max_iter=BP_ITERS, alpha=BP_ALPHA)
        self.dec_x = MinSumBPDecoder(hz, max_iter=BP_ITERS, alpha=BP_ALPHA)
        self.splits = {k: Split(k, self.code) for k in SPLITS}

    def graphs(self, s: Split):
        nt, ei, et = self.topo
        n, mx = self.n, self.mx
        out = []
        for i in range(s.shots):
            x = torch.zeros(n + mx + self.mz, 4)
            x[:n, 0] = float(s.llr[i]); x[:n, 1] = 1.0
            x[n:n + mx, 0] = torch.from_numpy(s.syn[i, :mx]); x[n:n + mx, 2] = 1.0
            x[n + mx:, 0] = torch.from_numpy(s.syn[i, mx:]); x[n + mx:, 3] = 1.0
            out.append(Data(x=x, edge_index=ei, edge_type=et, node_type=nt,
                            y=torch.from_numpy(s.ze[i])))
        return out

    def failures(self, s: Split, z_hat, x_hat):
        hx, hz, lx, lz = self.code
        return css_failures_from_errors(z_hat, x_hat, s.ze, s.xe, hx, hz, lx, lz)

    def bp(self, s: Split, corr=None, iters=BP_ITERS):
        hx, hz, _, _ = self.code
        dz, dx = ((self.dec_z, self.dec_x) if iters == BP_ITERS else
                  (MinSumBPDecoder(hx, max_iter=iters, alpha=BP_ALPHA),
                   MinSumBPDecoder(hz, max_iter=iters, alpha=BP_ALPHA)))
        llr = torch.from_numpy(np.outer(s.llr, np.ones(self.n, np.float32)))
        if corr is not None:
            llr = llr + corr
        with torch.no_grad():
            _, z_hat, _ = dz(torch.from_numpy(s.syn[:, :self.mx]), llr)
            _, x_hat, _ = dx(torch.from_numpy(s.syn[:, self.mx:]), llr)
        return self.failures(s, z_hat.numpy(), x_hat.numpy())

    def gnn_corrections(self, gnn, loader):
        gnn.eval()
        with torch.no_grad():
            return torch.cat([gnn(b).view(b.num_graphs, self.n) for b in loader])


def mcnemar(base, new):
    n10 = int((base & ~new).sum()); n01 = int((~base & new).sum()); d = n10 + n01
    chi2 = (abs(n10 - n01) - 1) ** 2 / d if d else 0.0
    return dict(n10=n10, n01=n01, chi2=chi2, p=math.erfc(math.sqrt(chi2 / 2)) if d else 1.0)


def wilson(k, n, z=1.96):
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def summarise(base, new):
    kb, kn, n = int(base.sum()), int(new.sum()), len(base)
    return dict(shots=n, bp_failures=kb, gnn_failures=kn, bp_ler=kb / n, gnn_ler=kn / n,
                bp_ci=wilson(kb, n), gnn_ci=wilson(kn, n),
                reduction=1 - kn / kb if kb else float("nan"),
                mcnemar=mcnemar(base, new))


def run_baselines():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    h = Harness()
    hx, hz, _, _ = h.code
    out = {}
    for name in ("val_p04",) + TEST_SPLITS:
        s = h.splits[name]
        rec = {}
        for it in (10, 100):
            r = h.bp(s, iters=it)
            rec[f"bp{it}"] = dict(r.counts(), ler=float(r.failure.mean()),
                                  ci=wilson(int(r.failure.sum()), s.shots))
        from ldpc import BpOsdDecoder
        p = float(s.pv.mean()); pz = p * ETA / (ETA + 1); px = p / (ETA + 1)
        kw = dict(max_iter=100, bp_method="ms", ms_scaling_factor=0.625,
                  schedule="parallel", osd_method="osd_cs", osd_order=10)
        dz = BpOsdDecoder(hx, error_rate=pz, **kw); dx = BpOsdDecoder(hz, error_rate=px, **kw)
        syn = s.syn.astype(np.uint8)
        z_hat = np.stack([dz.decode(v) for v in syn[:, :h.mx]])
        x_hat = np.stack([dx.decode(v) for v in syn[:, h.mx:]])
        r = h.failures(s, z_hat, x_hat)
        rec["bposd_cs10"] = dict(r.counts(), ler=float(r.failure.mean()),
                                 ci=wilson(int(r.failure.sum()), s.shots))
        out[name] = rec
        print(name, json.dumps({k: (v["failures"], round(v["ler"], 4)) for k, v in rec.items()}))
    (OUT_DIR / "baselines.json").write_text(json.dumps(out, indent=2))


def run_seed(seed: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    h = Harness()
    sp = h.splits
    train = h.graphs(sp["train_p02"]) + h.graphs(sp["train_p03"])
    loaders = {k: DataLoader(h.graphs(sp[k]), batch_size=512, shuffle=False)
               for k in ("val_p04",) + TEST_SPLITS}
    train_loader = DataLoader(train, batch_size=BATCH, shuffle=True)
    base = {k: h.bp(sp[k]) for k in loaders}

    gnn = TannerGNN(node_feat_dim=4, hidden_dim=32, num_mp_layers=3, dropout=0.0,
                    use_film=False, use_attention=False, use_residual=True, use_layer_norm=True)
    opt = AdamW(gnn.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def score(k):
        return h.bp(sp[k], corr=h.gnn_corrections(gnn, loaders[k]))

    history, best = [], (math.inf, 0, None)
    for ep in range(1, EPOCHS + 1):
        t0 = time.time(); gnn.train(); tot = nb = 0
        for b in train_loader:
            opt.zero_grad()
            d = gnn(b).view(-1)
            prior = b.x[b.node_type == 0, 0]
            loss = focal_loss(torch.sigmoid(-(prior + d)), b.y.view(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        v = int(score("val_p04").failure.sum())
        history.append(dict(epoch=ep, loss=tot / nb, val_failures=v))
        print(f"seed {seed} ep {ep:2d} loss {tot / nb:.5f} val {v} ({time.time() - t0:.0f}s)", flush=True)
        if v < best[0]:
            best = (v, ep, {k: t.clone() for k, t in gnn.state_dict().items()})
    last = {k: score(k).failure for k in TEST_SPLITS}

    gnn.load_state_dict(best[2])
    res = dict(seed=seed, best_epoch=best[1], history=history, splits={})
    for k in ("val_p04",) + TEST_SPLITS:
        new = score(k)
        rec = summarise(base[k].failure, new.failure)
        rec["gnn_syndrome_mismatch"] = int(new.syndrome_mismatch.sum())
        rec["bp_syndrome_mismatch"] = int(base[k].syndrome_mismatch.sum())
        if k in last:
            rec["last_epoch_gnn_failures"] = int(last[k].sum())
        res["splits"][k] = rec
        np.save(OUT_DIR / f"seed{seed}_{k}_gnn_fail.npy", new.failure)
        np.save(OUT_DIR / f"{k}_bp_fail.npy", base[k].failure)
    torch.save(best[2], OUT_DIR / f"seed{seed}_best.pt")
    (OUT_DIR / f"seed{seed}.json").write_text(json.dumps(res, indent=2))
    t = res["splits"]["test_p04"]
    print(f"seed {seed} DONE best ep {best[1]}: test_p04 BP {t['bp_failures']} GNN {t['gnn_failures']} "
          f"red {t['reduction']:.3f} n10/n01 {t['mcnemar']['n10']}/{t['mcnemar']['n01']}", flush=True)


def summary():
    runs = [json.loads(p.read_text()) for p in sorted(OUT_DIR.glob("seed*.json"))]
    out = dict(seeds=[r["seed"] for r in runs], best_epochs=[r["best_epoch"] for r in runs])
    for k in ("val_p04",) + TEST_SPLITS:
        recs = [r["splits"][k] for r in runs]
        red = np.array([x["reduction"] for x in recs])
        out[k] = dict(
            shots=recs[0]["shots"], bp_failures=recs[0]["bp_failures"], bp_ler=recs[0]["bp_ler"],
            gnn_failures=[x["gnn_failures"] for x in recs],
            gnn_ler_mean=float(np.mean([x["gnn_ler"] for x in recs])),
            gnn_ler_std=float(np.std([x["gnn_ler"] for x in recs], ddof=1)) if len(recs) > 1 else 0.0,
            reduction_mean=float(red.mean()),
            reduction_std=float(red.std(ddof=1)) if len(red) > 1 else 0.0,
            reduction_min=float(red.min()), reduction_max=float(red.max()),
            n10=[x["mcnemar"]["n10"] for x in recs], n01=[x["mcnemar"]["n01"] for x in recs],
            p_max=max(x["mcnemar"]["p"] for x in recs),
            last_epoch_gnn_failures=[x.get("last_epoch_gnn_failures") for x in recs],
        )
    base = OUT_DIR / "baselines.json"
    if base.exists():
        out["baselines"] = json.loads(base.read_text())
    (OUT_DIR / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make_data", action="store_true")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--seed", type=int, action="append", default=[])
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.make_data:
        make_data()
    if a.baselines:
        run_baselines()
    for s in a.seed:
        run_seed(s)
    if a.summary:
        summary()
