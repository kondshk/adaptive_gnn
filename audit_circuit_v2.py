#!/usr/bin/env python3
"""Circuit-level re-run on [[72,12,6]] with complete detectors (Table 8).

The original circuit had no first-round Z-check detectors and no detectors
comparing the last ancilla round with the final data measurement, so faults
near the time boundaries flipped observables without firing any detector.
This script:

  --uniform   BP-OSD logical failure rate under uniform circuit noise, with the
              fixed circuit and with the legacy detector set (for comparison);
  --oracle    the Table 8 oracle-gap experiment: per-qubit log-normal rates
              p_q = p_mean * exp(sigma * z_q), z_q ~ N(0,1), decoded with
              (a) a uniform prior at the log-normal mean rate, (b) a uniform prior
              at the median rate p_mean, and (c) the true per-qubit rates;
  --summary   collect results.

Noise model (astra_stim.biased_noise.apply_biased_circuit_noise): after every
1- and 2-qubit gate each touched qubit gets PAULI_CHANNEL_1 with
p_X = p/(eta+1), p_Z = p*eta/(eta+1), p_Y = 0; untouched qubits get the same
channel at every TICK (idle); measurements and resets flip with probability p.
Memory experiment in the Z basis, 6 rounds of X- then Z-check extraction.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from multiprocessing import Pool

import numpy as np
import stim

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from astra_stim.biased_noise import (BiasedCircuitNoiseSpec, apply_biased_circuit_noise,
                                     rescale_noise_per_qubit)
from astra_stim.qldpc_circuit import CSSCodeSpec, build_qldpc_memory_circuit_text
from codes.code_registry import get_code_params
from codes.codes_q import create_bivariate_bicycle_codes
from gnn_pipeline.decoding_failure import component_failures
from gnn_pipeline.dem_decoder import extract_dem_pcm

OUT_DIR = ROOT / "results" / "circuit_v2"
ROUNDS = 6
ETA = 20.0
SIGMA = 1.0
BPOSD = dict(max_iter=100, bp_method="ms", ms_scaling_factor=0.625, schedule="serial",
             osd_method="osd_cs", osd_order=10)


def base_circuit_text() -> str:
    css, _, _ = create_bivariate_bicycle_codes(**get_code_params("72_12_6"))
    spec = CSSCodeSpec(*(np.asarray(getattr(css, a), dtype=np.uint8) for a in ("hx", "hz", "lx", "lz")))
    return build_qldpc_memory_circuit_text(spec, rounds=ROUNDS, basis="z")


def legacy_detectors(text: str) -> str:
    """Drop the first-round and final detectors, reproducing the old circuit."""
    head, tail = text.rsplit("\nM ", 1)
    head = "\n".join(l for l in head.split("\n")
                     if not (l.startswith("DETECTOR") and l.count("rec[") == 1))
    tail = "\n".join(l for l in tail.split("\n") if not l.startswith("DETECTOR"))
    return head + "\nM " + tail


def noisy(text: str, p: float) -> stim.Circuit:
    return apply_biased_circuit_noise(stim.Circuit(text), spec=BiasedCircuitNoiseSpec(p=p, eta=ETA))


class DemDecoder:
    def __init__(self, circuit: stim.Circuit):
        from ldpc import BpOsdDecoder
        self.pcm, probs, self.obs = extract_dem_pcm(str(circuit))
        self.dec = BpOsdDecoder(self.pcm, error_channel=list(probs), **BPOSD)

    def fail(self, dets: np.ndarray, obs: np.ndarray) -> np.ndarray:
        dets = dets.astype(np.uint8)
        est = np.stack([self.dec.decode(d) for d in dets])
        return component_failures(est, dets, self.pcm, self.obs.T, obs.astype(np.uint8)).failure


def wilson(k, n, z=1.96):
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def mcnemar(a, b):
    n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); d = n10 + n01
    chi2 = (abs(n10 - n01) - 1) ** 2 / d if d else 0.0
    return dict(n10=n10, n01=n01, chi2=chi2, p=math.erfc(math.sqrt(chi2 / 2)) if d else 1.0)


# ---------------------------------------------------------------- uniform noise
def _uniform_chunk(args):
    variant, p, shots, seed = args
    text = base_circuit_text()
    c = noisy(legacy_detectors(text) if variant == "legacy" else text, p)
    dets, obs = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    return DemDecoder(c).fail(dets, obs)


def run_uniform(ps, shots, workers, variants=("fixed", "legacy"), tag=""):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    chunks = max(workers, 1)
    with Pool(workers) as pool:
        for variant in variants:
            for i, p in enumerate(ps):
                t0 = time.time()
                jobs = [(variant, p, shots // chunks, 10_000 + 100 * i + j) for j in range(chunks)]
                f = np.concatenate(pool.map(_uniform_chunk, jobs))
                c = noisy(legacy_detectors(base_circuit_text()) if variant == "legacy" else base_circuit_text(), p)
                k = int(f.sum())
                out[f"{variant}_p{p}"] = dict(variant=variant, p=p, shots=len(f), failures=k,
                                              ler=k / len(f), ci=wilson(k, len(f)),
                                              detectors=c.num_detectors)
                print(f"{variant:6s} p={p}: {k}/{len(f)} = {k / len(f):.4f} "
                      f"({c.num_detectors} detectors, {time.time() - t0:.0f}s)", flush=True)
    (OUT_DIR / f"uniform{tag}.json").write_text(json.dumps(out, indent=2))


# ---------------------------------------------------------------- oracle gap
def _oracle_profile(args):
    """One per-qubit rate profile: sample `shots` and decode with three priors."""
    p_mean, profile, shots = args
    text = base_circuit_text()
    ref = noisy(text, p_mean)
    rng = np.random.default_rng(profile)
    scale = np.exp(SIGMA * rng.standard_normal(ref.num_qubits))
    true_c = rescale_noise_per_qubit(ref, scale)
    dets, obs = true_c.compile_detector_sampler(seed=profile).sample(shots, separate_observables=True)
    mean_c = rescale_noise_per_qubit(ref, np.full(ref.num_qubits, math.exp(SIGMA ** 2 / 2)))
    return dict(profile=profile,
                mean=DemDecoder(mean_c).fail(dets, obs),
                median=DemDecoder(ref).fail(dets, obs),
                oracle=DemDecoder(true_c).fail(dets, obs))


def run_oracle(ps, profiles, shots_per_profile, workers, tag=""):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    with Pool(workers) as pool:
        for i, p in enumerate(ps):
            t0 = time.time()
            ids = [1_000_000 * (i + 1) + j for j in range(profiles)]
            res = pool.map(_oracle_profile, [(p, pid, shots_per_profile) for pid in ids])
            f = {k: np.concatenate([r[k] for r in res]) for k in ("mean", "median", "oracle")}
            per_profile = {k: [int(r[k].sum()) for r in res] for k in f}
            diff = np.array(per_profile["mean"]) - np.array(per_profile["oracle"])
            rec = dict(p_mean=p, sigma=SIGMA, profiles=profiles, shots_per_profile=shots_per_profile,
                       shots=len(f["oracle"]), per_profile_failures=per_profile,
                       profiles_mean_worse=int((diff > 0).sum()),
                       profiles_oracle_worse=int((diff < 0).sum()))
            for k, v in f.items():
                rec[f"{k}_failures"] = int(v.sum())
                rec[f"{k}_ler"] = float(v.mean())
                rec[f"{k}_ci"] = wilson(int(v.sum()), len(v))
            rec["mcnemar_mean_vs_oracle"] = mcnemar(f["mean"], f["oracle"])
            rec["mcnemar_median_vs_oracle"] = mcnemar(f["median"], f["oracle"])
            rec["gap_mean_vs_oracle"] = (1 - rec["oracle_failures"] / rec["mean_failures"]
                                         if rec["mean_failures"] else float("nan"))
            out[f"p{p}"] = rec
            np.savez_compressed(OUT_DIR / f"oracle_p{p}_failures.npz", **f)
            m = rec["mcnemar_mean_vs_oracle"]
            print(f"p_mean={p}: mean {rec['mean_failures']} median {rec['median_failures']} "
                  f"oracle {rec['oracle_failures']} / {rec['shots']}; n10/n01 {m['n10']}/{m['n01']} "
                  f"p={m['p']:.2g}; profiles mean-worse {rec['profiles_mean_worse']} "
                  f"oracle-worse {rec['profiles_oracle_worse']} ({time.time() - t0:.0f}s)", flush=True)
    (OUT_DIR / f"oracle{tag}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--uniform", action="store_true")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--p", type=float, nargs="+", default=[0.001, 0.003])
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--profiles", type=int, default=40)
    ap.add_argument("--shots_per_profile", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--variants", nargs="+", default=["fixed", "legacy"])
    ap.add_argument("--tag", default="", help="suffix for the output json")
    a = ap.parse_args()
    if a.uniform:
        run_uniform(a.p, a.shots, a.workers, a.variants, a.tag)
    if a.oracle:
        run_oracle(a.p, a.profiles, a.shots_per_profile, a.workers, a.tag)
