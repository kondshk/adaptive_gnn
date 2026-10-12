#!/usr/bin/env python3
"""Rebuild the paper's results tables from the stored per-shot outcomes.

Every number in the results tables (and the derived numbers quoted in the
text) is recomputed here from the per-shot failure vectors in results/, not
copied from a summary file. paper/build_v5.py inserts these rows into the
manuscript, and

    python make_tables.py                    # print every table and number
    python make_tables.py --check paper/STAR_GNN_BB_v5.tex

prints the rows or checks that each generated row appears verbatim (up to
whitespace) in the manuscript.

Conventions: LERs are failures / shots, rounded half-up; intervals are
Wilson 95%; McNemar tests are two-sided with continuity correction; ranges
are minimum--maximum over the five training seeds; Holm's step-down
correction is applied within the comparison family of each table.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
import sys
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
R = ROOT / "results"
SEEDS = range(5)


# ------------------------------------------------------------------ helpers
def q(x, dp):
    """Round half-up to dp decimals; x may be a float or an exact Decimal."""
    d = x if isinstance(x, Decimal) else Decimal(repr(float(x)))
    return str(d.quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP))


def ler(k, n, dp=4):
    return q(Decimal(int(k)) / Decimal(int(n)), dp)


def pct(x, dp=0):
    return q(Decimal(repr(float(x))) * 100, dp)


def num(k):
    """Integer with LaTeX thousands separator."""
    s = f"{int(k):,}"
    return s.replace(",", "{,}")


def rng(vals, fmt=num):
    lo, hi = min(vals), max(vals)
    return fmt(lo) if lo == hi else f"{fmt(lo)}--{fmt(hi)}"


def wilson(k, n, z=1.96):
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(c - h, 0.0), c + h


def mcnemar(a, b):
    """a: reference failures, b: compared failures; n10 = only a fails, n01 = only b fails."""
    n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); d = n10 + n01
    chi2 = (abs(n10 - n01) - 1) ** 2 / d if d else 0.0
    return n10, n01, (math.erfc(math.sqrt(chi2 / 2)) if d else 1.0)


def holm(ps):
    m = len(ps); order = sorted(range(m), key=lambda i: ps[i]); adj = [0.0] * m; run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n)


def sci(p, digits=1):
    """p-value as LaTeX a\\times10^{b} (or \\leq bound)."""
    if p == 0:
        return "0"
    e = math.floor(math.log10(p)); m = p / 10 ** e
    mq = q(m, digits)
    if mq.startswith("10"):
        e += 1; mq = q(m / 10, digits)
    return f"{mq}\\!\\times\\!10^{{{e}}}"


def load(path):
    return np.load(path)


# ------------------------------------------------------------------ tables
def table_configs():
    """Table: BP / BP-OSD configurations on [[288,12,18]], p = 0.04."""
    f = load(R / "tables_v2" / "bposd_configs_failures.npz")
    spec = [("Original BP", "parallel", "prod-sum", "30", "bp_parallel_ps30"),
            ("Reference BP", "serial", "ms(0.8)", "100", "bp_serial_ms100"),
            ("Original BP-OSD", "parallel", "prod-sum", "30", "bposd_parallel_ps30"),
            ("Ref.\\ BP-OSD", "serial", "ms(0.8)", "100", "bposd_serial_ms100")]
    rows = []
    for name, sched, meth, it, key in spec:
        a = f[key]; k, n = int(a.sum()), len(a)
        dp = 4 if k / n >= 1e-3 else 5
        lo, hi = wilson(k, n)
        rows.append(f"{name} & {sched} & {meth} & {it} & {ler(k, n, dp)} [{q(lo, dp)},{q(hi, dp)}] & {num(k)} \\\\")
    return rows


def numbers_decomposition():
    f = load(R / "tables_v2" / "decomposition_failures.npz")
    a, b, c = f["bp_flooding_ms100"], f["bp_serial_ms100"], f["bposd_serial_ms100"]
    n = len(a)
    la, lb, lc = a.sum() / n, b.sum() / n, c.sum() / n
    ab, bc = mcnemar(a, b), mcnemar(b, c)
    return dict(flood=f"{q(la * 100, 2)}\\times10^{{-2}}", serial=f"{q(lb * 1000, 2)}\\times10^{{-3}}",
                osd=f"{q(lc * 1e4, 1)}\\times10^{{-4}}", f1=q(la / lb, 1), f2=q(lb / lc, 1), f_all=q(la / lc, 0),
                n_ab=f"{ab[0]}/{ab[1]}", n_bc=f"{bc[0]}/{bc[1]}")


def headline():
    """Per split: BP-10 failures, GNN-BP failures per seed, McNemar per seed."""
    out = {}
    for split in ("test_p04", "test_p06"):
        bp = np.load(R / "headline_v2" / f"{split}_bp_fail.npy")
        g = [np.load(R / "headline_v2" / f"seed{s}_{split}_gnn_fail.npy") for s in SEEDS]
        fs = load(R / "failure_sets" / f"{split}_failures.npz")
        out[split] = dict(bp=bp, gnn=g, bp100=fs["bp100"], bposd=fs["bposd"], n=len(bp),
                          mc=[mcnemar(bp, x) for x in g])
    return out


def table_headline():
    h = headline()
    cols = {}
    for split, d in h.items():
        n = d["n"]; kb = int(d["bp"].sum())
        lers = np.array([x.sum() / n for x in d["gnn"]])
        red = np.array([1 - x.sum() / kb for x in d["gnn"]])
        cols[split] = dict(
            bp=ler(kb, n), gnn=f"${q(lers.mean(), 4)}\\pm{q(lers.std(ddof=1), 4)}$",
            red=f"${pct(red.mean(), 1)}\\%\\pm{pct(red.std(ddof=1), 1)}\\%$",
            nn=f"{rng([m[0] for m in d['mc']])} / {rng([m[1] for m in d['mc']])}",
            bp100=ler(d["bp100"].sum(), n), bposd=ler(d["bposd"].sum(), n))
    a, b = cols["test_p04"], cols["test_p06"]
    return [f"BP, 10 iter. & {a['bp']} & {b['bp']} \\\\",
            f"GNN-BP, 10 iter. & {a['gnn']} & {b['gnn']} \\\\",
            f"\\quad relative reduction & {a['red']} & {b['red']} \\\\",
            f"\\quad $n_{{10}}$ / $n_{{01}}$ & {a['nn']} & {b['nn']} \\\\",
            f"BP, 100 iter. & {a['bp100']} & {b['bp100']} \\\\",
            f"Serial BP-OSD & {a['bposd']} & {b['bposd']} \\\\"]


def table_components():
    """Z-, X- and either-component failures at p = 0.04 (counts of 20,000)."""
    f = load(R / "components" / "test_p04_failures.npz")

    def cell(prefix, comp):
        keys = [prefix] if prefix in ("bp10", "bp100", "bposd") else [f"{prefix}_s{s}" for s in SEEDS]
        return rng([int(f[f"{k}_{comp}"].sum()) for k in keys])

    spec = [("BP, 10 iter.", "bp10"), ("GNN-BP, $\\Delta$ on both", "gnn_bp10"),
            ("GNN-BP, $\\Delta$ on $Z$ only", "gnn_bp10_zonly"), ("BP, 100 iter.", "bp100"),
            ("Serial BP-OSD", "bposd"), ("GNN + BP-OSD", "gnn_bposd")]
    return [f"{name} & {cell(k, 'z')} & {cell(k, 'x')} & {cell(k, 'any')} \\\\" for name, k in spec]


def numbers_components():
    out = {}
    for split in ("test_p04", "test_p06"):
        f = load(R / "components" / f"{split}_failures.npz")
        kb = int(f["bp10_any"].sum())
        red_both = np.array([1 - f[f"gnn_bp10_s{s}_any"].sum() / kb for s in SEEDS])
        red_z = np.array([1 - f[f"gnn_bp10_zonly_s{s}_any"].sum() / kb for s in SEEDS])
        xs = [int(f[f"gnn_bp10_s{s}_x"].sum()) for s in SEEDS]
        out[split] = dict(red_both=f"{pct(red_both.mean(), 1)}\\%\\pm{pct(red_both.std(ddof=1), 1)}\\%",
                          red_zonly=f"{pct(red_z.mean(), 1)}\\%\\pm{pct(red_z.std(ddof=1), 1)}\\%",
                          x_gnn=rng(xs), x_bp=int(f["bp10_x"].sum()),
                          x_osd=int(f["bposd_x"].sum()),
                          x_gnnosd=rng([int(f[f"gnn_bposd_s{s}_x"].sum()) for s in SEEDS]))
    return out


def gnn_vs_bposd():
    f = load(R / "tables_v2" / "gnn_vs_bposd_failures.npz")
    ps = ["0.02", "0.03", "0.04", "0.05"]
    tests = {(p, s): mcnemar(f[f"p{p}_bposd"], f[f"p{p}_seed{s}_best"]) for p in ps for s in SEEDS}
    keys = list(tests)
    adj = dict(zip(keys, holm([tests[k][2] for k in keys])))
    return f, ps, tests, adj


def table_gnn_osd():
    f, ps, tests, adj = gnn_vs_bposd()
    rows = []
    for p in ps:
        n = len(f[f"p{p}_bposd"])
        gl = [int(f[f"p{p}_seed{s}_best"].sum()) for s in SEEDS]
        n10 = [tests[(p, s)][0] for s in SEEDS]; n01 = [tests[(p, s)][1] for s in SEEDS]
        worse = sum(1 for s in SEEDS if tests[(p, s)][2] < 0.05 and tests[(p, s)][1] > tests[(p, s)][0])
        worse_h = sum(1 for s in SEEDS if adj[(p, s)] < 0.05 and tests[(p, s)][1] > tests[(p, s)][0])
        rows.append(f"{q(Decimal(p), 3)} & {ler(f[f'p{p}_bposd'].sum(), n)} & "
                    f"{ler(min(gl), n)}--{ler(max(gl), n)} & {rng(n10)} / {rng(n01)} & {worse}/5 & {worse_h}/5 \\\\")
    return rows


def numbers_gnn_osd():
    f, ps, tests, adj = gnn_vs_bposd()
    better = sum(1 for k, t in tests.items() if t[2] < 0.05 and t[0] > t[1])
    hi = [k for k in tests if k[0] != "0.02"]
    worse = sum(1 for k in hi if tests[k][2] < 0.05 and tests[k][1] > tests[k][0])
    worse_h = sum(1 for k in hi if adj[k] < 0.05 and tests[k][1] > tests[k][0])
    return dict(better=better, worse_ge003=worse, worse_ge003_holm=worse_h, pairs_ge003=len(hi))


def table_failure_sets():
    d = json.loads((R / "failure_sets" / "test_p04.json").read_text())
    f = load(R / "failure_sets" / "test_p04_failures.npz")
    comp = load(R / "components" / "test_p04_failures.npz")
    bp, osd, b100 = f["bp10"], f["bposd"], f["bp100"]
    n = len(bp)
    gnn = [f"gnn_bp10_s{s}" for s in SEEDS]; gos = [f"gnn_bposd_s{s}" for s in SEEDS]
    mism = {k: d["describe"][k]["syndrome_mismatch"] for k in d["describe"]}
    both = int((bp & osd).sum())
    fixed = [bp & ~f[k] for k in gnn]
    fx = d["gnn_fixes"]
    nc = [fx[k]["fixed_syndrome_mismatch_in_bp10"] / fx[k]["fixed"] for k in gnn]
    of = [fx[k]["fixed_that_bposd_fails"] / fx[k]["fixed"] for k in gnn]
    cover = [int((osd & f[k]).sum()) / int(osd.sum()) for k in gnn]
    newly = [int((f[k] & ~osd).sum()) for k in gos]; resc = [int((~f[k] & osd).sum()) for k in gos]
    pr = lambda v: f"{pct(min(v))}--{pct(max(v))}\\%" if pct(min(v)) != pct(max(v)) else f"{pct(v[0])}\\%"
    return [
        f"BP, 10 iter. & {num(bp.sum())} & {num(mism['bp10'])} \\\\",
        f"BP, 100 iter. & {num(b100.sum())} & {num(mism['bp100'])} \\\\",
        f"Serial BP-OSD & {num(osd.sum())} & {num(mism['bposd'])} \\\\",
        f"GNN + BP, 10 iter. & {rng([int(f[k].sum()) for k in gnn])} & {rng([mism[k] for k in gnn])} \\\\",
        f"GNN + BP-OSD & {rng([int(f[k].sum()) for k in gos])} & {rng([mism[k] for k in gos])} \\\\",
        "\\midrule",
        "\\multicolumn{3}{@{}l}{\\textit{BP vs.\\ BP-OSD failure sets}} \\\\",
        f"Both fail & \\multicolumn{{2}}{{c}}{{{num(both)}}} \\\\",
        f"\\makecell[l]{{Only BP fails /\\\\only BP-OSD fails}} & \\multicolumn{{2}}{{c}}{{{num(int((bp & ~osd).sum()))} / {num(int((~bp & osd).sum()))}}} \\\\",
        f"BP fails, given BP-OSD fails & \\multicolumn{{2}}{{c}}{{{pct(both / osd.sum())}\\%}} \\\\",
        f"BP-OSD fails, given BP fails & \\multicolumn{{2}}{{c}}{{{pct(both / bp.sum())}\\%}} \\\\",
        f"\\makecell[l]{{BP-100 failures\\\\also BP failures}} & \\multicolumn{{2}}{{c}}{{{num(int((b100 & bp).sum()))} / {num(b100.sum())}}} \\\\",
        "\\midrule",
        "\\multicolumn{3}{@{}l}{\\textit{What the GNN changes}} \\\\",
        f"Shots GNN fixes for BP & \\multicolumn{{2}}{{c}}{{{rng([int(x.sum()) for x in fixed])}}} \\\\",
        f"\\quad of which BP did not converge & \\multicolumn{{2}}{{c}}{{{pr(nc)}}} \\\\",
        f"\\quad of which BP-OSD also fails & \\multicolumn{{2}}{{c}}{{{pr(of)}}} \\\\",
        f"\\makecell[l]{{BP-OSD failures also\\\\GNN+BP failures}} & \\multicolumn{{2}}{{c}}{{{pr(cover)}}} \\\\",
        f"\\makecell[l]{{GNN+BP-OSD vs.\\ BP-OSD:\\\\newly failed / rescued}} & \\multicolumn{{2}}{{c}}{{{rng(newly)} / {rng(resc)}}} \\\\",
    ]


def numbers_failure_sets():
    out = {}
    for split in ("test_p04", "test_p06"):
        f = load(R / "failure_sets" / f"{split}_failures.npz")
        bp, osd = f["bp10"], f["bposd"]
        out[split] = dict(bp_given_osd=pct((bp & osd).sum() / osd.sum()), osd_given_bp=pct((bp & osd).sum() / bp.sum()))
    d = json.loads((R / "failure_sets" / "test_p04.json").read_text())
    nm = d["describe"]["bp10"]
    out["bp_nonconv_frac"] = pct(nm["syndrome_mismatch"] / nm["failures"])
    f = load(R / "failure_sets" / "test_p04_failures.npz")
    bp, osd = f["bp10"], f["bposd"]
    out["both"], out["bp_only"], out["osd_only"] = (num(int((bp & osd).sum())), num(int((bp & ~osd).sum())),
                                                    num(int((~bp & osd).sum())))
    return out


def table_mechanism():
    d = json.loads((R / "mechanism" / "test_p04.json").read_text())
    c = d["bp10"]["classes"]
    fixed = [d["gnn"][f"s{s}"]["fixed"] for s in SEEDS]
    spec = [("Wrong class (syndrome met)", "converged_wrong"), ("Transient (met before it.~10)", "transient"),
            ("Slow (met by it.~100)", "slow"), ("Oscillating (never met)", "oscillating"),
            ("Stuck (never met)", "stuck")]
    rows = [f"{name} & {num(c[k])} & {rng([x[k] for x in fixed])} \\\\" for name, k in spec]
    rows.append("\\midrule")
    rows.append(f"Total & {num(c['n'])} & {rng([x['n'] for x in fixed])} \\\\")
    rows.append(f"\\quad symmetric configuration & {num(c['symmetric'])} & {rng([x['symmetric'] for x in fixed])} \\\\")
    return rows


def numbers_mechanism():
    d = json.loads((R / "mechanism" / "test_p04.json").read_text())
    sym = d["symmetric_all"]
    bs = d["by_symmetric"]
    c = d["bp10"]["classes"]
    fixed = [d["gnn"][f"s{s}"]["fixed"] for s in SEEDS]
    raw = np.load(ROOT / "data" / "headline_v2" / "test_p04.npz")
    ze, hz = raw["z_errors"].astype(np.int64), raw["hz"].astype(np.int64)
    w3 = ze.sum(1) == 3
    s3 = ((ze @ hz.T) == 3).any(1)
    bp = load(R / "failure_sets" / "test_p04_failures.npz")["bp10"]
    return dict(classes={k: num(c[k]) for k in ("converged_wrong", "transient", "slow", "oscillating", "stuck", "n")},
                fixed={k: rng([x[k] for x in fixed]) for k in ("converged_wrong", "transient", "slow", "oscillating", "n")},
                nonconv=num(c["n"] - c["converged_wrong"]),
                sym_share_of_failures=pct(c["symmetric"] / c["n"]),
                w3_nosym_n=num(int((w3 & ~s3).sum())), w3_nosym_bp=pct(bp[w3 & ~s3].mean(), 1),
                sym_shots=num(sym["symmetric"]), sym_frac=pct(sym["symmetric_frac"], 1),
                bp_fail_sym=pct(bs["bp10"]["rate_with"]), bp_fail_nosym=pct(bs["bp10"]["rate_without"]),
                osd_fail_sym=pct(bs["bposd"]["rate_with"]), bp100_fail_sym=pct(bs["bp100"]["rate_with"]),
                gnn_fail_sym=f"{pct(min(bs[f'gnn_bp10_s{s}']['rate_with'] for s in SEEDS))}--"
                             f"{pct(max(bs[f'gnn_bp10_s{s}']['rate_with'] for s in SEEDS))}",
                early_stop=num(d["bp10"]["early_stopping_failures"]), bp_fail=num(d["bp10"]["failures"]))


def weight3_symmetric():
    """Failures on weight-3 Z errors that cover half of a Z stabilizer, per decoder."""
    out = {}
    for split in ("test_p04", "test_p06"):
        raw = np.load(ROOT / "data" / "headline_v2" / f"{split}.npz")
        ze, hz = raw["z_errors"].astype(np.int64), raw["hz"].astype(np.int64)
        sym = ((ze @ hz.T) == 3).any(1) & (ze.sum(1) == 3)
        f = load(R / "failure_sets" / f"{split}_failures.npz")
        out[split] = dict(n=int(sym.sum()), **{k: int((f[k] & sym).sum()) for k in
                                                 ["bp10", "bp100", "bposd"] + [f"gnn_bp10_s{s}" for s in SEEDS]})
    return out


def oracle():
    f = load(R / "tables_v2" / "oracle_mean_failures.npz")
    return f


def table_oracle():
    f = oracle()
    rows = []
    spec = [("72_12_6", "0.04", "$\\code{72}{12}{6}$ (6)"), ("144_12_12", "0.06", "$\\code{144}{12}{12}$ (12)"),
            ("288_12_18", "0.07", "$\\code{288}{12}{18}$ (18)")]
    for code, p, label in spec:
        for i, s in enumerate(("0.25", "0.5", "1.0")):
            m, o = f[f"{code}_p{p}_s{s}_mean"], f[f"{code}_p{p}_s{s}_oracle"]
            n = len(m)
            km, ko = int(m.sum()), int(o.sum())
            n10, n01, pv = mcnemar(m, o)
            (ml, mh), (ol, oh) = wilson(km, n), wilson(ko, n)
            first = f"{label} & {p}" if i == 0 else " & "
            rows.append(f"{first} & {s} & {ler(km, n)} [{q(ml, 4)},{q(mh, 4)}] & {ler(ko, n)} [{q(ol, 4)},{q(oh, 4)}]"
                        f" & {pct(1 - ko / km)}\\% & {num(n10)} / {num(n01)} \\\\")
    return rows


def numbers_oracle():
    f = oracle()
    ps = []
    for code, p in (("72_12_6", "0.04"), ("144_12_12", "0.06"), ("288_12_18", "0.07")):
        for s in ("0.25", "0.5", "1.0"):
            ps.append(mcnemar(f[f"{code}_p{p}_s{s}_mean"], f[f"{code}_p{p}_s{s}_oracle"])[2])
    same_p = {}
    for code in ("72_12_6", "144_12_12", "288_12_18"):
        m, o = f[f"{code}_p0.04_s1.0_mean"], f[f"{code}_p0.04_s1.0_oracle"]
        same_p[code] = dict(mean=int(m.sum()), oracle=int(o.sum()), gap=pct(1 - o.sum() / m.sum()))
    return dict(max_p=max(ps), max_p_holm=max(holm(ps)), same_p004_sigma1=same_p)


def circuit():
    pts = [("0.001", R / "circuit_v2" / "oracle_v4_paper_points_p0.001_failures.npz", 20),
           ("0.003", R / "circuit_v2" / "oracle_v4_paper_points_p0.003_failures.npz", 20),
           ("0.01", R / "circuit_v2" / "oracle_v4_p010_p0.01_failures.npz", 40)]
    return [(p, load(path), prof) for p, path, prof in pts]


def table_circuit():
    rows = []
    for p, f, prof in circuit():
        m, o = f["mean"], f["oracle"]
        n = len(m); km, ko = int(m.sum()), int(o.sum())
        n10, n01, pv = mcnemar(m, o)
        if km == 0 and ko == 0:
            ub = q(wilson(0, n)[1] * 100, 2)
            rows.append(f"{p} & {num(n)} & 0 [0,{ub}] & 0 [0,{ub}] & 0 & 0 & --- \\\\")
            continue
        (ml, mh), (ol, oh) = wilson(km, n), wilson(ko, n)
        rows.append(f"{p} & {num(n)} & {q(km / n * 100, 2)} [{q(ml * 100, 2)},{q(mh * 100, 2)}] & "
                    f"{q(ko / n * 100, 2)} [{q(ol * 100, 2)},{q(oh * 100, 2)}] & {n10} & {n01} & ${sci(pv, 0)}$ \\\\")
    return rows


def numbers_circuit():
    p, f, prof = circuit()[-1]
    m, o = f["mean"], f["oracle"]
    per = len(m) // prof
    pm = m.reshape(prof, per).sum(1); po = o.reshape(prof, per).sum(1)
    better, worse = int((po < pm).sum()), int((po > pm).sum())
    n10, n01, pv = mcnemar(m, o)
    return dict(gap=pct(1 - o.sum() / m.sum()), mean=q(m.mean() * 100, 2), oracle=q(o.mean() * 100, 2),
                p=sci(pv, 1), better=better, worse=worse, profiles=prof, sign_p=sci(sign_test(better, worse), 1))


def drift_available():
    return (R / "drift" / "drift.json").exists()


def table_drift():
    d = json.loads((R / "drift" / "drift.json").read_text())
    f = load(R / "drift" / "drift_failures.npz")
    rows = []
    gnn = [f"gnn_s{s}" for s in d["seeds"]]
    for vol in d["vols"]:
        v = str(vol)
        r = d["results"][v]
        n = r["oracle"]["shots"]
        cell = lambda k: ler(r[k]["failures"], n, 4)
        gl = [r[k]["failures"] for k in gnn]
        st = r["sign_tests"]
        rows.append(f"{q(Decimal(v), 2)} & {cell('oracle')} & {cell('stale')} & {cell('mean')} & {cell('empir')} & "
                    f"{rng(gl, lambda k: ler(k, n, 4))} & "
                    f"{st['gnn_vs_stale']['gnn_better']}/{st['gnn_vs_stale']['gnn_worse']} & "
                    f"{st['gnn_vs_empir']['gnn_better']}/{st['gnn_vs_empir']['gnn_worse']} \\\\")
    return rows


TABLES = {"configs": table_configs, "headline": table_headline, "components": table_components,
          "gnn_osd": table_gnn_osd, "failure_sets": table_failure_sets, "mechanism": table_mechanism,
          "oracle": table_oracle, "circuit": table_circuit, "drift": table_drift}


def numbers():
    out = dict(decomposition=numbers_decomposition(), components=numbers_components(),
               gnn_osd=numbers_gnn_osd(), failure_sets=numbers_failure_sets(), mechanism=numbers_mechanism(),
               weight3_symmetric=weight3_symmetric(), oracle=numbers_oracle(), circuit=numbers_circuit())
    return out


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


if __name__ == "__main__":
    tables = {k: fn() for k, fn in TABLES.items() if k != "drift" or drift_available()}
    if len(sys.argv) > 2 and sys.argv[1] == "--check":
        tex = norm(pathlib.Path(sys.argv[2]).read_text())
        bad = [(k, r) for k, rows in tables.items() for r in rows if norm(r) not in tex]
        for k, r in bad:
            print(f"MISSING [{k}] {r}")
        print(f"{sum(len(r) for r in tables.values()) - len(bad)} rows found, {len(bad)} missing")
        sys.exit(1 if bad else 0)
    for k, rows in tables.items():
        print(f"% ---- {k}")
        print("\n".join(rows))
    print(json.dumps(numbers(), indent=1, default=str))
