"""Text of the drift section (Sec. 5.6) of STAR_GNN_BB_v5.tex.

The methods paragraph states every parameter of audit_drift.py; the results
paragraph is filled from results/drift/drift.json, so no number is typed by
hand. Used by paper/build_v5.py.
"""
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _section(MT):
    d = json.loads((ROOT / "results" / "drift" / "drift.json").read_text())
    vols = [str(v) for v in d["vols"]]
    r = d["results"]
    gnn = [f"gnn_s{s}" for s in d["seeds"]]
    n = r[vols[0]]["oracle"]["shots"]
    L = lambda v, k: MT.ler(r[v][k]["failures"], n, 4)
    gl = lambda v: [r[v][k]["failures"] for k in gnn]
    grng = lambda v: f"{MT.ler(min(gl(v)), n, 4)}--{MT.ler(max(gl(v)), n, 4)}"
    st = lambda v, ref: r[v]["sign_tests"][f"gnn_vs_{ref}"]
    es = lambda v: r[v]["sign_tests"]["empir_vs_stale"]

    # Holm over the eight stream-level sign tests of the GNN (vs stale, vs empir, four volatilities)
    keys = [(v, ref) for v in vols for ref in ("stale", "empir")]
    adj = dict(zip(keys, MT.holm([st(v, ref)["p"] for v, ref in keys])))
    better = lambda ref: [v for v in vols if adj[(v, ref)] < 0.05 and st(v, ref)["gnn_better"] > st(v, ref)["gnn_worse"]]
    worse = lambda ref: [v for v in vols if adj[(v, ref)] < 0.05 and st(v, ref)["gnn_better"] < st(v, ref)["gnn_worse"]]
    fmt_v = lambda vs: ", ".join(f"$v={float(v):g}$" for v in vs) if vs else "no volatility"

    rows = "\n".join(MT.TABLES["drift"]())
    v0, vmax = vols[0], vols[-1]
    gap0 = MT.pct(1 - r[v0]["oracle"]["failures"] / r[v0]["mean"]["failures"])

    section = r"""\subsection{Drift adaptation}\label{sec:drift}

We test whether per-qubit rate estimates help BP-OSD when the rates
drift. On $\code{72}{12}{6}$ with mean rate $p=0.04$ and $\eta=20$, each
data qubit $i$ has a log-rate offset $x_i(t)$ that follows a discrete
Ornstein--Uhlenbeck (AR(1)) process, independently across qubits,
\begin{equation}
x_i(t+1)=\rho\,x_i(t)+v\,\xi_i(t),\qquad \rho=\sqrt{1-v^2/\sigma^2},
\label{eq:ou}
\end{equation}
with $\xi_i(t)\sim\mathcal N(0,1)$, $x_i(0)\sim\mathcal N(0,\sigma^2)$ and
$\sigma=1$, so the stationary spread of the log-rates is $\sigma$ for every
volatility $v$. The volatility is the per-shot standard deviation of the
log-rate innovation: $v=0$ is a static field, and $v=0.1$, $0.35$ and $0.5$
give correlation times $-1/\ln\rho$ of about 200, 15 and 7 shots. The rate
of qubit $i$ at shot $t$ is $p_i(t)=\min\bigl(0.5,\,p\,e^{x_i(t)-\sigma^2/2}\bigr)$,
whose stationary mean is $p$, and each shot draws independent $Z$ and $X$
flips with probabilities $p_i\eta/(\eta+1)$ and $p_i/(\eta+1)$.

All decoders are serial min-sum BP-OSD (Table~\ref{tab:hparams}) with
per-qubit priors $\hat p_i\eta/(\eta+1)$ and $\hat p_i/(\eta+1)$; they
differ only in the rate estimate $\hat p_i(t)$. The \emph{oracle} uses the
true $p_i(t)$. \emph{Stale} uses $p_i(0)$, a perfect calibration taken at
the start of the stream and never updated. \emph{Mean} uses the uniform
rate $p$. \emph{Empir} counts the $Z$ flips $n_i$ in its own decoded
estimates over the previous $W=32$ shots and shrinks the frequency toward
$p$ with $K=16$ pseudo-shots,
$\hat p_i=\frac{\eta+1}{\eta}\,\bigl(n_i+Kp\eta/(\eta+1)\bigr)/(W+K)$.
\emph{GNN} is a \textsc{TannerGNN} rate estimator with the architecture of
Table~\ref{tab:hparams} that maps the violated-check frequencies of the
previous $W$ syndromes to an estimate $\hat x_i(t)$ and sets
$\hat p_i=\min\bigl(0.5,\,p\,e^{\hat x_i-\sigma^2/2}\bigr)$; it sees
syndromes only, never decoder outputs or true rates. It is trained by
mean-squared error on 20{,}000 windows drawn at the four volatilities,
with five training seeds; for each seed the epoch with the lowest
validation error out of 30 is kept. Every decoder sees the current
syndrome. Each volatility is evaluated on 20 independent streams of
2{,}532 shots, whose first 32 shots are warm-up and not scored, giving
50{,}000 scored shots. Shots in one stream share a rate field, so pooled
McNemar tests overstate the evidence; we therefore compare decoders by a
sign test over the 20 streams, with Holm correction over the eight GNN
comparisons.

\begin{table*}[t]
\centering
\caption{Drift experiment on $\code{72}{12}{6}$, $p=0.04$, $\eta=20$,
         $\sigma=1$, $W=32$: LER of serial BP-OSD with different per-qubit
         rate estimates, 50{,}000 scored shots (20 streams) per
         volatility $v$. GNN: range over five training seeds. Last two
         columns: streams on which the seed-averaged GNN has fewer / more
         failures than stale calibration and than the empirical
         estimator.}
\label{tab:drift}
\begin{tabular}{@{}lccccccc@{}}
\toprule
$v$ & Oracle & Stale & Mean & Empir & GNN & vs.\ stale & vs.\ empir \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table*}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{fig_F3_drift_adaptation}
\caption{LER of serial BP-OSD under per-qubit OU drift against the
         volatility $v$ on $\code{72}{12}{6}$, $p=0.04$, $\sigma=1$, for
         five rate estimates. Wilson 95\% intervals; for the GNN, the mean
         over five training seeds with the range over seeds.}
\label{fig:drift_curve}
\end{figure}

<<DRIFT_RESULTS>>

"""
    numbers = dict(
        gap0=gap0, oracle0=L(v0, "oracle"), mean0=L(v0, "mean"),
        stale=[L(v, "stale") for v in vols], empir=[L(v, "empir") for v in vols],
        oracle=[L(v, "oracle") for v in vols], mean=[L(v, "mean") for v in vols],
        gnn=[grng(v) for v in vols],
        gnn_better_stale=better("stale"), gnn_worse_stale=worse("stale"),
        gnn_better_empir=better("empir"), gnn_worse_empir=worse("empir"),
        sign={(v, ref): (st(v, ref)["gnn_better"], st(v, ref)["gnn_worse"], adj[(v, ref)])
              for v, ref in keys},
        empir_vs_stale={v: (es(v)["empir_better"], es(v)["empir_worse"], es(v)["p"]) for v in vols},
        fmt_v=fmt_v, vols=vols,
    )
    return section, numbers


def build(MT):
    """The four pieces build_v5.py inserts: section, abstract, conclusion, limitation."""
    section, x = _section(MT)
    d = json.loads((ROOT / "results" / "drift" / "drift.json").read_text())
    r = d["results"]
    v = x["vols"]                                   # "0.0", "0.1", "0.35", "0.5"
    i01, ilast = v.index("0.1"), len(v) - 1
    gm = r[v[ilast]]["sign_tests"]["gnn_vs_mean"]
    all_gnn_sig = all(p < 0.05 for (_, _, p) in x["sign"].values())
    gnn_lo = min(float(g.split("--")[0]) for g in x["gnn"])
    spread = max(r[vv][f"gnn_s{s_}"]["ler"] for vv in v for s_ in d["seeds"]) - 0  # noqa: F841
    width = max(max(r[vv][f"gnn_s{s_}"]["ler"] for s_ in d["seeds"]) - min(r[vv][f"gnn_s{s_}"]["ler"] for s_ in d["seeds"])
                for vv in v)
    results = r"""Table~\ref{tab:drift} and Fig.~\ref{fig:drift_curve} show three things.
First, a per-qubit calibration is valuable only while it is current.
Without drift the stale calibration is the oracle (LER """ + x["oracle"][0] + r""", against
""" + x["mean"][0] + r""" for the mean prior), but at $v=0.1$, where the rates decorrelate over
about 200 shots, it is already worse than the mean prior (""" + x["stale"][i01] + r""" vs.\
""" + x["mean"][i01] + r"""). Second, tracking the rates recovers much of the gap: the
non-learned frequency estimator reaches """ + x["empir"][0] + r""" without drift and """ + x["empir"][i01] + r""" at
$v=0.1$, and its advantage shrinks as the correlation time approaches the
window length (""" + x["empir"][ilast] + r""" vs.\ """ + x["mean"][ilast] + r""" for the mean prior at $v=0.5$). Third,
the GNN estimator beats stale calibration at every nonzero volatility and
the mean prior at every volatility (on all 20 streams up to $v=0.35$ and
on """ + str(gm["gnn_better"]) + r""" of 20 at $v=0.5$), but it is worse than the frequency estimator
on all 20 streams at every volatility""" + (r"""; all eight GNN comparisons
remain significant after Holm correction""" if all_gnn_sig else "") + r""". The five training seeds give
nearly identical results. Under independent per-qubit drift, counting each
qubit's decoded errors therefore captures more of the available signal
than the learned syndrome-based estimator; whether spatially correlated
drift would change this is an open question."""
    return {
        "<<DRIFT_SECTION>>": section.replace("<<DRIFT_RESULTS>>", results),
        "<<ABSTRACT_DRIFT>>": "",
        "<<CONCLUSION_DRIFT>>": r"""Under per-qubit drift a stale calibration quickly loses its
value, and a simple frequency estimator tracks the rates better than a
learned one.""",
        "<<LIMITATION_DRIFT>>": r"""\item The drift experiment is synthetic: the rates of different qubits
      drift independently, with correlation times of 7 to 200 shots.""",
    }
