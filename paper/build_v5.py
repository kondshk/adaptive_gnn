#!/usr/bin/env python3
"""Build STAR_GNN_BB_v5.tex from the version Nithin reviewed.

Every change is an exact, single-match replacement R(old, new, tag) applied to
paper/reviewed/STAR_GNN_BB_reviewed.tex, so the full edit list can be audited
here. Tags name the review item each edit answers (M1-M14, R1-R7 of the
October 2026 review memo; 'proof' = proofreading carried over from v4).
Numeric table rows are not typed by hand: they come from make_tables.py, which
recomputes them from the stored per-shot outcomes.

Usage (from the repository root):
    python paper/build_v5.py
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import make_tables as MT  # noqa: E402

SRC = ROOT / "paper" / "reviewed" / "STAR_GNN_BB_reviewed.tex"
DST = ROOT / "paper" / "STAR_GNN_BB_v5.tex"
s = SRC.read_text()
log = []


def R(old, new, tag):
    global s
    n = s.count(old)
    if n != 1:
        sys.exit(f"FAIL {tag}: found {n} matches for:\n{old[:200]}")
    s = s.replace(old, new)
    log.append(tag)


def rows(name):
    return "\n".join(MT.TABLES[name]())


N = MT.numbers()
C04, C06 = N["components"]["test_p04"], N["components"]["test_p06"]
G = N["gnn_osd"]
FS = N["failure_sets"]
ME = N["mechanism"]
W3 = N["weight3_symmetric"]
CI = N["circuit"]
DE = N["decomposition"]

# =====================================================================
# Abstract (R6: about 20% shorter; M1, M7, M12 wording)
# =====================================================================
a0 = s.index(r"\begin{abstract}") + len(r"\begin{abstract}")
a1 = s.index(r"\end{abstract}")
old_abstract = s[a0:a1]
R(old_abstract, r"""
Bivariate bicycle (BB) quantum low-density parity-check (QLDPC) codes
combine finite encoding rate with low-weight checks, but their loopy
Tanner graphs remain challenging for iterative decoding. We study a
Tanner-graph neural network (\textsc{TannerGNN}) that adds qubit-wise
log-likelihood corrections to belief propagation (BP) under biased CSS
noise. The network is trained with a per-qubit focal loss on the sampled
error, and its checkpoint is selected with 10-iteration flooding BP. On
the $\code{72}{12}{6}$ BB code at $p=0.04$ and $\eta=20$, it reduces the
logical error rate (LER) of that decoder by $12.5\%\pm0.6\%$ on 20{,}000
held-out syndromes (five training seeds; paired McNemar $p<10^{-36}$ for
every seed). The gain does not transfer to a stronger reference decoder,
serial min-sum BP-OSD: the correction never improves it and, after Holm
correction, is significantly worse for """ + str(G["worse_ge003_holm"]) + r""" of the """ + str(G["pairs_ge003"]) + r""" seed and
error-rate pairs with $p\geq0.03$. Most BP-OSD failures are also BP
failures, the GNN mainly repairs shots on which BP converges slowly and
which BP-OSD already decodes, and weight-three errors that cover half of a
stabilizer defeat BP with or without the GNN but never BP-OSD. Baseline
experiments show that BP schedule, update rule, iteration count and OSD
settings change the LER by orders of magnitude, and we prove that
uniformly rescaling the channel LLRs cannot change the hard decisions of
unclipped min-sum BP. Revealing the true per-qubit error rates, by
contrast, reduces the BP-OSD LER by up to 88\% under synthetic
heterogeneous noise and by """ + CI["gap"] + r"""\% under circuit-level noise.
<<ABSTRACT_DRIFT>>""", "R6/M1/M7/M12 abstract")

# =====================================================================
# Introduction
# =====================================================================
R(r"""well as theoretically interesting~\cite{IBMRoadmap2025}.""",
  r"""well as theoretically interesting~\cite{IBMRoadmap2025,Yoder2025TourDeGross}.""", "R7 intro cite")
R(r"""\emph{what gain does the learned correction provide relative to the BP
decoder used in training, and what additional information is needed for
noise adaptation?}""", r"""\emph{what gain does the learned correction provide relative to the BP
decoder it is developed with, and what additional information is needed
for noise adaptation?}""", "M1 intro question")
R(r"""approaches make the choice of baseline unusually consequential: a neural
correction trained against one BP schedule or stopping rule need not
address the residual failures of a stronger decoder.""",
  r"""approaches make the choice of baseline unusually consequential: a neural
correction developed for one BP schedule or stopping rule need not
address the residual failures of a stronger decoder.""", "M1 intro")

# M8 / R2: related work, organized by decoder mechanism
R(r"""\subsection{Contributions}\label{sec:contributions}""",
  r"""\subsection{Related work}\label{sec:related}

Quantum Tanner graphs combine short cycles with degeneracy, and both
cause non-convergence and finite-length error floors in
BP~\cite{MacKay2004Sparse,Poulin2008Iterative,Roffe2020Landscape}.
Trapping-set analysis makes these failures structural rather than
mysterious: specific subgraphs, including symmetric configurations created
by stabilizers, can trap iterative decoders~\cite{Raveendran2021Trapping},
and degeneracy itself can be exploited inside BP~\cite{Kuo2022Degeneracy}.

Several families of methods repair such failures, with different latency
and complexity trade-offs. OSD post-processing~\cite{Fossorier1995OSD,Panteleev2021QLDPC,Roffe2020Landscape}
is the most widely used reference. Alternatives include stabilizer
inactivation~\cite{duCrest2022Inactivation},
check-agnosia~\cite{duCrest2024CheckAgnosia}, localized statistics
decoding~\cite{Hillmann2025LSD}, syndrome-based linear
programming~\cite{Javed2026LP}, collective bit
flipping~\cite{Chytas2025Collective}, and trapping-set-aware iterative
decoders of linear complexity~\cite{Pradhan2026Linear}. For BB codes in
particular, recent work targets real-time decoding: Relay-BP~\cite{Muller2025Relay},
graph rewiring with ensembles of normalized min-sum
decoders~\cite{Maan2026Correlated}, diversity and ensemble
methods~\cite{GarciaHerrero2026Diversity}, ambiguity
clustering~\cite{Wolanski2024Ambiguity}, and hardware implementations of
BP-OSD~\cite{Bascones2025FPGA}.

Learned decoders either modify BP messages or take over part of the
inference. Neural BP with a degeneracy-aware loss~\cite{Liu2019Neural},
GNN stages interleaved with BP~\cite{Gong2024GNN}, learned message
passing~\cite{Maan2025MLBP}, circuit-level neural decoders for BB
codes~\cite{Blue2026MLBB}, and calibration-conditioned
decoders~\cite{Stein2026FiLM} report both strong gains and limits in
scaling and transfer. In this landscape serial min-sum BP-OSD is a useful,
fully specified reference decoder, not the state of the art, and we use it
only in that role. Our question is narrower: what a learned LLR correction
gains relative to the BP configuration it is developed with, whether that
gain survives a stronger reference decoder, and what side information
adaptation actually needs.

\subsection{Contributions}\label{sec:contributions}""", "M8/R2 related work")

R(r"""\item \textbf{A held-out GNN gain over its training-aligned BP baseline.}
On $\code{72}{12}{6}$ at $p=0.04$ and $\eta=20$,
\textsc{TannerGNN}+flooding BP reduces LER by $12.5\%\pm0.6\%$ on
20{,}000 held-out syndromes over five training seeds (McNemar
$p<10^{-36}$ for every seed). The training rates span
$p\in[0.005,0.045]$; at $p=0.06$, outside that range, the reduction is
$10.6\%\pm0.7\%$ (Sec.~\ref{sec:gnn_positive}).""",
  r"""\item \textbf{A held-out GNN gain over the BP decoder it is developed
with.} On $\code{72}{12}{6}$ at $p=0.04$ and $\eta=20$,
\textsc{TannerGNN}+flooding BP reduces LER by $12.5\%\pm0.6\%$ on
20{,}000 held-out syndromes over five training seeds (McNemar
$p<10^{-36}$ for every seed). The training rates span
$p\in[0.005,0.045]$; at $p=0.06$, outside that range, the reduction is
$10.6\%\pm0.7\%$. The whole gain is in the $Z$ component; adding the
correction to the $X$ component as well only adds failures there
(Sec.~\ref{sec:gnn_positive}).""", "M1/M3 contribution 1")

R(r"""\item \textbf{Baseline configuration and alignment can dominate the
apparent learned gain.} Serial min-sum BP with OSD post-processing is
substantially stronger than the flooding BP used in GNN training. The
same learned correction does not improve this stronger decoder and is
significantly worse from $p=0.03$ on. A paired failure-set analysis shows
that 93\% of BP-OSD failures are also BP failures and that most shots
the GNN repairs are already decoded by BP-OSD. Separate experiments on
$\code{288}{12}{18}$ show order-of-magnitude sensitivity to BP schedule,
update rule, iteration count, and OSD settings. The
training decoder and evaluation decoder are therefore part of the
experimental condition rather than interchangeable implementations of
``BP''
(Secs.~\ref{sec:baseline} and~\ref{sec:mismatch}).""",
  r"""\item \textbf{Baseline configuration and alignment can dominate the
apparent learned gain.} Serial min-sum BP with OSD post-processing is
substantially stronger than the flooding BP used to select the GNN
checkpoint. The same learned correction never improves this stronger
decoder and, after Holm correction, is significantly worse for """ + str(G["worse_ge003_holm"]) + r""" of
the """ + str(G["pairs_ge003"]) + r""" seed and error-rate pairs with $p\geq0.03$. A paired failure-set
analysis shows that """ + FS["test_p04"]["bp_given_osd"] + r"""\% of BP-OSD failures are also BP failures
(""" + FS["test_p04"]["osd_given_bp"] + r"""\% the other way) and that most shots the GNN repairs are
already decoded by BP-OSD. Separate experiments on
$\code{288}{12}{18}$ show order-of-magnitude sensitivity to BP schedule,
update rule, iteration count, and OSD settings. The decoder used for
model selection and the evaluation decoder are therefore part of the
experimental condition rather than interchangeable implementations of
``BP''
(Secs.~\ref{sec:baseline} and~\ref{sec:mismatch}).

\item \textbf{A recurring BP failure that the GNN does not repair.} When
a weight-three $Z$ error covers half of a weight-six stabilizer, the error
and its stabilizer complement are equally likely and logically
equivalent. BP, 100-iteration BP and GNN-BP fail on every such shot in our
test sets and BP-OSD on none; the GNN's bitwise loss gives both halves the
same target, so it cannot learn to break the tie
(Secs.~\ref{sec:training} and~\ref{sec:mismatch}).""", "M7/M12/R1 contributions 2-3")

# =====================================================================
# Background
# =====================================================================
R(r"""         $\Hmat_x$ and $\Hmat_z$ follow Eq.~\eqref{eq:bb} over
         $\F_2[x,y]/(x^\ell-1,\,y^m-1)$.}""",
  r"""         $\Hmat_x$ and $\Hmat_z$ follow Eq.~\eqref{eq:bb} over
         $\F_2[x,y]/(x^\ell-1,\,y^m-1)$. The names ``gross'' and
         ``two-gross'' follow IBM's usage~\cite{Yoder2025TourDeGross}.}""", "R7 code names")

R(r"""The channel LLR for qubit $i$ under $Z$-biased noise is
$\llr_{z,i} = \ln\bigl((1-p_z)/p_z\bigr)$,
$p_z = p\eta/(\eta+1)$.
CSS decoding separates $X$ and $Z$ components, processing each with
its respective parity-check matrix.""",
  r"""CSS decoding separates the $X$ and $Z$ components, processing each with
its respective parity-check matrix. Under the noise model of
Sec.~\ref{sec:noise} the matched channel LLRs are
$\llr_{z,i} = \ln\bigl((1-p_z)/p_z\bigr)$ for $Z$ errors and
$\llr_{x,i} = \ln\bigl((1-p_x)/p_x\bigr)$ for $X$ errors; at $p=0.04$ and
$\eta=20$, $p_z=0.0381$, $p_x=0.0019$ and
$\llr_{x}/\llr_{z}=6.26/3.23\approx1.94$. Each experiment below states
which prior every decoder receives.""", "M4/M5 channel LLRs")

R(r"""For temporal drift experiments we apply three modulation models
to the base rate $p$: sinusoidal ($p(t) = p + A\sin(2\pi t/T)$),
Ornstein--Uhlenbeck, and random telegraph noise (RTN), all
motivated by TLS physics~\cite{Klimov2018,Carroll2022}. The decoder
receives no direct access to instantaneous noise parameters in any
experiment.""",
  r"""Two kinds of temporal variation appear, both motivated by fluctuating
two-level systems~\cite{Klimov2018,Carroll2022}: the GNN training data
modulate the global rate sinusoidally, $p(t) = p_0 + A\sin(2\pi t/T)$
(Sec.~\ref{sec:training}), and the drift experiment of
Sec.~\ref{sec:drift} lets the rate of each qubit follow its own
Ornstein--Uhlenbeck process. Which noise parameters a decoder sees is
stated with each experiment.""", "M4 noise models")

# =====================================================================
# Methods (R3: FiLM / interleaved moved to Appendix A; M1, M2, M4, M6)
# =====================================================================
R(r"""\section{The \textsc{AdaptiveBP} Framework}\label{sec:methods}
""",
  r"""\section{The \textsc{AdaptiveBP} Framework}\label{sec:methods}

This section describes the model evaluated in Sec.~\ref{sec:results}. Two
further variants, FiLM conditioning and an interleaved GNN-BP pipeline,
are described in Appendix~\ref{app:variants}; they are not evaluated.
""", "R3 methods intro")

R(r"""variable node $v_i$ receives $\bvec{x}_i=[\ell,1,0,0]$, where
$\ell=\ln\bigl((1-p)/p\bigr)$ is the prior LLR, and an $X$-check
($Z$-check) node $c_j$ receives $[s_j,0,1,0]$ ($[s_j,0,0,1]$), where
$s_j$ is its syndrome bit.""",
  r"""variable node $v_i$ receives $\bvec{x}_i=[\ell,1,0,0]$, and an $X$-check
($Z$-check) node $c_j$ receives $[s_j,0,1,0]$ ($[s_j,0,0,1]$), where
$s_j$ is its syndrome bit. The prior LLR $\ell=\ln\bigl((1-p)/p\bigr)$ is
computed from the total physical error rate $p$ of the shot, not from
$p_z$ or $p_x$: for the drifting training streams it is the
instantaneous rate $p(t)$, and for the test sets the fixed $p$. The same
$\ell$ is the channel prior of both CSS components in the flooding-BP
decoders of Sec.~\ref{sec:results}. Its magnitude does not change
min-sum decisions (Sec.~\ref{sec:invariance}), but it sets the scale
against which the correction $\Delta_i$ acts.""", "M4 features")

R(r"""         normalisation follows at each node, and a readout MLP on the
         variable nodes gives the LLR corrections. In the FiLM variant
         (dashed), FiLM modulation scales the aggregated messages at
         each layer; the model of Sec.~\ref{sec:gnn_positive} does not
         use FiLM.}""",
  r"""         normalisation follows at each node, and a readout MLP on the
         variable nodes gives the LLR corrections. In the FiLM variant
         (dashed; Appendix~\ref{app:variants}), FiLM modulation scales the
         aggregated messages at each layer; the model of
         Sec.~\ref{sec:gnn_positive} does not use FiLM.}""", "R3 fig1 caption")

R(r"""\begin{remark}[Architectural limitation under biased noise]
The current implementation applies the same scalar $\Delta_i$ to
both $\llr_{z,i}$ and $\llr_{x,i}$, and computes GNN input
features from a single prior LLR $\ell$. Under $\eta = 20$, $\llr_{z,i}$
and $\llr_{x,i}$ differ substantially (factor $\approx 1.7$ in
LLR magnitude at $p=0.04$), so a single correction cannot
simultaneously be optimal for both error types. Separate $Z$- and
$X$-readout heads provide a direct way to remove this coupling and should
be tested under the same biased-noise conditions.
\end{remark}""",
  "", "M3/M5 remark (moved to Sec. 3.2)")

# cut the FiLM / neural-BP / interleaved subsections out of Sec. 3
f0 = s.index(r"\subsection{FiLM conditioning}\label{sec:film}")
f1 = s.index(r"\subsection{Training}\label{sec:training}")
moved = s[f0:f1]
R(moved, "", "R3 move FiLM/interleaved to appendix")

R(r"""Batch size / epochs             & 64 / 20, gradient clipping 1.0 \\
Checkpoint                      & \makecell[l]{fewest failures on\\4{,}000 validation shots} \\
Training seeds                  & 5 \\
BP in training loop             & \makecell[l]{flooding min-sum,\\scaling 0.8, 10 iter.} \\""",
  r"""Batch size / epochs             & 64 / 20, gradient clipping 1.0 \\
Checkpoint                      & \makecell[l]{fewest failures on\\4{,}000 validation shots} \\
Training seeds                  & 5 \\
BP for model selection          & \makecell[l]{flooding min-sum, scaling 0.8,\\10 iter.; not in the loss} \\""", "M1 table2 BP row")
R(r"""Loss                            & \makecell[l]{focal, $(\alpha,\gamma)=(0.25,2.0)$, on $e_z$;\\no syndrome term} \\""",
  r"""Prior feature $\ell$            & \makecell[l]{$\ln((1-p)/p)$, true $p$ of the shot,\\both components} \\
Loss                            & \makecell[l]{focal, $(\alpha,\gamma)=(0.25,2.0)$, on\\$q_j=\sigma(-(\ell+\Delta_j))$ vs.\ $e_z$;\\no syndrome term} \\""", "M1/M4 table2 loss")
R(r"""\texttt{bp\_method}             & \texttt{ms} (min-sum) \\""",
  r"""Implementation                  & \texttt{ldpc} 2.4.1 \texttt{BpOsdDecoder} \\
\texttt{bp\_method}             & \texttt{ms} (min-sum) \\""", "M8 table2 implementation")

R(r"""For a shot with prior LLR $\ell=\ln\bigl((1-p)/p\bigr)$ on every qubit,
the GNN outputs a correction $\Delta_j$ for each qubit $j$, giving the
corrected error probability $q_j=\sigma\bigl(-(\ell+\Delta_j)\bigr)$.
The model of Sec.~\ref{sec:gnn_positive} is trained with the focal
loss~\cite{Lin2017Focal} against the true $Z$ error $e_z$,
\begin{equation}""",
  r"""The model is trained as a per-qubit classifier of the sampled $Z$ error;
BP is not part of its loss. For a shot with prior LLR $\ell$ on every
qubit, the GNN outputs a correction $\Delta_j$ for each qubit $j$, giving
the corrected error probability $q_j=\sigma\bigl(-(\ell+\Delta_j)\bigr)$,
and the loss is the focal loss~\cite{Lin2017Focal} against the true $Z$
error $e_z$,
\begin{equation}""", "M1 training text 1")
R(r"""where $q_t=q_j$ and $\alpha_t=\alpha$ if $e_{z,j}=1$, and $q_t=1-q_j$
and $\alpha_t=1-\alpha$ otherwise, with $\alpha=0.25$ and $\gamma=2$;
no syndrome-consistency term is used, and the corrected LLRs
$\ell+\Delta_j$ are passed to both CSS components. The training data are
8{,}000 shots from two sinusoidally drifting streams,
$p(t)=p_0+0.015\sin(2\pi t/500)$ with $p_0\in\{0.02,0.03\}$, so the
training rates span $p\in[0.005,0.045]$. The FiLM and interleaved
variants of Secs.~\ref{sec:film}--\ref{sec:neuralbp} are trained on data
regenerated each epoch with $p\sim\mathrm{Uniform}[0.01,0.10]$, and add a
syndrome-consistency term, the binary cross-entropy between the measured
syndrome and $\tfrac12\bigl(1-\prod_{j\in\mathcal N(c)}(1-2q_j)\bigr)$
for each check $c$, with weight $0.1$. Table~\ref{tab:hparams} summarises
the configuration.""",
  r"""where $q_t=q_j$ and $\alpha_t=\alpha$ if $e_{z,j}=1$, and $q_t=1-q_j$
and $\alpha_t=1-\alpha$ otherwise, with $\alpha=0.25$ and $\gamma=2$; no
syndrome-consistency term is used. The computation graph is therefore
$(\ell,\synd)\to\textsc{TannerGNN}\to\Delta\to q\to\mathcal{L}$, and
gradients reach the GNN without passing through BP. Ten-iteration flooding
BP enters training only through model selection: after each of the 20
epochs the corrected priors $\ell+\Delta$ are decoded with that BP on
4{,}000 validation shots, and the epoch with the fewest failures is kept.
The training data are 8{,}000 shots from two sinusoidally drifting
streams, $p(t)=p_0+0.015\sin(2\pi t/500)$ with $p_0\in\{0.02,0.03\}$, so
the training rates span $p\in[0.005,0.045]$. Table~\ref{tab:hparams}
summarises the configuration.

The target is the sampled error $e_z$, so the loss is not
degeneracy-aware: an estimate that differs from $e_z$ by a stabilizer is
logically correct but is penalized like any other error. The extreme case
is an error that covers exactly half of a weight-six stabilizer $g$. Then
$e_z$ and $e_z\oplus g$ have the same weight and the same syndrome, so
they are equally likely and the network receives the same input for
both; the loss rewards treating the six qubits alike and gives no
incentive to prefer either half. Section~\ref{sec:mismatch} shows that
such configurations are a major cause of BP failures. Degeneracy-aware
objectives, such as that of Liu and Poulin~\cite{Liu2019Neural}, avoid
this problem; we did not use one.

\begin{remark}[One correction for two CSS components]
The GNN outputs one correction $\Delta_i$ per qubit and is trained on $Z$
errors only. In the experiments of
Sec.~\ref{sec:results} the same $\Delta_i$ is added to the prior $\ell$ of
both components, although the two error types differ: $X$ errors are
$\eta=20$ times rarer, and the matched LLRs differ by a factor
$\llr_x/\llr_z\approx1.94$ at $p=0.04$. Section~\ref{sec:gnn_positive}
measures the consequence and shows that adding $\Delta_i$ to the $Z$
component only is better.
\end{remark}""", "M1/M2 training text 2 + M3/M5 remark")

# =====================================================================
# Experimental setup (M1 fig. 3, M8 regimes, M12 protocol)
# =====================================================================
R(r"""\caption{Training and evaluation pipeline of the GNN experiments. The
         GNN is trained through 10-iteration flooding BP and evaluated on
         separately seeded held-out shots in Regime~A (flooding BP) and
         Regime~B (serial BP-OSD); each decoder receives identical
         syndrome batches to enable McNemar paired testing.}""",
  r"""\caption{Training and evaluation pipeline of the GNN experiments. The
         GNN is trained with a per-qubit focal loss, without BP in the
         loss; 10-iteration flooding BP selects the checkpoint on
         validation shots. The model is evaluated on separately seeded
         held-out shots in Regime~A (flooding BP) and Regime~B (serial
         BP-OSD); each decoder receives identical syndrome batches to
         enable McNemar paired testing.}""", "M1 fig3 caption")
R(r"""\textbf{Regime A (training-aligned baseline):} The GNN is evaluated
against flooding BP, the base decoder used during training. This
comparison isolates the gain relative to the decoder represented in the
training computation graph.""",
  r"""\textbf{Regime A (development baseline):} The GNN is evaluated with
flooding BP, the decoder used to select its checkpoint. This comparison
isolates the gain relative to that decoder.""", "M1 regime A")
R(r"""\textbf{Regime B (stronger classical baseline):} The same learned
correction is evaluated with the serial-schedule min-sum BP-OSD reference
configuration. This tests whether the correction also improves the
stronger classical decoder considered here; Sec.~\ref{sec:mismatch}
shows that it does not in the tested regime.""",
  r"""\textbf{Regime B (stronger reference decoder):} The same learned
correction is evaluated with serial-schedule min-sum BP-OSD
(Table~\ref{tab:hparams}). This tests whether the correction also
improves a stronger classical decoder; BP-OSD serves here as a reference
point, not as the best available decoder (Sec.~\ref{sec:related}). Plain
BP-OSD uses a uniform prior in each component ($p_z$ and $p_x$, or $p$ in
Table~\ref{tab:gnn_null}); the value of a uniform prior does not change its
decisions, which we confirmed on all 20{,}000 test shots at $p=0.04$. With
the GNN, qubit $i$ enters BP-OSD with prior error probability
$\sigma\bigl(-(\ell+\Delta_i)\bigr)$ in both components.
Section~\ref{sec:mismatch} shows that the correction does not help in the
tested regime.""", "M4/M8 regime B")
R(r"""McNemar tests~\cite{McNemar1947}. We report Wilson score 95\% confidence
intervals for LER estimates.""",
  r"""McNemar tests~\cite{McNemar1947} (two-sided, with continuity
correction). We report Wilson score 95\% confidence intervals for LER
estimates.

\paragraph{Comparison families.} Each results table defines one family
of paired tests: the ten GNN-BP vs.\ BP tests of
Table~\ref{tab:gnn_flooding} (two error rates, five seeds), the twenty
GNN+OSD vs.\ BP-OSD tests of Table~\ref{tab:gnn_null} (four error rates,
five seeds), the nine oracle cells of Table~\ref{tab:oracle}, and the GNN
comparisons of the drift experiment (Table~\ref{tab:drift}). Within a
family we control the familywise error rate with Holm's step-down
procedure~\cite{Holm1979} at level 0.05 and say so where adjusted results
are reported; other $p$-values are unadjusted and exploratory. For
emphasized comparisons we report the discordant counts $n_{10}$ and
$n_{01}$, and we never round a $p$-value to zero.""", "M12 protocol")

# =====================================================================
# Results 5.1 (table rows from the raw outcomes)
# =====================================================================
R(r"""Original BP     & parallel & prod-sum & 30  & 0.0569 [0.0555,0.0583] & 5{,}689 \\
Reference BP    & serial   & ms(0.8)  & 100 & 0.0017 [0.0015,0.0020] &    174 \\
Original BP-OSD & parallel & prod-sum & 30  & 0.0398 [0.0386,0.0411] & 3{,}983 \\
Ref.\ BP-OSD    & serial   & ms(0.8)  & 100 & 0.00034 [0.00024,0.00048] &   34 \\""",
  rows("configs"), "M13 table3 rows")

# =====================================================================
# Results 5.3 GNN vs flooding BP (M1, M3, M12, gradient audit)
# =====================================================================
R(r"""We evaluated the GNN against flooding BP---the decoder it was
trained against---on separately seeded held-out syndromes: 4{,}000""",
  r"""We evaluated the GNN with flooding BP---the decoder used to select its
checkpoint---on separately seeded held-out syndromes: 4{,}000""", "M1 5.3 intro")
R(r"""         ($n_{01}$): shots only BP (only GNN-BP) fails. BP is flooding
         min-sum.}""",
  r"""         ($n_{01}$): shots only BP (only GNN-BP) fails. BP is flooding
         min-sum run for exactly 10 or 100 iterations; the last hard
         decision is used.}""", "M1 table4 caption")
R(r"""BP, 10 iter.                 & 0.1077 & 0.2867 \\
GNN-BP, 10 iter.             & $0.0942\pm0.0006$ & $0.2564\pm0.0019$ \\
\quad relative reduction     & $12.5\%\pm0.6\%$ & $10.6\%\pm0.7\%$ \\
\quad $n_{10}$ / $n_{01}$    & 313--370 / 64--95 & 343--394 / 66--80 \\
BP, 100 iter.                & 0.0863 & 0.2403 \\
Serial BP-OSD                & 0.0707 & 0.2138 \\""", rows("headline"), "M13 table4 rows")
R(r"""the training range (McNemar $p<10^{-36}$ for every seed). The gain comes""",
  r"""the training range (McNemar $p<10^{-36}$ for every seed, also after Holm
correction over the ten tests). The gain comes""", "M12 5.3 holm")
R(r"""iterations and serial BP-OSD both remain better than GNN-BP. A gradient audit found non-zero
signal through the network (input-to-readout gradient ratio $0.22$) and
non-trivial LLR corrections (mean $|\Delta_i|=0.854$).""",
  r"""iterations and serial BP-OSD both remain better than GNN-BP. A gradient
audit on the five checkpoints found non-zero signal through the network
(input-to-readout gradient ratio $0.63$--$2.19$) and non-trivial LLR
corrections (mean $|\Delta_i|=0.54$--$1.01$).

\begin{table}[t]
\centering
\caption{Failures by CSS component on $\code{72}{12}{6}$, $p=0.04$,
         $\eta=20$, 20{,}000 held-out shots (counts). $Z$: $Z$ errors
         decoded on $\Hmat_x$; $X$: $X$ errors decoded on $\Hmat_z$;
         Either: the failure criterion of Sec.~\ref{sec:metrics}. GNN rows
         give the range over five training seeds.}
\label{tab:components}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lccc@{}}
\toprule
Decoder & $Z$ & $X$ & Either \\
\midrule
""" + rows("components") + r"""
\bottomrule
\end{tabular}
\end{table}

\paragraph{The two CSS components.} Table~\ref{tab:components} separates
the failures at $p=0.04$ by component. All failures of BP, 100-iteration
BP and BP-OSD are in the $Z$ component: at $p_x=0.0019$ none of them
fails on the $X$ component in 20{,}000 shots. The whole GNN gain is
therefore in the $Z$ component. Adding $\Delta$ to the $X$ component as
well, as in Table~\ref{tab:gnn_flooding}, gains nothing there and creates
""" + C04["x_gnn"] + r""" $X$-component failures. Adding it to the $Z$ component only removes
them and gives a reduction of $""" + C04["red_zonly"] + r"""$ ($""" + C06["red_zonly"] + r"""$ at $p=0.06$). This
variant was chosen after seeing the test results, so we keep the
pre-specified setting in Table~\ref{tab:gnn_flooding}. Because BP makes
no $X$-component errors at these operating points, a separately trained
$X$ head would have nothing to correct, and leaving the $X$ prior
unchanged is the appropriate choice. With BP-OSD the $X$ component is
essentially unaffected (at most """ + {"1": "one"}.get(C04["x_gnnosd"].split("--")[-1], C04["x_gnnosd"].split("--")[-1]) + r""" extra failure at $p=0.04$), so the
comparisons below do not depend on this choice.""", "M3 components + gradient audit")
R(r"""generalization result. The interleaved architecture of
Sec.~\ref{sec:neuralbp} is not evaluated in this work.""",
  r"""generalization result. The FiLM and interleaved variants
(Appendix~\ref{app:variants}) are not evaluated in this work.""", "R3 5.3 end")

# =====================================================================
# Results 5.4 mismatch (M1, M7, M12, M13, R1)
# =====================================================================
R(r"""\subsection{Training/evaluation baseline mismatch}\label{sec:mismatch}""",
  r"""\subsection{Development/evaluation baseline mismatch}\label{sec:mismatch}""", "M1 5.4 title")
R(r"""syndrome sets, and their LER values are not directly comparable:""",
  r"""syndrome sets, and their LER values are not directly comparable.""", "proof colon")
R(r"""\caption{GNN-BP vs.\ serial BP-OSD on $\code{72}{12}{6}$, $\eta=20$,
         20{,}000 identical shots per $p$, five training seeds. Last
         column: seeds for which GNN+OSD is significantly worse
         (McNemar $p<0.05$).}
\label{tab:gnn_null}
\footnotesize
\begin{tabular}{@{}lccc@{}}
\toprule
$p$ & Serial BP-OSD & GNN+OSD & Seeds worse \\
\midrule
0.020 & 0.0093 & 0.0095--0.0102 & 1/5 \\
0.030 & 0.0297 & 0.0306--0.0317 & 5/5 \\
0.040 & 0.0676 & 0.0688--0.0709 & 4/5 \\
0.050 & 0.1354 & 0.1384--0.1418 & 5/5 \\
\bottomrule
\end{tabular}""",
  r"""\caption{GNN+OSD vs.\ serial BP-OSD on $\code{72}{12}{6}$, $\eta=20$,
         20{,}000 identical shots per $p$, five training seeds (ranges
         over seeds). $n_{10}$ ($n_{01}$): shots only BP-OSD (only
         GNN+OSD) fails. Last two columns: seeds for which GNN+OSD is
         significantly worse (McNemar $p<0.05$), unadjusted and after
         Holm correction over the 20 tests. No seed is significantly
         better.}
\label{tab:gnn_null}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lccccc@{}}
\toprule
 & & & & \multicolumn{2}{c}{Seeds worse} \\
$p$ & BP-OSD & GNN+OSD & $n_{10}$ / $n_{01}$ & unadj. & Holm \\
\midrule
""" + rows("gnn_osd") + r"""
\bottomrule
\end{tabular}}""", "M12/M13 table gnn_null")
R(r"""         $\eta=20$, 20{,}000 shots per point; GNN+OSD is the mean over
         five training seeds. Red $\times$ marks operating points where
         every seed is statistically worse (McNemar $p<0.05$).
         Wilson 95\% CI error bars.}""",
  r"""         $\eta=20$, 20{,}000 shots per point. BP-OSD: Wilson 95\%
         intervals; GNN+OSD: mean over five training seeds, error bars
         give the range over seeds. The red $\times$ marks the operating
         point where every seed is significantly worse after Holm
         correction.}""", "M12 fig5 caption")
R(r"""Table~\ref{tab:gnn_null} and Fig.~\ref{fig:gnn_null_curve} show that
the GNN-corrected decoder never improves on serial BP-OSD: at $p=0.02$
one seed of five is significantly worse, and from $p=0.03$ on 14 of 15
seed--error-rate pairs are significantly \emph{worse}. The same checkpoint improves flooding BP, so these
data show that the learned correction does not transfer unchanged to the
stronger decoder.""",
  r"""Table~\ref{tab:gnn_null} and Fig.~\ref{fig:gnn_null_curve} show that
the GNN-corrected decoder never improves on serial BP-OSD: no seed is
significantly better at any error rate. Without correction, one seed of
five is significantly worse at $p=0.02$ and """ + str(G["worse_ge003"]) + r""" of the """ + str(G["pairs_ge003"]) + r"""
seed--error-rate pairs are significantly \emph{worse} from $p=0.03$ on;
after Holm correction over the 20 tests, """ + str(G["worse_ge003_holm"]) + r""" of these """ + str(G["pairs_ge003"]) + r""" remain,
including all five seeds at $p=0.05$. The same checkpoints improve
flooding BP, so these data show that the learned correction does not
transfer unchanged to the stronger decoder.""", "M12 5.4 text")
R(r"""The GNN was optimized through a
10-iteration flooding-BP computation graph and is then inserted ahead of
a decoder with a different schedule, iteration budget, and OSD
post-processing. The experiment therefore does not establish that LLR
corrections learned through the flooding-BP computation graph remain
useful after the schedule, iteration budget, and post-processing are
changed.""",
  r"""The GNN checkpoint was selected for 10-iteration flooding BP and is then
inserted ahead of a decoder with a different schedule, iteration budget,
and OSD post-processing.""", "M1/proof 5.4")
t0 = s.index(r"\multicolumn{3}{@{}l}{\textit{Overlaps}} \\")
R(r"""BP, 10 iter.            & 2{,}155 & 1{,}740 \\
BP, 100 iter.           & 1{,}727 & 700 \\
Serial BP-OSD           & 1{,}414 & 0 \\
GNN + BP, 10 iter.      & 1{,}874--1{,}906 & 838--975 \\
GNN + BP-OSD            & 1{,}439--1{,}451 & 0 \\
\midrule
\multicolumn{3}{@{}l}{\textit{Overlaps}} \\
\makecell[l]{BP-OSD failures\\also failed by BP} & \multicolumn{2}{c}{1{,}319 / 1{,}414 (93\%)} \\
\makecell[l]{BP-100 failures\\also failed by BP} & \multicolumn{2}{c}{1{,}727 / 1{,}727 (100\%)} \\
Shots GNN fixes for BP            & \multicolumn{2}{c}{313--370} \\
\quad of which BP did not converge & \multicolumn{2}{c}{91--93\%} \\
\quad of which BP-OSD also fails   & \multicolumn{2}{c}{31--36\%} \\
\makecell[l]{BP-OSD failures also\\failed by GNN+BP} & \multicolumn{2}{c}{87--88\%} \\
\makecell[l]{GNN+BP-OSD vs.\ BP-OSD:\\newly failed / rescued} & \multicolumn{2}{c}{237--282 / 206--254} \\""",
  rows("failure_sets"), "M7/M13 table failure sets")
R(r"""A paired failure-set analysis (Table~\ref{tab:failure_sets}) shows that
the two decoders do not fail on disjoint sets: every shot that
100-iteration BP fails is also failed by 10-iteration BP, and 93\% of
BP-OSD's failures are BP failures. What separates them is mainly""",
  r"""A paired failure-set analysis (Table~\ref{tab:failure_sets}) shows that
the failure sets of BP and BP-OSD overlap strongly but are not nested:
""" + FS["both"] + r""" shots fail under both, """ + FS["bp_only"] + r""" only under BP and """ + FS["osd_only"] + r""" only under
BP-OSD, so """ + FS["test_p04"]["bp_given_osd"] + r"""\% of BP-OSD's failures are BP failures and
""" + FS["test_p04"]["osd_given_bp"] + r"""\% of BP's failures are BP-OSD failures. The 100-iteration BP
failures, by contrast, are a strict subset of the 10-iteration ones.
What separates BP from BP-OSD is mainly""", "M7 5.4 overlap text")
R(r"""fails. The same pattern holds at $p=0.06$, where 96\% of BP-OSD failures
are BP failures. Training directly against the target decoder, or against a
differentiable surrogate of its post-processing stage, remains the
appropriate test of whether learned augmentation can improve BP-OSD.""",
  r"""fails. The same pattern holds at $p=0.06$, where """ + FS["test_p06"]["bp_given_osd"] + r"""\% of BP-OSD failures
are BP failures (and """ + FS["test_p06"]["osd_given_bp"] + r"""\% the other way). Training directly against
the target decoder, or against a differentiable surrogate of its
post-processing stage, remains the appropriate test of whether learned
augmentation can improve BP-OSD.

\begin{table}[t]
\centering
\caption{Classes of the BP failures on $\code{72}{12}{6}$ at $p=0.04$
         (10-iteration flooding BP, traced for 100 iterations) and of the
         shots the GNN repairs (range over five seeds). ``Met'': the hard
         decision reproduces the syndrome. A symmetric
         configuration is a $Z$ error covering three qubits of a
         weight-six $Z$ stabilizer.}
\label{tab:mechanism}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcc@{}}
\toprule
Class & BP failures & GNN repairs \\
\midrule
""" + rows("mechanism") + r"""
\bottomrule
\end{tabular}
\end{table}

\paragraph{Which failures the GNN repairs.} To see what kind of failure
the correction repairs, we traced flooding BP for 100 iterations on the
$p=0.04$ test set (Table~\ref{tab:mechanism}). Of the """ + ME["classes"]["n"] + r""" BP
failures, """ + ME["classes"]["converged_wrong"] + r""" reach a syndrome-consistent estimate in the wrong logical
class. The other """ + ME["nonconv"] + r""" do not reproduce the syndrome after 10
iterations: """ + ME["classes"]["transient"] + r""" had reproduced it earlier and moved away, """ + ME["classes"]["slow"] + r"""
reproduce it later, """ + ME["classes"]["oscillating"] + r""" never reproduce it and keep changing their hard
decision, and """ + ME["classes"]["stuck"] + r""" is stuck. Stopping each shot at its first
syndrome-consistent iteration would change the BP failure count only from
""" + ME["bp_fail"] + r""" to """ + ME["early_stop"] + r""", so the fixed iteration budget is not a handicap of the
baseline. The shots the GNN repairs are mostly slow ones (""" + ME["fixed"]["slow"] + r""" of
""" + ME["fixed"]["n"] + r"""), followed by oscillating (""" + ME["fixed"]["oscillating"] + r""") and wrongly converged
(""" + ME["fixed"]["converged_wrong"] + r""") shots: the correction mainly speeds up convergence.

A recurring structure among the failures is a symmetric stabilizer
configuration~\cite{Raveendran2021Trapping}: the $Z$ error covers exactly
three qubits of a weight-six $Z$ stabilizer $g$, so that $e_z$ and
$e_z\oplus g$ have the same weight and syndrome and are logically
equivalent. Such configurations occur in """ + ME["sym_frac"] + r"""\% of the test shots but in
""" + ME["sym_share_of_failures"] + r"""\% of BP failures; BP fails on """ + ME["bp_fail_sym"] + r"""\% of the shots that contain one and on
""" + ME["bp_fail_nosym"] + r"""\% of the others. Error weight does not explain this. On all
""" + str(W3["test_p04"]["n"]) + r""" weight-three errors of this kind, BP, 100-iteration BP and GNN-BP with
every seed fail, while BP-OSD fails on none (""" + str(W3["test_p06"]["n"]) + r""" of """ + str(W3["test_p06"]["n"]) + r""" and none at
$p=0.06$); on the """ + ME["w3_nosym_n"] + r""" weight-three errors without such a
configuration BP fails on """ + ME["w3_nosym_bp"] + r"""\%. The GNN lowers BP's failure rate on
symmetric configurations from """ + ME["bp_fail_sym"] + r"""\% to """ + ME["gnn_fail_sym"] + r"""\%, all of it on configurations of
weight four and more; on the weight-three ones it never helps, as expected
from its loss (Sec.~\ref{sec:training}). OSD resolves the tie by
construction, which is one reason why many BP failures are not BP-OSD
failures.""", "R1 mechanism")

# =====================================================================
# Results 5.5 oracle (R4, M9)
# =====================================================================
R(r"""fresh every shot (capped at 0.5), so the mean rate is $p$ for every
$\sigma$. A decoder knowing the true per-qubit rates (serial min-sum""",
  r"""fresh every shot (capped at 0.5), so the mean rate is $p$ for every
$\sigma$; the $Z$ and $X$ rates of qubit $i$ are $p_i\eta/(\eta+1)$ and
$p_i/(\eta+1)$. A decoder knowing the true per-qubit rates (serial min-sum""", "R4 5.5 noise")
R(r"""\includegraphics[width=\columnwidth]{fig_F2_oracle_vs_distance}
\caption{Oracle calibration gap (percentage LER reduction from
         per-qubit vs.\ mean-prior BP-OSD), 50k identical shots per
         point.
         Operating points: $p=0.04$, $0.06$, $0.07$ respectively;
         cross-code comparisons are partially confounded by the
         differing operating points.}""",
  r"""\includegraphics[width=\columnwidth]{fig_F2_oracle_gap}
\caption{Mean-prior and oracle serial BP-OSD LER with Wilson 95\%
         intervals, 50{,}000 identical shots per point. Each code is
         shown at its own operating point ($p=0.04$, $0.06$, $0.07$);
         because code size and operating point change together, the
         figure does not show a distance trend.}""", "R4 fig6")
R(r"""\caption{Oracle gap: percentage LER reduction from per-qubit
         over mean-prior BP-OSD, 50{,}000 identical shots per cell, at
         $p=0.04$, $0.06$, $0.07$ for the three codes. All McNemar
         $p<10^{-20}$. Parenthetical counts are mean-prior error events.}
\label{tab:oracle}
\begin{tabular}{@{}lccc@{}}
\toprule
Code ($d$) & $\sigma=0.25$ & $\sigma=0.5$ & $\sigma=1.0$ \\
\midrule
$\code{72}{12}{6}$   (6)  & $+11\%$ (3508) & $+29\%$ (3432) & $+65\%$ (3438) \\
$\code{144}{12}{12}$ (12) & $+12\%$ (3165) & $+35\%$ (3203) & $+80\%$ (2936) \\
$\code{288}{12}{18}$ (18) & $+13\%$ (3338) & $+41\%$ (3307) & $+88\%$ (2786) \\
\bottomrule
\end{tabular}""",
  r"""\caption{Oracle gap: LER of mean-prior and per-qubit (oracle) serial
         BP-OSD with Wilson 95\% intervals, 50{,}000 identical shots per
         row; the three codes are at different operating points.
         Reduction: $1-\mathrm{LER}_\mathrm{oracle}/\mathrm{LER}_\mathrm{mean}$.
         $n_{10}$ ($n_{01}$): shots only the mean-prior decoder (only the
         oracle) fails. All nine McNemar $p<10^{-22}$, also after Holm
         correction.}
\label{tab:oracle}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lcccccc@{}}
\toprule
Code ($d$) & $p$ & $\sigma$ & Mean prior & Oracle & Reduction & $n_{10}$ / $n_{01}$ \\
\midrule
""" + rows("oracle") + r"""
\bottomrule
\end{tabular}""", "R4/M13 table oracle")
_sp = N["oracle"]["same_p004_sigma1"]
R(r"""Consequently, these data do not isolate code distance as the cause of
the cross-code trend.""",
  r"""Consequently, these data do not isolate code distance as the cause of
the cross-code trend. At a common operating point ($p=0.04$, $\sigma=1$)
the reductions are """ + _sp["72_12_6"]["gap"] + r"""\%, """ + _sp["144_12_12"]["gap"] + r"""\% and """ + _sp["288_12_18"]["gap"] + r"""\% for the three codes, but the
two larger codes record only """ + str(_sp["144_12_12"]["mean"]) + r""" and """ + str(_sp["288_12_18"]["mean"]) + r""" mean-prior failures there, so
this controlled comparison is preliminary.""", "R4 same-p comparison")
R(r"""implementation can benefit when the realized per-qubit rates are
revealed. The gap is \emph{not}
recoverable from a single syndrome shot under i.i.d.\ per-qubit
noise: with a fresh field redrawn every shot, the uniform mean
prior is Bayes-optimal within the class of estimators that observe
only the syndrome, so no syndrome-conditioned GNN can do better
than the mean prior in this regime.""",
  r"""implementation can benefit when the realized per-qubit rates are
revealed. When the field is redrawn independently for every shot, these
rates are not identifiable from that shot's syndrome: averaged over the
field, the errors of each CSS component are independent across qubits
with the uniform rates $p_z$ and $p_x$, exactly as under uniform noise, so
a component's syndrome carries no information about the realized field.
(The field does correlate the $X$ and $Z$ errors of a qubit, which a joint
decoder could exploit; ours decode the components separately.) A
syndrome-only decoder could still outperform mean-prior BP-OSD by being a
better decoder for i.i.d.\ noise, but such a gain would not come from
estimating the $p_i$.""", "M9 identifiability")

# =====================================================================
# Results 5.6 drift (M10): rewritten from results/drift
# =====================================================================
d0 = s.index(r"\subsection{Drift adaptation}\label{sec:drift}")
d1 = s.index(r"\subsection{Circuit-level evaluation}\label{sec:circuit}")
R(s[d0:d1], "<<DRIFT_SECTION>>", "M10 drift section")

# =====================================================================
# Results 5.7 circuit (M11, M13)
# =====================================================================
R(r"""support, giving 432 detectors and 12 logical observables. Noise is a
biased single-qubit Pauli channel with $p_X=p/(\eta+1)$,
$p_Z=p\,\eta/(\eta+1)$, $p_Y=0$, $\eta=20$, applied to every qubit after
each one- and two-qubit gate and to every idle qubit at each time step;
measurements and resets flip with probability $p$. Under spatially
non-uniform noise each qubit $q$ (data and ancilla) has rate""",
  r"""support, giving 432 detectors and 12 logical observables. The noise,
summarized in Table~\ref{tab:circuitnoise}, is a custom biased Pauli
model, not standard circuit-level depolarizing noise: every one- and
two-qubit gate is followed by an independent single-qubit channel on each
qubit it acts on, with mutually exclusive $X$ and $Z$ faults and no $Y$
fault, so a two-qubit gate has no correlated two-qubit faults. Under
spatially non-uniform noise each qubit $q$ (data and ancilla) has rate""", "M11 circuit noise text")
R(r"""\paragraph{Decoders.} We decode the detector error model (4{,}068 fault
mechanisms) with serial min-sum BP-OSD (scaling 0.8, 100 iterations,
OSD-CS order 10), as in Table~\ref{tab:hparams}. The mean-prior decoder
uses the uniform rate $p$; the oracle uses the true $p_q$. We draw
20 rate profiles at $p\le0.003$ and 40 at
$p=0.01$, with 500 shots per profile. Failures follow
Sec.~\ref{sec:metrics}.""",
  r"""\begin{table}[t]
\centering
\caption{Circuit noise for a qubit $q$ with rate $p_q$ ($p_q=p$ under
         uniform noise), $\eta=20$. Each line acts independently on
         every qubit concerned.}
\label{tab:circuitnoise}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}ll@{}}
\toprule
Location & Fault \\
\midrule
After reset to $\ket{0}$ & $X$, prob.\ $p_q$ \\
\makecell[l]{After $H$ or CNOT\\(each qubit)} & \makecell[l]{$X$, prob.\ $p_q/(\eta+1)$, or\\$Z$, prob.\ $p_q\eta/(\eta+1)$} \\
\makecell[l]{Idle qubit,\\each time step} & as after a gate \\
Before $Z$ measurement & outcome flip, prob.\ $p_q$ \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Decoders.} Stim~\cite{Gidney2021Stim} converts each noisy
circuit into a detector error model (DEM) with 4{,}068 fault mechanisms on
432 detectors. Each mechanism collects the circuit faults that flip the
same detectors and observables, and its prior is their combined
probability; the exclusive $X$ and $Z$ outcomes of a channel are treated
as independent faults (Stim's \texttt{approximate\_disjoint\_errors}). The
mean-prior decoder uses the DEM of the circuit with every qubit at rate
$p$, and the oracle uses the DEM of the circuit with each qubit at its true
rate $p_q$; the two DEMs have the same mechanisms and differ only in their
priors. Both are decoded with serial min-sum BP-OSD (scaling 0.8, 100
iterations, OSD-CS order 10), as in Table~\ref{tab:hparams}. We draw 20
rate profiles at $p\le0.003$ and 40 at $p=0.01$, with 500 shots per
profile. Failures follow Sec.~\ref{sec:metrics}.""", "M11 noise table + DEM priors")
R(r"""         $\eta=20$, $\sigma=1$). $n_{10}$ ($n_{01}$): shots only the
         mean-prior decoder (only the oracle) fails. Zero failures in
         $10^4$ shots means LER $<3.8\times10^{-4}$ (95\% upper bound).}
\label{tab:circuit}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lccccc@{}}
\toprule
$p$ & Shots & Mean prior & Oracle & $n_{10}/n_{01}$ & $p_\mathrm{McN}$ \\
\midrule
0.001 & 10{,}000 & 0 & 0 & 0/0 & --- \\
0.003 & 10{,}000 & 0 & 0 & 0/0 & --- \\
0.01  & 20{,}000 & $1.35\%$ & $0.60\%$ & 177/28 & $5\!\times\!10^{-25}$ \\
\bottomrule
\end{tabular}""",
  r"""         $\eta=20$, $\sigma=1$). LER in percent with Wilson 95\%
         intervals. $n_{10}$ ($n_{01}$): shots only the mean-prior decoder
         (only the oracle) fails.}
\label{tab:circuit}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lcccccc@{}}
\toprule
$p$ & Shots & Mean prior & Oracle & $n_{10}$ & $n_{01}$ & $p_\mathrm{McN}$ \\
\midrule
""" + rows("circuit") + r"""
\bottomrule
\end{tabular}}""", "M13 circuit table")

# =====================================================================
# Discussion
# =====================================================================
R(r"""\textsc{TannerGNN} reduces the LER of the flooding BP decoder used in
its training loop by $12.5\%\pm0.6\%$ over five training seeds, and
gradient diagnostics show that the network produces non-trivial
corrections. The gain comes mainly from helping BP converge, and
100-iteration BP and serial BP-OSD remain stronger than the corrected
decoder.""",
  r"""\textsc{TannerGNN} reduces the LER of the flooding BP decoder used to
select its checkpoint by $12.5\%\pm0.6\%$ over five training seeds, and
gradient diagnostics show that the network produces non-trivial
corrections. The gain comes mainly from helping BP converge on shots where
it converges slowly, and 100-iteration BP and serial BP-OSD remain
stronger than the corrected decoder.""", "M1 6.1 para1")
R(r"""not transfer automatically to serial BP-OSD. This is consistent with the
training/evaluation mismatch: the failure sets of BP and BP-OSD are
nested, and the GNN's repairs fall mostly on shots BP-OSD already
decodes (Table~\ref{tab:failure_sets}).
Training against BP-OSD directly is complicated by the non-differentiable
OSD stage; differentiable surrogates, distillation, or training on the
residual failures of the target decoder are possible alternatives.""",
  r"""not transfer automatically to serial BP-OSD. This is consistent with the
mismatch between the development and evaluation decoders: most BP-OSD
failures are also BP failures, and the GNN's repairs fall mostly on shots
BP-OSD already decodes (Table~\ref{tab:failure_sets}). The symmetric
stabilizer configurations of Sec.~\ref{sec:mismatch} show the reverse
case: failures that BP-OSD avoids by construction and that a bitwise loss
cannot teach the GNN to repair. Training against BP-OSD directly is
complicated by the non-differentiable OSD stage; differentiable
surrogates, distillation, training on the residual failures of the target
decoder, or a degeneracy-aware loss~\cite{Liu2019Neural} are possible
alternatives.""", "M7/R1 6.1 para2")
R(r"""Second, the training and evaluation decoders must be distinguished. An
LLR correction learned through flooding BP need not remain beneficial""",
  r"""Second, the decoder used for model selection and the evaluation decoder
must be distinguished. An LLR correction selected for flooding BP need not
remain beneficial""", "M1 6.2")
R(r"""variant follows the same broad design principle, but its decoding
performance is not evaluated here, so we do not claim a competitive
result for that architecture.""",
  r"""variant (Appendix~\ref{app:variants}) follows the same broad design
principle, but its decoding performance is not evaluated here, so we do
not claim a competitive result for that architecture.""", "R3 6.3 gong")
R(r"""a fully specified competitive non-learned baseline, not only against
flooding BP.""",
  r"""a fully specified strong non-learned baseline, not only against
flooding BP.""", "M8 6.3 blue")
R(r"""and calibration-derived, unlike the single scalar $\phat$ used in our
initial model.""",
  r"""and calibration-derived, unlike the single global rate $p$ fed to our
FiLM variant (Appendix~\ref{app:variants}).""", "R3/M4 6.3 stein")
R(r"""single graph-wide scalar. Such a decoder should be compared with the""",
  r"""single graph-wide scalar. Soft syndrome information~\cite{Raveendran2023Soft}
is another form of side information that iterative decoders can use. Such
a decoder should be compared with the""", "R2 6.4 side information")

R(r"""\item All GNN variants were trained against 10-iteration
      flooding BP. No variant was trained against serial BP-OSD,
      which is the competitive classical baseline. Whether GNN
      augmentation of serial BP-OSD yields gains is untested.""",
  r"""\item The evaluated GNN was trained with a per-qubit loss and
      selected for 10-iteration flooding BP; no model was trained or
      selected for serial BP-OSD, our reference decoder. Whether GNN
      augmentation of serial BP-OSD yields gains is untested.""", "M1/M8 limitation i")
R(r"""\item The interleaved and FiLM architectures are described but their
      decoding performance is not evaluated.
\item All training and evaluation used CPU only, which limited
      hyperparameter search and training duration.
\item The single per-qubit correction $\Delta_i$ is applied to
      both $\llr_{z,i}$ and $\llr_{x,i}$, which is suboptimal
      under $\eta = 20$ biased noise; separate $Z$/$X$ readout
      heads are the recommended fix.
\item Code-capacity noise omits measurement errors, leakage,
      and correlated faults present in real circuits.""",
  r"""\item The interleaved and FiLM architectures
      (Appendix~\ref{app:variants}) are described but their decoding
      performance is not evaluated.
\item The correction is trained on $Z$ errors and, in the headline
      setting, also added to the $X$ component, where it only adds
      failures (Table~\ref{tab:components}); the $Z$-only setting was
      chosen after seeing the test results.
\item The training loss is not degeneracy-aware (Sec.~\ref{sec:training}).
\item Code-capacity noise omits measurement errors, leakage,
      and correlated faults present in real circuits, and the circuit
      model uses independent single-qubit channels rather than
      correlated two-qubit faults.
<<LIMITATION_DRIFT>>""", "M3/M2/M11/proof limitations")

# =====================================================================
# Conclusion (R6: shorter; M1, M7, M12)
# =====================================================================
c0 = s.index(r"\section{Conclusion}\label{sec:conclusion}") + len(r"\section{Conclusion}\label{sec:conclusion}")
c1 = s.index(r"\section*{Acknowledgments}")
R(s[c0:c1], r"""

On $\code{72}{12}{6}$, a Tanner-graph GNN whose checkpoint is selected
with 10-iteration flooding BP reduces the held-out LER of that decoder by
$12.5\%\pm0.6\%$ at $p=0.04$ over five training seeds, all of it in the
$Z$ component. The same correction never improves serial min-sum BP-OSD
and is often significantly worse; it mostly repairs shots on which BP
converges slowly and which BP-OSD already decodes. Weight-three errors
that cover half of a stabilizer defeat BP with or without the GNN but
never BP-OSD, and the bitwise training loss cannot teach the network to
break that tie. The demonstrated gain is therefore specific to the
flooding-BP baseline.

BP schedule, update rule, iteration budget and OSD configuration change
the finite-length LER by orders of magnitude, so they belong to the
decoder specification. A uniform rescaling of the channel LLRs cannot
change min-sum decisions, whereas the true per-qubit rates reduce
BP-OSD's LER by up to 88\% under synthetic heterogeneous noise and by """ + CI["gap"] + r"""\%
under circuit-level noise. <<CONCLUSION_DRIFT>> A natural next test is a
decoder conditioned on persistent spatial calibration data, trained with a
degeneracy-aware loss, and compared with fully specified BP-OSD and the
frequency estimator.
""", "R6/M1/M7 conclusion")

# =====================================================================
# Availability (M14)
# =====================================================================
R(r"""Source code, trained checkpoints, datasets and the analysis scripts
behind every table and figure except the drift experiment
(Sec.~\ref{sec:drift}) are available at
\url{https://github.com/kondshk/adaptive_gnn}. Code-capacity data are
sampled directly with NumPy; circuit-level data are generated with
Stim~\cite{Gidney2021Stim}.""",
  r"""{\raggedright All code, data, trained checkpoints and per-shot outcomes
behind every table and figure are available at
\url{https://github.com/kondshk/adaptive_gnn} (tag \texttt{paper-v5}).
The script \texttt{make\_tables.py} rebuilds every results table from the
stored outcomes, and \texttt{requirements-lock.txt} lists the exact
package versions. Code-capacity data are sampled directly with NumPy;
circuit-level data are generated with Stim~\cite{Gidney2021Stim}.\par}""", "M14 availability")

# =====================================================================
# Appendix A (R3, M6): the unevaluated variants, described as implemented
# =====================================================================
alg0 = moved.index(r"\begin{algorithm}[t]")
alg1 = moved.index(r"\end{algorithm}") + len(r"\end{algorithm}")
algorithm = moved[alg0:alg1]
algorithm = algorithm.replace(r"\KwIn{$\synd$, channel LLR $\llr$, noise estimate $\phat$}",
                              r"\KwIn{$\synd$, channel LLR $\llr$, global noise level $p$}")
algorithm = algorithm.replace(r"$(\gamma,\beta) \leftarrow \mathrm{FiLMGen}(\phat)$",
                              r"$(\gamma,\beta) \leftarrow \mathrm{FiLMGen}(p)$")
assert r"\phat" not in algorithm
fig0 = moved.index(r"\begin{figure*}[t]")
fig1 = moved.index(r"\end{figure*}") + len(r"\end{figure*}")
figure2 = moved[fig0:fig1].replace(r"""         additional input. Both stages are trained end-to-end through
         the BP unroll.}""", r"""         additional input. Both stages are trained end-to-end through
         the BP unroll. This variant is not evaluated in this paper.}""")
assert figure2 != moved[fig0:fig1]
appendix = r"""
\appendix

\section{Unevaluated variants}\label{app:variants}

The code base contains two further variants of the model. Neither is
evaluated in this paper, and no result in the main text depends on them.

\paragraph{FiLM conditioning.} Feature-wise Linear Modulation
(FiLM)~\cite{Perez2018FiLM} conditions each GNN layer on an external
context. A small generator network maps a global noise level to per-layer
affine parameters $(\gamma^{(\ell)}, \beta^{(\ell)})$, which modulate the
aggregated messages as $\gamma^{(\ell)} \odot \bvec{m}^{(\ell)} +
\beta^{(\ell)}$ (Fig.~\ref{fig:tannergnn_arch}). In the implementation the
generator receives the physical error rate $p$ of the shot, i.e.\ the true
global rate rather than an estimate from the syndrome. This input is one
scalar shared by the whole graph and contains no per-qubit reliability
information. Proposition~\ref{prop:invariance} shows that uniformly
rescaling the channel prior cannot change the hard decisions of unclipped
min-sum BP; it does not imply that a GNN cannot use a global scalar as
context.

\paragraph{Neural BP.} Differentiable BP generalises
Eqs.~\eqref{eq:vtc}--\eqref{eq:ctv} with learnable per-iteration weights on
the channel LLR, the variable-to-check messages and the check-to-variable
messages, $w^{(t)} = \mathrm{softplus}(w_\mathrm{raw}^{(t)})/\ln 2$.
Dividing by $\ln 2$ makes $w_\mathrm{raw}=0$ give $w=1$ exactly, so at
initialisation the unrolled network is standard min-sum BP with scaling
factor 0.8.

\paragraph{Interleaved pipeline.} Algorithm~\ref{alg:interleaved} and
Fig.~\ref{fig:interleaved_pipeline} give the interleaved pipeline. The GNN
intervenes twice: before BP to correct channel priors, and after $T_1$
iterations using the intermediate posterior information. Qubits with
$p_i^\mathrm{mid} \approx 0.5$ have low posterior confidence and can
receive a second learned correction. Unlike the model of
Sec.~\ref{sec:training}, both correction stages are trained jointly
through the differentiable BP unroll, so the loss is computed on BP's
output marginals. The FiLM and interleaved variants are trained on data
regenerated each epoch with $p\sim\mathrm{Uniform}[0.01,0.10]$, with the
focal loss on both components and a syndrome-consistency term, the binary
cross-entropy between the measured syndrome and
$\tfrac12\bigl(1-\prod_{j\in\mathcal N(c)}(1-2q_j)\bigr)$ for each check
$c$, with weight $0.1$.

""" + algorithm + "\n\n" + figure2 + "\n"
R(r"""\bibliographystyle{quantum}
\bibliography{references}
""", r"""\bibliographystyle{quantum}
\bibliography{references}
""" + appendix, "R3/M6 appendix")

assert r"sec:film" not in s and r"sec:neuralbp" not in s, "dangling refs to moved sections"

# =====================================================================
# Drift section (M10), filled from results/drift/drift.json
# =====================================================================
sys.path.insert(0, str(ROOT / "paper"))
import drift_text  # noqa: E402  (paper/drift_text.py)
dt = drift_text.build(MT)
for key in ("<<DRIFT_SECTION>>", "<<ABSTRACT_DRIFT>>", "<<CONCLUSION_DRIFT>>", "<<LIMITATION_DRIFT>>"):
    if s.count(key) != 1:
        sys.exit(f"FAIL placeholder {key}")
    s = s.replace(key, dt[key])
log.append("M10 drift text")

s = s.replace("decision, and 1 is stuck.", "decision, and one is stuck.")
DST.write_text(s)
print(f"wrote {DST.relative_to(ROOT)}: {len(log)} edits")
for t in log:
    print("  ", t)
