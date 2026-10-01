import sys
SRC='paper_v4/STAR_GNN_BB_v2_base.tex'
DST='paper_v4/STAR_GNN_BB_v4.tex'
s=open(SRC).read()
log=[]
def R(old,new,tag):
    global s
    n=s.count(old)
    if n!=1:
        print('FAIL',tag,'count',n); sys.exit(1)
    s=s.replace(old,new); log.append(tag)

# ---- Title (Nithin) ----
R(r'''\title{GNN-Augmented Belief Propagation for Bivariate Bicycle Codes: Gains, Baseline Sensitivity, and Calibration Side Information}''',
  r'''\title{GNN-Augmented BP Architectures for QLDPC Decoding}''','title')

# ---- Abstract: numbers only ----
R(r'''BP reduces the logical error rate (LER) from $0.0990$ to $0.0838$ on
8{,}000 held-out syndromes at $p=0.04$ and $\eta=20$, a $15.4\%$
relative reduction (paired McNemar $p<10^{-15}$). This gain does not
transfer to a stronger serial min-sum BP-OSD decoder: the learned
correction is statistically indistinguishable at lower tested error
rates and degrades performance at $p\geq 0.04$.''',
r'''BP reduces the logical error rate (LER) of that decoder by
$12.5\%\pm0.6\%$ on 20{,}000 held-out syndromes at $p=0.04$ and
$\eta=20$ (five training seeds; paired McNemar $p<10^{-36}$ for every
seed). This gain does not transfer to a stronger serial min-sum BP-OSD
decoder: the learned correction never improves it and is significantly
worse at $p\geq 0.03$. A paired failure-set analysis shows that the
failure sets of the two decoders are nested and that most shots the GNN
repairs are already decoded by BP-OSD.''','abstract')

# ---- Contributions ----
R(r'''On $\code{72}{12}{6}$ at $p=0.04$ and $\eta=20$,
\textsc{TannerGNN}+flooding BP reduces LER from $0.0990$ to $0.0838$
on 8{,}000 held-out syndromes, corresponding to a $15.4\%$ relative
reduction (McNemar $p<10^{-15}$). The model is trained only on
$p\in[0.02,0.03]$, so this test is also outside the training error-rate
range (Sec.~\ref{sec:gnn_positive}).''',
r'''On $\code{72}{12}{6}$ at $p=0.04$ and $\eta=20$,
\textsc{TannerGNN}+flooding BP reduces LER by $12.5\%\pm0.6\%$ on
20{,}000 held-out syndromes over five training seeds (McNemar
$p<10^{-36}$ for every seed). The training rates span
$p\in[0.005,0.045]$; at $p=0.06$, outside that range, the reduction is
$10.6\%\pm0.7\%$ (Sec.~\ref{sec:gnn_positive}).''','contrib1')
R(r'''same learned correction does not improve this stronger decoder and is
worse at the two highest tested error rates.''',
r'''same learned correction does not improve this stronger decoder and is
significantly worse from $p=0.03$ on. A paired failure-set analysis shows
that 93\% of BP-OSD failures are also BP failures and that most shots
the GNN repairs are already decoded by BP-OSD.''','contrib2')

# ---- Table 1 (Nithin: polynomials, names) ----
R(r'''\begin{table}[t]
\centering
\small
\caption{Bivariate bicycle codes studied in this work, with
         generator polynomials from \cite{Bravyi2024Memory}.}
\label{tab:codes}
\begin{tabular}{@{}lcccccc@{}}
\toprule
Code & $n$ & $k$ & $d$ & Rate & $\ell$ & $m$ \\
\midrule
$\code{72}{12}{6}$   &  72 & 12 &  6 & 1/6  &  6 &  6 \\
$\code{144}{12}{12}$ & 144 & 12 & 12 & 1/12 & 12 &  6 \\
$\code{288}{12}{18}$ & 288 & 12 & 18 & 1/24 & 12 & 12 \\
\bottomrule
\end{tabular}
\end{table}''',
r'''\begin{table*}[t]
\centering
\caption{Bivariate bicycle codes studied in this work, with
         generator polynomials from \cite{Bravyi2024Memory};
         $\Hmat_x$ and $\Hmat_z$ follow Eq.~\eqref{eq:bb} over
         $\F_2[x,y]/(x^\ell-1,\,y^m-1)$.}
\label{tab:codes}
\begin{tabular}{@{}llcccccccc@{}}
\toprule
Code & Name & $n$ & $k$ & $d$ & Rate & $\ell$ & $m$ & $A(x,y)$ & $B(x,y)$ \\
\midrule
$\code{72}{12}{6}$   & ---       &  72 & 12 &  6 & 1/6  &  6 &  6 & $x^3+y+y^2$   & $y^3+x+x^2$ \\
$\code{144}{12}{12}$ & gross     & 144 & 12 & 12 & 1/12 & 12 &  6 & $x^3+y+y^2$   & $y^3+x+x^2$ \\
$\code{288}{12}{18}$ & two-gross & 288 & 12 & 18 & 1/24 & 12 & 12 & $x^3+y^2+y^7$ & $y^3+x+x^2$ \\
\bottomrule
\end{tabular}
\end{table*}''','table1')

# ---- Noise model (Nithin) ----
R(r'''We use a $Z$-biased CSS component-noise model with $\eta = 20$:
\pending{Confirm the simulator's treatment of $Y$ errors. The equations
below specify only $X$- and $Z$-component rates; unless a $Y$ component
is explicitly sampled, this should not be called depolarizing noise.}
\begin{equation}
p_z = \frac{p\eta}{\eta+1}, \qquad p_x = \frac{p}{\eta+1}.
\label{eq:bias}
\end{equation}''',
r'''We use a $Z$-biased CSS component-noise model with $\eta = 20$:
\begin{equation}
p_z = \frac{p\eta}{\eta+1}, \qquad p_x = \frac{p}{\eta+1}.
\label{eq:bias}
\end{equation}
Each data qubit independently suffers a $Z$ flip with probability $p_z$
and an $X$ flip with probability $p_x$. No separate $Y$ channel is
sampled: a qubit carries a $Y$ error only if it receives both flips,
which happens with probability $p_xp_z\approx p^2/22$, so this is not
depolarizing noise. Syndromes are $s_x=\Hmat_x e_z$ and
$s_z=\Hmat_z e_x$, and the two components are decoded separately.''','noise')

# ---- TannerGNN description (major: must match code and Fig. 1) ----
R(r'''Each variable node $v_i$ is initialised with a 4-dimensional feature
vector $\bvec{x}_i = [\llr_i, f_i, 1, 0]$, where $f_i$ is the
fraction of neighbouring checks that are violated; each check node
$c_j$ receives $\bvec{x}_j = [0, s_j, 0, 1]$.

An input projection maps features to hidden dimension $d_h = 64$ via
$\bvec{h}_i^{(0)} = \mathrm{LayerNorm}(\Wmat_\mathrm{proj}\bvec{x}_i)$.''',
r'''Each node receives a 4-dimensional feature vector whose first entry is
a value and whose last three entries one-hot encode the node type: a
variable node $v_i$ receives $\bvec{x}_i=[\ell,1,0,0]$, where
$\ell=\ln\bigl((1-p)/p\bigr)$ is the prior LLR, and an $X$-check
($Z$-check) node $c_j$ receives $[s_j,0,1,0]$ ($[s_j,0,0,1]$), where
$s_j$ is its syndrome bit.

An input projection maps features to hidden dimension $d_h$ ($d_h=32$
for the model of Sec.~\ref{sec:gnn_positive}) via
$\bvec{h}_i^{(0)} = \mathrm{ReLU}(\mathrm{LayerNorm}(\Wmat_\mathrm{proj}\bvec{x}_i))$.''','gnn_features')
R(r'''  \mathrm{MLP}_{e,\tau(i,j)}^{(\ell)}\!\bigl(\bvec{h}_j^{(\ell)}\bigr),''',
  r'''  \mathrm{MLP}_{e,\tau(i,j)}^{(\ell)}\!\bigl(\bigl[\bvec{h}_i^{(\ell)},\bvec{h}_j^{(\ell)}\bigr]\bigr),''','gnn_msg')
R(r'''MLPs; a GRU update aggregates messages at each node. FiLM
         modulation (not shown) scales hidden activations with
         syndrome-derived conditioning at each layer.}''',
  r'''MLPs; a GRU update with a residual connection and layer
         normalisation follows at each node, and a readout MLP on the
         variable nodes gives the LLR corrections. In the FiLM variant
         (dashed), FiLM modulation scales the aggregated messages at
         each layer; the model of Sec.~\ref{sec:gnn_positive} does not
         use FiLM.}''','fig1_caption')


# ---- Training: Table 2 + loss + ranges (Nithin) ----
R(r"""\begin{table}[t]
\centering
\caption{Architecture and training hyperparameters.}
\label{tab:hparams}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}ll@{}}
\toprule
\multicolumn{2}{@{}l}{\textit{GNN Architecture}} \\
\midrule
Hidden dimension $d_h$          & 64 \\
Message-passing layers $L$      & 3 \\
Edge types                      & 2 ($\Hmat_x$, $\Hmat_z$) \\
Node update                     & GRU \\
Node feature dim.               & 4 \\
FiLM conditioning               & 1 (scalar $\phat$) \\
Parameters                      & \makecell[l]{GNN-BP: 154k\\FiLM: 183k\\Interleaved: 183k} \\
\midrule
\multicolumn{2}{@{}l}{\textit{Training}} \\
\midrule
Optimizer                       & AdamW \\
Learning rate                   & $10^{-4}$ \\
Weight decay                    & $10^{-4}$ \\
Schedule                        & Cosine + 2-epoch warmup \\
Batch size                      & 16 \\
Focal loss $(\alpha,\gamma)$    & $(0.25,2.0)$ \\
Syndrome loss weight            & $0.1$ \\
BP iters (training / eval)      & 10 (flooding) / 100 \\
Best epoch                      & \makecell[l]{GNN-BP: 5 / 17\\FiLM: 18\\Interleaved: 1} \\
\midrule
\multicolumn{2}{@{}l}{\textit{Reference BP-OSD baseline}} \\
\midrule
\texttt{bp\_method}             & \texttt{ms} (min-sum) \\
\texttt{ms\_scaling\_factor}    & $0.8$ \\
\texttt{schedule}               & \texttt{serial} (layered) \\
\texttt{max\_iter}              & 100 \\
OSD (where used)                & \texttt{osd\_cs}, order 10 \\
\bottomrule
\end{tabular}}
\end{table}

Training uses focal loss~\cite{Lin2017Focal} with $\alpha=0.25$,
$\gamma=2.0$ for class-imbalanced qubit errors, together with a syndrome
consistency term. The noise range is experiment dependent; in particular,
the positive held-out result of Sec.~\ref{sec:gnn_positive} uses training
rates $p\in[0.02,0.03]$. \pending{Reconcile this with the earlier training
configuration $p\sim\mathrm{Uniform}[0.01,0.10]$ and report the data
range, number of samples per epoch, and checkpoint-selection rule for
each model variant.} Key hyperparameters are summarized in
Table~\ref{tab:hparams}.""",
r"""\begin{table}[t]
\centering
\footnotesize
\caption{Architecture and training configuration of the
         \textsc{TannerGNN} model of Secs.~\ref{sec:gnn_positive}
         and~\ref{sec:mismatch}.}
\label{tab:hparams}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}ll@{}}
\toprule
\multicolumn{2}{@{}l}{\textit{GNN architecture}} \\
\midrule
Layers $L$ / width $d_h$       & 3 / 32 \\
Edge types                      & 2 ($\Hmat_x$, $\Hmat_z$) \\
Node update                     & GRU, residual, layer norm \\
Node feature dim.               & 4 \\
FiLM / attention                & not used \\
Parameters                      & 39{,}329 \\
\midrule
\multicolumn{2}{@{}l}{\textit{Training}} \\
\midrule
Data                            & \makecell[l]{8{,}000 shots, sinusoidal drift,\\$p\in[0.005,0.045]$} \\
Loss                            & \makecell[l]{focal, $(\alpha,\gamma)=(0.25,2.0)$, on $e_z$;\\no syndrome term} \\
Optimizer                       & \makecell[l]{AdamW, lr $2\times10^{-3}$,\\weight decay $10^{-4}$} \\
Batch size / epochs             & 64 / 20, gradient clipping 1.0 \\
Checkpoint                      & \makecell[l]{fewest failures on\\4{,}000 validation shots} \\
Training seeds                  & 5 \\
BP in training loop             & \makecell[l]{flooding min-sum,\\scaling 0.8, 10 iter.} \\
\midrule
\multicolumn{2}{@{}l}{\textit{Reference BP-OSD baseline}} \\
\midrule
\texttt{bp\_method}             & \texttt{ms} (min-sum) \\
\texttt{ms\_scaling\_factor}    & $0.8$ \\
\texttt{schedule}               & \texttt{serial} (layered) \\
\texttt{max\_iter}              & 100 \\
OSD (where used)                & \texttt{osd\_cs}, order 10 \\
\bottomrule
\end{tabular}
\end{table}

For a shot with prior LLR $\ell=\ln\bigl((1-p)/p\bigr)$ on every qubit,
the GNN outputs a correction $\Delta_j$ for each qubit $j$, giving the
corrected error probability $q_j=\sigma\bigl(-(\ell+\Delta_j)\bigr)$.
The model of Sec.~\ref{sec:gnn_positive} is trained with the focal
loss~\cite{Lin2017Focal} against the true $Z$ error $e_z$,
\begin{equation}
\mathcal{L}=-\frac{1}{Bn}\sum_{b=1}^{B}\sum_{j=1}^{n}
\alpha_{t}\,(1-q_{t})^{\gamma}\log q_{t},
\label{eq:loss}
\end{equation}
where $q_t=q_j$ and $\alpha_t=\alpha$ if $e_{z,j}=1$, and $q_t=1-q_j$
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
the configuration.""",'training')

# ---- Fig. 3 caption (figure redrawn) ----
R(r"""\caption{Experimental evaluation pipeline. Syndromes are sampled under
         static or drift noise models; each decoder receives identical
         syndrome batches to enable McNemar paired testing. Oracle
         experiments substitute true per-qubit rates for the uniform
         mean prior.}""",
r"""\caption{Training and evaluation pipeline of the GNN experiments. The
         GNN is trained through 10-iteration flooding BP and evaluated on
         separately seeded held-out shots in Regime~A (flooding BP) and
         Regime~B (serial BP-OSD); each decoder receives identical
         syndrome batches to enable McNemar paired testing.}""",'fig3_caption')

# ---- Statistical protocol: logical failure (Nithin) + seeds ----
R(r"""\subsection{Statistical protocol}

For absolute LER comparisons""",
r"""\subsection{Statistical protocol}\label{sec:metrics}

\paragraph{Logical failure.} For each shot let $\hat{e}_z$ and
$\hat{e}_x$ be the decoder's hard decisions and $r_z=e_z\oplus\hat{e}_z$,
$r_x=e_x\oplus\hat{e}_x$ the residual errors. A shot is a logical failure
if (i) the estimate does not reproduce the measured syndrome,
$\Hmat_x\hat{e}_z\neq s_x$ or $\Hmat_z\hat{e}_x\neq s_z$, or (ii) the
syndrome is reproduced but the residual acts nontrivially on the code
space, $\mathbf{L}_x r_z\neq\mathbf{0}$ or $\mathbf{L}_z r_x\neq\mathbf{0}$
over $\F_2$, where the rows of $\mathbf{L}_x$ ($\mathbf{L}_z$) span the
$k=12$ independent logical operators that detect $Z$ ($X$) errors.
Condition (i) is needed because a syndrome-inconsistent residual is not a
logical operator, so its parity against fixed logical representatives is
not meaningful; non-converged BP outputs therefore always count as
failures. The $X$ and $Z$ components are combined by logical OR, so the
reported LER is the probability that at least one of the 12 logical
qubits suffers a logical error of any Pauli type in a shot. At circuit
level the same rule is applied to the detector error model: a shot fails
if the decoded fault set does not reproduce the detection events, or if
it does and predicts the wrong observable flips.

For absolute LER comparisons""",'logical_failure')
R(r"""Random seeds are fixed at $s=42$ for syndrome generation and $s+999$ for
validation splits. Unless otherwise stated, the reported learned-decoder
result is based on a single training seed; its confidence interval and
McNemar test therefore quantify test-set sampling uncertainty conditional
on that trained checkpoint, not training-to-training variability. All
training and evaluation used CPU only; no GPU was available for this work.
Representative per-epoch training times are $\sim\!17$\,s
(154k-parameter GNN-BP) and $\sim\!37$\,s (183k-parameter
interleaved model).""",
r"""Training, validation and test data are drawn with distinct random
seeds, so no evaluation shot shares a random stream with a training
shot, and learned-decoder results are reported over five training seeds
(initialisation and minibatch order). All training and evaluation used
CPU only; no GPU was available for this work. One training epoch of the
39k-parameter model of Sec.~\ref{sec:gnn_positive} takes about 16\,s.""",'seeds')

# ---- Table 3 + 100k decomposition (Nithin: shots, exact config) ----
R(r"""\caption{Effect of BP-OSD configuration on $\code{288}{12}{18}$,
         $p=0.04$, $\eta=20$, 5{,}000 identical shots. CIs in brackets.}
\label{tab:smoking_gun}
\begin{tabular}{@{}llllcc@{}}
\toprule
Config & Schedule & Method & Iter & LER & Errors \\
\midrule
Original BP     & parallel & prod-sum & 30  & 0.0506 [0.045,0.057] & 253 \\
Reference BP    & serial   & ms(0.8)  & 100 & 0.0014 [0.001,0.003] &   7 \\
Original BP-OSD & parallel & prod-sum & 30  & 0.0346 [0.030,0.040] & 173 \\
Ref.\ BP-OSD    & serial   & ms(0.8)  & 100 & 0.0002 [0.000,0.001] &   1 \\""",
r"""\caption{Effect of BP-OSD configuration on $\code{288}{12}{18}$,
         $p=0.04$, $\eta=20$, 100{,}000 identical shots, priors $p_z$ and
         $p_x$ of Eq.~\eqref{eq:bias}. 95\% Wilson intervals in brackets.}
\label{tab:smoking_gun}
\begin{tabular}{@{}llllcc@{}}
\toprule
Config & Schedule & Method & Iter & LER & Errors \\
\midrule
Original BP     & parallel & prod-sum & 30  & 0.0569 [0.0555,0.0583] & 5{,}689 \\
Reference BP    & serial   & ms(0.8)  & 100 & 0.0017 [0.0015,0.0020] &    174 \\
Original BP-OSD & parallel & prod-sum & 30  & 0.0398 [0.0386,0.0411] & 3{,}983 \\
Ref.\ BP-OSD    & serial   & ms(0.8)  & 100 & 0.00034 [0.00024,0.00048] &   34 \\""",'table3')
R(r"""configuration changes the finite-length result. In the 5{,}000-shot
pilot, the original parallel product-sum BP-OSD configuration reduces
LER from $0.0506$ to $0.0346$ relative to the corresponding plain BP,
but remains far worse than the serial min-sum configurations. The
reference serial min-sum BP-OSD produces only one logical error in this
sample, so the value $0.0002$ should be interpreted as a preliminary
estimate rather than a precisely resolved LER.

A separate $N=100{,}000$-shot experiment is shown in
Fig.~\ref{fig:baseline_decomp}. In that experiment, changing to the
serial schedule and then adding OSD-CS-10 yields an overall
approximately $40\times$ reduction relative to the flooding comparator.
\pending{State the exact BP method, scaling factor, iteration count, and
stopping rule for the 100k flooding comparator. These must match the
serial configuration except for the factor being isolated; otherwise the
$6.7\times$ schedule decomposition is confounded.}""",
r"""configuration changes the finite-length result. The original parallel
product-sum BP-OSD configuration reduces LER from $0.0569$ to $0.0398$
relative to the corresponding plain BP, but remains far worse than the
serial min-sum configurations. The reference serial min-sum BP-OSD
records 34 logical errors in $10^5$ shots, so its value is preliminary
by the criterion of Sec.~\ref{sec:metrics}.

A separate $N=100{,}000$-shot experiment (Fig.~\ref{fig:baseline_decomp})
changes one factor at a time. The flooding comparator is min-sum BP with
scaling factor $0.8$, at most 100 iterations, stopping at the first
iteration whose hard decision satisfies the syndrome, with a parallel
(flooding) schedule; the second decoder is identical except for the
serial schedule; the third adds OSD-CS-10. The LER falls from
$1.28\times10^{-2}$ to $1.45\times10^{-3}$ ($8.8\times$, paired
$n_{10}/n_{01}=1138/3$) and then to $2.7\times10^{-4}$ ($5.4\times$,
$118/0$), about $47\times$ overall.""",'decomp_text')
R(r"""\caption{Baseline decomposition on $\code{288}{12}{18}$, $p=0.04$,
         $\eta=20$, 100k shots. The figure is intended to isolate the
         effect of serial scheduling and subsequent OSD-CS-10; the exact
         flooding-BP comparator configuration must be reported in the
         text before the multiplicative factors are interpreted causally.}""",
r"""\caption{Baseline decomposition on $\code{288}{12}{18}$, $p=0.04$,
         $\eta=20$, 100k identical shots. Min-sum BP (scaling 0.8, at most
         100 iterations) with flooding and then serial schedule, then
         serial BP-OSD-CS-10. Wilson 95\% CI error bars; labels give the
         factor gained at each step.}""",'decomp_caption')

# ---- Invariance paragraph (major: v2 claims did not reproduce) ----
R(r"""We observed the expected invariance in the implemented decoder:
$1{,}740/1{,}740$ bit-identical outputs when changing the assumed prior
from $p=0.015$ to $p=0.0286$ on $\code{288}{12}{18}$, and
$1{,}000/1{,}000$ bit-identical outputs for $p=0.04$ versus $p=0.02$ in
the tested min-sum setting. The same 1{,}000-shot test was also identical
for the tested sum-product configuration; because Proposition~\ref{prop:invariance}
does not apply to the tanh nonlinearity, we report that observation as
empirical rather than structural.""",
r"""We checked the proposition on 2{,}000 shots each. With unclipped
min-sum BP (serial or flooding, 100 iterations), changing the assumed
uniform rate from $p=0.015$ to $p=0.0286$ on $\code{288}{12}{18}$, or from
$p=0.04$ to $p=0.02$ on $\code{72}{12}{6}$, leaves every hard decision
unchanged (2{,}000/2{,}000). Two cases fall outside the proposition, and
the invariance fails there: our PyTorch min-sum implementation clamps
messages to $\pm20$ and changes 18 of 2{,}000 decisions on
$\code{288}{12}{18}$, and sum-product BP changes 25--43 of 2{,}000.""",'invariance')


# ---- Sec. 5.3 (Nithin: seeds, ranges, logical failure) ----
R(r"""We evaluated the GNN against flooding BP---the decoder it was
trained against---on 8{,}000 held-out syndromes.""",
r"""We evaluated the GNN against flooding BP---the decoder it was
trained against---on separately seeded held-out syndromes: 4{,}000
validation shots at $p=0.04$ select the checkpoint over 20 epochs, and
test sets of 20{,}000 shots at $p=0.04$ (inside the training range) and
10{,}000 shots at $p=0.06$ (outside it) are each scored once on that
checkpoint. Training is repeated with five seeds.""",'s53_intro')
R(r"""\caption{GNN-BP vs.\ flooding BP on $\code{72}{12}{6}$, $\eta=20$,
         8{,}000 held-out test syndromes. GNN trained for 20 epochs
         on separate 8{,}000 training syndromes at $p\in[0.02,0.03]$.
         Best checkpoint at epoch 5 by validation LER.}
\label{tab:gnn_flooding}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lcccc@{}}
\toprule
 & Flooding BP & GNN-BP & $n_{10}$ / $n_{01}$ & McNemar $p$ \\
\midrule
$p=0.04$, ep. 5 & 0.0990 & 0.0838 & 170 / 48 & $\approx 0$ \\
$p=0.04$, ep. 1 & 0.0990 & 0.0975 & 91 / 79  & 0.399 \\
\bottomrule
\end{tabular}}""",
r"""\caption{GNN-BP vs.\ flooding BP on $\code{72}{12}{6}$, $\eta=20$,
         20{,}000 ($p=0.04$) and 10{,}000 ($p=0.06$) held-out test
         syndromes, five training seeds (mean $\pm$ s.d.). $n_{10}$
         ($n_{01}$): shots only BP (only GNN-BP) fails. BP is flooding
         min-sum.}
\label{tab:gnn_flooding}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lcc@{}}
\toprule
 & $p=0.04$ & $p=0.06$ \\
\midrule
BP, 10 iter.                 & 0.1077 & 0.2867 \\
GNN-BP, 10 iter.             & $0.0942\pm0.0006$ & $0.2564\pm0.0019$ \\
\quad relative reduction     & $12.5\%\pm0.6\%$ & $10.6\%\pm0.7\%$ \\
\quad $n_{10}$ / $n_{01}$    & 313--370 / 64--95 & 343--394 / 66--80 \\
BP, 100 iter.                & 0.0863 & 0.2403 \\
Serial BP-OSD                & 0.0707 & 0.2138 \\
\bottomrule
\end{tabular}""",'table4')
R(r"""At the epoch-5 checkpoint, GNN-BP reduces LER by $15.4\%$ relative to
flooding BP on 8{,}000 held-out syndromes at $p=0.04$
($n_{10}=170$, $n_{01}=48$, McNemar $\chi^2=67.16$,
$p<10^{-15}$). The test point lies outside the training range
$p\in[0.02,0.03]$, so the result demonstrates transfer to a higher
physical error rate for this checkpoint. Because only one training seed
is reported, we do not interpret the paired test as evidence for
training-to-training reproducibility. A gradient audit""",
r"""Over five training seeds, GNN-BP reduces LER relative to flooding BP by
$12.5\%\pm0.6\%$ at $p=0.04$ and by $10.6\%\pm0.7\%$ at $p=0.06$, outside
the training range (McNemar $p<10^{-36}$ for every seed). The gain comes
mostly from convergence: shots on which BP does not reproduce the
syndrome fall from 1{,}740 to 838--975 of 20{,}000. Flooding BP with 100
iterations and serial BP-OSD both remain better than GNN-BP. A gradient audit""",'s53_text')
R(r"""trained on 1{,}000 fixed syndromes for 50 epochs achieved a $14\%$ LER
reduction on those same syndromes (McNemar $p=0.004$, $n_{10}=17$,
$n_{01}=3$). This experiment tests the ability of the architecture and
optimizer to fit the supplied examples; it is not a generalization
result. The interleaved model reached its best validation checkpoint at
epoch 1 and did not train successfully, so no performance conclusion
about that architecture is drawn from the current run (see Limitations).

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{fig_F5_interleaved_training}
\caption{Interleaved model training curves (183k parameters).
         The best validation checkpoint occurs at epoch~1 and the
         validation loss increases thereafter. The current run does
         not provide a trained interleaved decoder for comparison.}
\label{fig:interleaved_training}
\end{figure}""",
r"""trained on 1{,}000 fixed syndromes for 50 epochs reduced the failure
count on those same syndromes from 105 to 93 ($11\%$; McNemar
$p=0.0095$, $n_{10}=15$, $n_{01}=3$). This experiment tests the ability of
the architecture and optimizer to fit the supplied examples; it is not a
generalization result. The interleaved architecture of
Sec.~\ref{sec:neuralbp} is not evaluated in this work.""",'capacity_F5')

# ---- Sec. 5.4 (Nithin: shots, seeds, failure-set analysis) ----
R(r"""\caption{GNN-BP vs.\ serial BP-OSD on $\code{72}{12}{6}$, $\eta=20$,
         8{,}000 shots. Decoders evaluated on identical syndromes.}
\label{tab:gnn_null}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lcccl@{}}
\toprule
$p$ & Serial BP-OSD & GNN+OSD & $p_\mathrm{McN}$ & Verdict \\
\midrule
0.020 & 0.0046 & 0.0050 & 0.68 & tie \\
0.030 & 0.0377 & 0.0399 & 0.63 & tie \\
0.040 & 0.0831 & 0.0876 & 0.033 & GNN \emph{worse} \\
0.050 & 0.1515 & 0.1555 & $3.6\!\times\!10^{-5}$ & GNN \emph{worse} \\
\bottomrule
\end{tabular}}""",
r"""\caption{GNN-BP vs.\ serial BP-OSD on $\code{72}{12}{6}$, $\eta=20$,
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
\end{tabular}""",'table5')
R(r"""\caption{GNN+OSD vs.\ serial BP-OSD LER on $\code{72}{12}{6}$,
         $\eta=20$, 8{,}000 shots. Red $\times$ marks operating points
         where GNN is statistically worse (McNemar $p<0.05$).
         Wilson 95\% CI error bars.}""",
r"""\caption{GNN+OSD vs.\ serial BP-OSD LER on $\code{72}{12}{6}$,
         $\eta=20$, 20{,}000 shots per point; GNN+OSD is the mean over
         five training seeds. Red $\times$ marks operating points where
         every seed is statistically worse (McNemar $p<0.05$).
         Wilson 95\% CI error bars.}""",'fig_F4_caption')
R(r"""the GNN-corrected decoder is statistically indistinguishable from serial
BP-OSD at the two lower error rates and significantly \emph{worse} at
the two higher rates.""",
r"""the GNN-corrected decoder never improves on serial BP-OSD: at $p=0.02$
one seed of five is significantly worse, and from $p=0.03$ on 14 of 15
seed--error-rate pairs are significantly \emph{worse}.""",'s54_text')
R(r"""The experiment establishes a transfer failure, but it does not by itself
identify the residual-failure mechanism. The GNN was optimized through a""",
r"""The GNN was optimized through a""",'s54_mech')
R(r"""changed. At the same time,
claims that the two decoders fail on ``structurally different'' trapping
sets require an explicit paired failure-overlap or trapping-set analysis,
which is not included here. \pending{If available, add the Jaccard/overlap
statistics of the two failure sets and classify representative failures;
otherwise keep the explanation at the level of training/evaluation
mismatch.} Training directly""",
r"""changed.

\begin{table}[t]
\centering
\caption{Paired failure sets on $\code{72}{12}{6}$, $p=0.04$, $\eta=20$,
         20{,}000 identical held-out shots. BP: 10-iteration flooding BP.
         GNN rows give the range over five training seeds. ``Non-conv.'':
         the estimate does not reproduce the syndrome.}
\label{tab:failure_sets}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcc@{}}
\toprule
Decoder & Failures & Non-conv. \\
\midrule
BP, 10 iter.            & 2{,}155 & 1{,}740 \\
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
\makecell[l]{GNN+BP-OSD vs.\ BP-OSD:\\newly failed / rescued} & \multicolumn{2}{c}{237--282 / 206--254} \\
\bottomrule
\end{tabular}
\end{table}

A paired failure-set analysis (Table~\ref{tab:failure_sets}) shows that
the two decoders do not fail on disjoint sets: every shot that
100-iteration BP fails is also failed by 10-iteration BP, and 93\% of
BP-OSD's failures are BP failures. What separates them is mainly
convergence: 81\% of BP failures are shots on which BP does not reach a
syndrome-consistent estimate, whereas BP-OSD always returns one. Of the
313--370 shots the GNN fixes for BP, 91--93\% are non-converged BP shots
and only 31--36\% are shots that BP-OSD fails, so the GNN's gains lie
mostly where BP-OSD already succeeds. When the same corrections are
passed to BP-OSD they reshuffle its failures (237--282 newly failed vs.\
206--254 rescued shots), almost entirely among shots on which BP itself
fails. The same pattern holds at $p=0.06$, where 96\% of BP-OSD failures
are BP failures. Training directly""",'failure_sets')

# ---- Sec. 5.5 (major: noise convention; Nithin: shots) ----
R(r"""Under spatially non-uniform per-qubit noise ($p_i \sim
\mathrm{LogNormal}(\log p_\mathrm{mean}, \sigma)$, fresh per shot),
a decoder knowing the true per-qubit rates substantially outperforms
a mean-prior decoder.""",
r"""Under spatially non-uniform per-qubit noise, each data qubit has rate
$p_i=p\,e^{\sigma z_i-\sigma^2/2}$ with $z_i\sim\mathcal N(0,1)$ drawn
fresh every shot (capped at 0.5), so the mean rate is $p$ for every
$\sigma$. A decoder knowing the true per-qubit rates (serial min-sum
BP-OSD-CS-10 with the true $p_i$ as priors) substantially outperforms a
decoder using the uniform mean prior $p$.""",'s55_noise')
R(r"""\caption{Oracle calibration gap (percentage LER reduction from
         per-qubit vs.\ mean-prior BP-OSD) at $\sigma=1.0$,
         50k shots ($\code{72}{12}{6}$, $\code{144}{12}{12}$)
         and 30k shots ($\code{288}{12}{18}$).""",
r"""\caption{Oracle calibration gap (percentage LER reduction from
         per-qubit vs.\ mean-prior BP-OSD), 50k identical shots per
         point.""",'fig_F2_caption')
R(r"""\caption{Oracle gap: percentage LER reduction from per-qubit
         over mean-prior BP-OSD. All McNemar $p \ll 10^{-3}$.
         Parenthetical counts are mean-prior error events.}""",
r"""\caption{Oracle gap: percentage LER reduction from per-qubit
         over mean-prior BP-OSD, 50{,}000 identical shots per cell, at
         $p=0.04$, $0.06$, $0.07$ for the three codes. All McNemar
         $p<10^{-20}$. Parenthetical counts are mean-prior error events.}""",'table6_caption')
R(r"""$\code{72}{12}{6}$   (6)  & $+11\%$ (3457) & $+30\%$ (3420) & $+66\%$ (2988) \\
$\code{144}{12}{12}$ (12) & $+16\%$ (276)  & $+38\%$ (268)  & $+74\%$ (2604) \\
$\code{288}{12}{18}$ (18) & $+18\%$ (922)  & $+48\%$ (524)  & $+87\%$ (1750) \\""",
r"""$\code{72}{12}{6}$   (6)  & $+11\%$ (3508) & $+29\%$ (3432) & $+65\%$ (3438) \\
$\code{144}{12}{12}$ (12) & $+12\%$ (3165) & $+35\%$ (3203) & $+80\%$ (2936) \\
$\code{288}{12}{18}$ (18) & $+13\%$ (3338) & $+41\%$ (3307) & $+88\%$ (2786) \\""",'table6')
R(r"""reaching
$87\%$ on $\code{288}{12}{18}$""",r"""reaching
$88\%$ on $\code{288}{12}{18}$""",'s55_88')
R(r"""than the mean prior in this regime. A GNN that attempts to estimate
per-qubit rates from a single syndrome is significantly
\emph{worse} than the mean prior at $\sigma=1.0$ (GNN LER = 0.069
vs.\ mean LER = 0.064, McNemar $p = 2.7\times10^{-4}$)---not
because the architecture is broken, but because the task is
information-theoretically infeasible.""",r"""than the mean prior in this regime.""",'single_shot')


# ---- Sec. 5.7 circuit level (Nithin: full spec, regenerate Table 8) ----
i=s.index(r"""\pending{Specify the complete circuit-level noise model""")
j=s.index(r"""calibration side information is irrelevant at circuit level.""")+len(r"""calibration side information is irrelevant at circuit level.""")
s=s[:i]+r"""\paragraph{Circuit and noise model.} We simulate a $Z$-basis memory
experiment on $\code{72}{12}{6}$ with 6 rounds of syndrome extraction
(72 data qubits, 36 $X$-check and 36 $Z$-check ancillas). Data qubits are
prepared in $\ket{0}$; each round measures all $X$-checks and then all
$Z$-checks with dedicated ancillas, using CNOT layers from a greedy
conflict-free schedule; the data qubits are finally measured in the $Z$
basis. Detectors are (i) the 36 $Z$-checks of round 1, which are
deterministic for the prepared state, (ii) the XOR of consecutive
measurements of every check in rounds 2--6, and (iii) the XOR of the last
$Z$-check outcome with the parity of the measured data qubits in its
support, giving 432 detectors and 12 logical observables. Noise is a
biased single-qubit Pauli channel with $p_X=p/(\eta+1)$,
$p_Z=p\,\eta/(\eta+1)$, $p_Y=0$, $\eta=20$, applied to every qubit after
each one- and two-qubit gate and to every idle qubit at each time step;
measurements and resets flip with probability $p$. Under spatially
non-uniform noise each qubit $q$ (data and ancilla) has rate
$p_q=p\,e^{\sigma z_q-\sigma^2/2}$, $z_q\sim\mathcal N(0,1)$, $\sigma=1$,
on all of its fault locations, so the mean rate is $p$ as in
Sec.~\ref{sec:oraclegap}.

\paragraph{Decoders.} We decode the detector error model (4{,}068 fault
mechanisms) with serial min-sum BP-OSD (scaling 0.8, 100 iterations,
OSD-CS order 10), as in Table~\ref{tab:hparams}. The mean-prior decoder
uses the uniform rate $p$; the oracle uses the true $p_q$. We draw
<<PROFILES_LOW>> rate profiles at $p\le0.003$ and <<PROFILES_HIGH>> at
$p=0.01$, with 500 shots per profile. Failures follow
Sec.~\ref{sec:metrics}.

\begin{table}[t]
\centering
\caption{Circuit-level oracle gap on $\code{72}{12}{6}$ (6 rounds,
         $\eta=20$, $\sigma=1$). $n_{10}$ ($n_{01}$): shots only the
         mean-prior decoder (only the oracle) fails.}
\label{tab:circuit}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lccccc@{}}
\toprule
$p$ & Shots & Mean prior & Oracle & $n_{10}/n_{01}$ & $p_\mathrm{McN}$ \\
\midrule
<<CIRC_ROWS>>
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{fig_F6_circuit_oracle}
\caption{Circuit-level oracle gap on $\code{72}{12}{6}$, 6 rounds,
         $\eta=20$, $\sigma=1$. Points: LER with 95\% Wilson intervals;
         open triangles: 95\% upper bound when no failures were
         observed. Labels: discordant pairs and McNemar $p$.}
\label{fig:circuit_null}
\end{figure}

<<CIRC_TEXT>>"""+s[j:]
log.append('circuit')

# ---- Discussion ----
R(r"""\textsc{TannerGNN} achieves a $15.4\%$ LER reduction over the flooding
BP decoder used in its training loop. The paired held-out test supports a
real difference for this trained checkpoint, while gradient diagnostics
show that the network produces non-trivial corrections. Robustness to
training initialization has not yet been measured, so we do not treat the
single-seed result as a reproducibility claim.""",
r"""\textsc{TannerGNN} reduces the LER of the flooding BP decoder used in
its training loop by $12.5\%\pm0.6\%$ over five training seeds, and
gradient diagnostics show that the network produces non-trivial
corrections. The gain comes mainly from helping BP converge, and
100-iteration BP and serial BP-OSD remain stronger than the corrected
decoder.""",'disc1')
R(r"""This is consistent with the
training/evaluation mismatch, but the residual failure sets have not yet
been characterized well enough to claim a trapping-set mechanism.""",
r"""This is consistent with the
training/evaluation mismatch: the failure sets of BP and BP-OSD are
nested, and the GNN's repairs fall mostly on shots BP-OSD already
decodes (Table~\ref{tab:failure_sets}).""",'disc2')
R(r"""places our $15.4\%$ flooding-BP gain""",r"""places our $12.5\%$ flooding-BP gain""",'priorwork')
R(r"""variant follows the same broad design principle, but the checkpoint
reported here does not train successfully. We therefore do not claim a
competitive result for that architecture.""",
r"""variant follows the same broad design principle, but its decoding
performance is not evaluated here, so we do not claim a competitive
result for that architecture.""",'gong')

# ---- Limitations ----
R(r"""\item The primary $15.4\%$ GNN result uses one training seed. Statistical
      uncertainty over independent training runs has not been measured.
""","",'lim_seed')
R(r"""\item The interleaved model reached its best validation checkpoint at epoch 1,
      after which the validation loss increased. The current run is therefore
      insufficient to evaluate the interleaved architecture.""",
r"""\item The interleaved and FiLM architectures are described but their
      decoding performance is not evaluated.""",'lim_interleaved')

# ---- Conclusion ----
R(r"""10-iteration flooding BP reduces the held-out LER by $15.4\%$ at
$p=0.04$ and $\eta=20$. The same learned correction does not improve the
serial min-sum BP-OSD reference decoder and increases LER at the two
highest tested error rates.""",
r"""10-iteration flooding BP reduces the held-out LER of that decoder by
$12.5\%\pm0.6\%$ at $p=0.04$ and $\eta=20$ over five training seeds. The
same learned correction does not improve the serial min-sum BP-OSD
reference decoder and is significantly worse from $p=0.03$ on; the shots
it repairs are largely ones BP-OSD already decodes.""",'concl1')
R(r"""noise. The current cross-code experiments do not isolate how this gain
scales with distance, and the circuit-level study does not yet have
sufficient statistical resolution.""",
r"""noise, and <<CONCL_CIRC>> The current cross-code experiments do not
isolate how this gain scales with distance.""",'concl3')

# ---- Data availability (repo moved; scripts) ----
R(r"""Source code, trained checkpoints, all datasets, and the audit scripts
that produced the findings of Secs.~\ref{sec:baseline}
and~\ref{sec:invariance} are available at
\url{https://github.com/anish-chedalla/adaptive_gnn}.
Syndrome data were generated using Stim~\cite{Gidney2021Stim}.""",
r"""Source code, trained checkpoints, datasets and the analysis scripts
behind every table and figure except the drift experiment
(Sec.~\ref{sec:drift}) are available at
\url{https://github.com/kondshk/adaptive_gnn}. Code-capacity data are
sampled directly with NumPy; circuit-level data are generated with
Stim~\cite{Gidney2021Stim}.""",'availability')


# ---- Remark 1: consistency with corrected feature description ----
R(r"""both $\llr_{z,i}$ and $\llr_{x,i}$, and computes GNN input
features from their average.""",r"""both $\llr_{z,i}$ and $\llr_{x,i}$, and computes GNN input
features from a single prior LLR $\ell$.""",'remark1')


# ---- Circuit-level numbers (results/circuit_v2/oracle_v4_*.json) ----
FILL = {
 '<<PROFILES_LOW>>': '20',
 '<<PROFILES_HIGH>>': '40',
 '<<CIRC_ROWS>>': r"""0.001 & 10{,}000 & 0 & 0 & 0/0 & --- \\
0.003 & 10{,}000 & 0 & 0 & 0/0 & --- \\
0.01  & 20{,}000 & $1.35\%$ & $0.60\%$ & 177/28 & $5\!\times\!10^{-25}$ \\""",
 '<<CIRC_TEXT>>': r"""Knowing the per-qubit rates also helps at circuit level
(Table~\ref{tab:circuit}, Fig.~\ref{fig:circuit_null}). At $p=0.01$ the
oracle reduces the logical error rate from $1.35\%$ to $0.60\%$ ($55\%$;
McNemar $p=4.8\times10^{-25}$), and it is better on 38 of 40 independent
rate profiles (sign test $p=1.5\times10^{-9}$), so the result is not
driven by a few extreme profiles. At $p=0.001$ and $p=0.003$ neither
decoder fails in $10^4$ shots, so these operating points cannot resolve a
gap. The circuit-level gap is somewhat smaller than the code-capacity gap
on the same code ($65\%$ at $\sigma=1$, Table~\ref{tab:oracle}), at a
different operating point, but it does not vanish.""",
 '<<CONCL_CIRC>>': r"""under circuit-level noise it reduces BP-OSD's LER by $55\%$ at
$p=0.01$.""",
}
for k, v in FILL.items():
    if s.count(k) != 1:
        print('FAIL fill', k, s.count(k)); sys.exit(1)
    s = s.replace(k, v)
R(r"""         mean-prior decoder (only the oracle) fails.}""",
  r"""         mean-prior decoder (only the oracle) fails. Zero failures in
         $10^4$ shots means LER $<3.8\times10^{-4}$ (95\% upper bound).}""",'circ_caption')
log.append('circuit_fill')

open(DST,'w').write(s)
print('ok', log)
