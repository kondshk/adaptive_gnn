# Revision v5: what changed for each review item

This file answers the October 2026 review memo item by item. It starts from
the PDF that was reviewed, a 14-page build dated 3 October. Its source is
reconstructed exactly as `paper/reviewed/STAR_GNN_BB_reviewed.tex`, and the
text was checked word for word against that PDF.

- `paper/build_v5.py` applies every change as an exact replacement. Each
  change carries a tag naming the item it answers.
- `paper/STAR_GNN_BB_reviewed_to_v5_tracked_changes.pdf` shows the changes
  in the text.
- Table values are not typed by hand. `make_tables.py` recomputes them from
  the stored per-shot outcomes, and `python make_tables.py --check
  paper/STAR_GNN_BB_v5.tex` confirms that every generated row is in the
  manuscript.

Each item below has four parts:
- **Asked:** what the memo asked for.
- **Evidence:** what the code and data show.
- **Changed:** what was edited, in the manuscript and in the code.
- **Open:** what is still unresolved.

The first pass of these checks was AI-assisted (Claude Code).

## MUST FIX

### M1: training graph and loss
- **Asked:**
  - Trace the code and write the exact computation graph.
  - Remove any claim that the GNN is trained through BP if gradients do not pass through BP.
- **Evidence:**
  - The loss is `focal_loss(sigmoid(-(prior + d)), b.y)` (`audit_headline_v2.py`, `run_seed`).
  - So the graph is (l, s) → TannerGNN → Δ → q = σ(−(l+Δ)) → focal loss vs e_z. BP is not in the loss.
  - BP enters only through checkpoint selection: after each epoch, `score("val_p04")` decodes 4,000 validation shots with 10-iteration flooding BP, and the epoch with the fewest failures is kept.
  - The reviewed text said or implied otherwise in many places, including:
    - "a model trained with 10-iteration flooding BP" (abstract) and "trained through 10-iteration flooding BP" (conclusion, Fig. 3 caption);
    - "BP in training loop" (Table 2);
    - "the decoder represented in the training computation graph" (Regime A);
    - "optimized through a 10-iteration flooding-BP computation graph" (Sec. 5.4);
    - "backpropagation through the BP unroll" (Fig. 3).
- **Changed:**
  - Sec. 3.2 now gives the computation graph explicitly and says that BP is used only for model selection.
  - Table 2: the "BP in training loop" row became "BP for model selection … not in the loss", and the Loss row names q_j.
  - The training/evaluation pipeline figure (Fig. 2, previously Fig. 3) was redrawn. The loss now sits on q_j, gradients go to the GNN only, and a separate box shows checkpoint selection. Its caption was rewritten to match.
  - The same wording was corrected throughout: abstract, contributions, Regime A (now "development baseline"), Secs. 5.3–5.4, 6.1–6.2, limitation (i) and the conclusion. "Trained through/against BP" became "selected with/for BP".
  - The interleaved variant, which is trained through BP, is described in Appendix A as the exception.
- **Open:** none.

### M2: degeneracy of the loss
- **Asked:**
  - Say whether stabilizer-equivalent corrections get the same loss.
  - Cite the code that implements the target.
- **Evidence:**
  - The target is the sampled error, `y = z_errors` (`Harness.graphs`), compared bit by bit (`gnn_pipeline/loss_functions.py`, `focal_loss`). The loss is not degeneracy-aware.
  - The mechanism audit (R1) shows the consequence. A weight-3 error covering half of a weight-6 stabilizer gives two equally likely, logically equivalent errors with the same syndrome. The network receives the same input for both, so the loss cannot favour either half.
  - BP and GNN-BP fail on every such shot: 57/57 at p = 0.04 and 24/24 at p = 0.06.
- **Changed:**
  - New paragraph in Sec. 3.2 on degeneracy, citing Liu & Poulin (PRL 2019).
  - New limitation item.
  - Discussion (6.1) and conclusion mention a degeneracy-aware loss as the fix.
- **Open:** no degeneracy-aware loss was trained; that is a new experiment.

### M3: X and Z components
- **Asked:**
  - Report X-only, Z-only and aggregate LER for the main result.
  - Test separate heads, or justify the shared correction.
- **Evidence:** `audit_components.py`, `results/components/`. At p = 0.04, failures per 20,000 shots:
  - BP, BP-100 and BP-OSD fail only on the Z component: 2,155, 1,727 and 1,414 Z failures, and 0 X failures each.
  - GNN-BP with Δ added to both components (the paper's setting): Z 1,845–1,899, X 7–54.
  - GNN-BP with Δ added to Z only: Z 1,845–1,899, X 0, giving a reduction of 13.3% ± 0.9% (10.8% ± 0.7% at p = 0.06).
  - GNN+BP-OSD: at most one extra X failure.
- **Changed:**
  - New Table 5 (failures by component) and the paragraph "The two CSS components" in Sec. 5.3.
  - Remark 1 rewritten with these numbers and moved to the end of Sec. 3.2.
  - The input/output definition per component is in Sec. 3.1 and Table 2.
  - Separate heads: BP makes no X-component errors at these operating points, so a trained X head has nothing to correct. Leaving the X prior unchanged (Δ^X = 0) is the appropriate choice.
- **Open:** the Z-only variant was chosen after seeing the test results, so the paper keeps the pre-specified setting as the headline and labels Z-only as post hoc.

### M4: prior feature vs. simulator
- **Asked:**
  - State which prior the GNN sees for each component.
  - Make the feature equations and the simulator describe the same thing.
- **Evidence:**
  - l = ln((1−p)/p) uses the shot's total physical error rate (`Split.llr` from `p_values`).
  - In the drifting training streams that rate is the true instantaneous p(t). The same l (+Δ) is the BP prior for both components (`Harness.bp`).
  - The reviewed Sec. 2.3 said "the decoder receives no direct access to instantaneous noise parameters in any experiment". That was false: the GNN experiments use the shot's true p, and the oracle experiments use true per-qubit rates by design.
  - The FiLM generator in the code also receives the true per-shot p (`gnn_model.py`: `data.p_value`), not the syndrome-weight estimate p̂ that the paper described.
- **Changed:**
  - Sec. 3.1 defines the feature and the per-component prior.
  - Sec. 2.2 gives the matched LLRs Λ_z and Λ_x.
  - Sec. 2.3's false sentence was removed, along with the random-telegraph-noise model, which no experiment used.
  - Table 2 has a new "Prior feature" row.
  - Sec. 4.1 (Regime B) now states the BP-OSD priors. Plain BP-OSD uses a uniform prior per component; matched (p_z, p_x) and unmatched (p) uniform priors gave identical decisions on all 20,000 test shots. With the GNN, each qubit enters BP-OSD with prior σ(−(ℓ+Δ_i)) in both components.
  - The FiLM description in Appendix A, Algorithm 1, Fig. 1 and Fig. 8 now say the input is p.
- **Open:** none.

### M5: LLR ratio
- **Asked:** recompute the ratio, given as 1.7 in the reviewed text.
- **Evidence:** p_z = 0.0380952 and p_x = 0.0019048, so Λ_z = 3.2288, Λ_x = 6.2615 and Λ_x/Λ_z = 1.9392. Nothing else in the paper depends on the ratio.
- **Changed:**
  - Sec. 2.2 gives "6.26/3.23 ≈ 1.94".
  - Remark 1 now says ≈ 1.94; the reviewed text had ≈ 1.7.
- **Open:** none.

### M6: Neural-BP initialisation
- **Asked:**
  - State the actual initialisation.
  - Correct "softplus(0) = 0.693 recovers standard BP".
- **Evidence:** `MinSumBPDecoder._bp_weight` returns softplus(raw)/ln 2, so raw = 0 gives weight 1 exactly. The paper's 0.693 was wrong about the code, not about softplus.
- **Changed:** Appendix A now says that dividing by ln 2 gives w = 1 exactly, so the network starts as standard min-sum BP with scaling 0.8.
- **Open:** none.

### M7: "nested"
- **Asked:**
  - Use precise set language.
  - Report both counts, the intersection and both conditional fractions.
- **Evidence:** `results/failure_sets/`, at p = 0.04.
  - BP fails on 2,155 shots and BP-OSD on 1,414; both fail on 1,319.
  - Only BP fails on 836 shots, only BP-OSD on 95.
  - P(BP fails | BP-OSD fails) = 93%; P(BP-OSD fails | BP fails) = 61%.
  - At p = 0.06 the two fractions are 96% and 71%.
  - BP-100 failures are a true subset of BP-10 failures (1,727 of 1,727).
- **Changed:**
  - "Nested" was removed from the abstract, Sec. 5.4 and Sec. 6.1.
  - Table 7 now gives both, only-BP, only-BP-OSD, and both conditional fractions.
  - Text updated.
- **Open:** none.

### M8: BP-OSD as a reference baseline
- **Asked:**
  - Say exactly which BP-OSD is used.
  - Stop treating it as the frontier.
  - Revise the related work.
- **Changed:**
  - New Sec. 1.1, "Related work", organized in four steps:
    1. why message passing fails;
    2. structural and post-processing remedies (OSD, inactivation, check-agnosia, LSD, LP, collective bit flipping, linear-time trapping-set-aware decoders);
    3. the modern BB / real-time frontier (Relay-BP, GARI with normalized min-sum ensembles, diversity methods, ambiguity clustering, hardware BP-OSD);
    4. learned decoders.
  - It states that BP-OSD is "a useful, fully specified reference decoder, not the state of the art".
  - Table 2 adds the implementation: ldpc 2.4.1 `BpOsdDecoder`. Sec. 4.1 adds the priors it is given, with and without the GNN (see M4).
  - Regime B is renamed "stronger reference decoder".
  - "Competitive classical baseline" was removed from limitation (i) and Sec. 6.3.
  - 20 references were added: 18 from the memo's list, plus Holm (1979) for the multiple-comparison correction and Yoder et al. ("Tour de gross") for the code names. All 20, and Gong et al.'s added page range, were checked against publisher, arXiv or JSTOR records by web search; direct fetches of publisher sites were blocked from this environment. Preprints are marked.
  - Uncited entries were removed from `references.bib`.
- **Open:** confirm the article number of Javed et al. (TQE 7, 2100319) on IEEE Xplore; only one source confirmed it.

### M9: oracle identifiability
- **Asked:** restrict the claim to identifiability of the latent rates.
- **Changed:** the last paragraph of Sec. 5.5 now says:
  - With a field redrawn every shot, the rates are not identifiable from one syndrome. Within each CSS component the errors are, averaged over the field, i.i.d. at p_z or p_x.
  - A syndrome-only decoder could still beat mean-prior BP-OSD by being a better i.i.d. decoder, but not by estimating the p_i.
  - The field does correlate a qubit's X and Z errors, which a joint decoder could exploit; ours decode the components separately.
- **Open:** none.

### M10: drift experiment
- **Asked:** specify the OU drift experiment completely.
- **Evidence:** no code for the reviewed drift run exists anywhere in the repository or its history. Its numbers first appear hard-coded in a figure script on 25 August.
- **Changed:**
  - With the authors' agreement, the experiment was re-run with a fully specified, committed script, `audit_drift.py`: model, estimators, five GNN seeds, 20 streams × 2,500 scored shots per volatility.
  - Sec. 5.6, Table 10 and Fig. 6 were rewritten from its results (`results/drift/`).
  - Spelled out in Sec. 5.6: the AR(1) update (Eq. 8), ρ = √(1 − v²/σ²), v as the per-shot innovation s.d. (correlation times of about 200, 15 and 7 shots), initialization x(0) ~ N(0, σ²), the cap at 0.5, what "stale" means (true rates at t = 0), the empirical estimator formula (W = 32, K = 16), what each decoder sees, and the GNN estimator's training and checkpoint rule.
  - **Result:** stale calibration equals the oracle without drift (LER 0.0210) but is already worse than the uniform mean prior at v = 0.1 (0.0747 vs 0.0655). The frequency estimator is the best non-oracle estimate at every v (0.0333 to 0.0601). The GNN estimator beats stale at every nonzero v and the mean prior at every v, but is worse than the frequency estimator on all 20 streams at every v. All eight GNN sign tests survive Holm.
  - **Compared with the reviewed version:** its claim that the GNN beats stale only at the highest volatility is not reproduced. Its finding that the frequency estimator beats the GNN at every volatility is reproduced. Its oracle LER falling from 0.033 to 0.004 does not occur, because here the spread of the rates is held fixed.
- **Open:** the design (per-shot OU innovations at fixed stationary spread σ = 1, W = 32, K = 16, the GNN estimator) is our specification. The original run may have used a different definition of volatility, which cannot be checked without its code.

### M11: circuit noise
- **Asked:**
  - Use precise terminology.
  - State how the decoder priors are assigned to DEM mechanisms.
- **Evidence:**
  - `astra_stim/biased_noise.py`: after every 1- and 2-qubit gate, each touched qubit gets an independent PAULI_CHANNEL_1(p/(η+1), 0, pη/(η+1)); X and Z are exclusive, with no Y.
  - Idle qubits get the same channel at each TICK. X_ERROR(p) is applied after resets and before Z measurements.
  - `rescale_noise_per_qubit` scales every channel on qubit q.
  - `gnn_pipeline/dem_decoder.py`: Stim's `detector_error_model(approximate_disjoint_errors=True)`.
- **Changed:**
  - New Table 11: location → fault → probability.
  - Sec. 5.7 states that this is not standard depolarizing noise and that a two-qubit gate has no correlated faults.
  - The "Decoders" paragraph says how the mean-prior and oracle DEMs are built: the same mechanisms, different priors.
  - Limitations mention the independent single-qubit channels.
- **Open:** none.

### M12: statistics
- **Asked:**
  - Define comparison families and apply Holm or label p-values exploratory.
  - Report n10/n01.
  - Avoid overclaiming "significantly worse from p = 0.03".
- **Evidence:** Holm correction over the 20 GNN+OSD vs BP-OSD tests.
  - At p ≥ 0.03, 11 of 15 seed/p pairs stay significantly worse (14 unadjusted).
  - At p = 0.04 only 2/5 seeds survive (4 unadjusted).
  - No seed is significantly better at any p.
  - The Table 4 family (10 tests) and the oracle family (9 tests) survive Holm.
- **Changed:**
  - New "Comparison families" paragraph in Sec. 4.2.
  - Table 6 adds n10/n01 and both unadjusted and Holm counts.
  - Fig. 4 marks only points where every seed survives Holm, which leaves only p = 0.05. Its error bars are now Wilson intervals for BP-OSD and seed ranges for GNN+OSD; the old bars were normal-approximation intervals.
  - Abstract, contributions, Sec. 5.4 and conclusion use the Holm numbers.
  - Table 9 adds n10/n01; Table 12 splits n10 and n01.
  - The drift experiment uses stream-level sign tests with Holm correction.
- **Open:** none.

### M13: circuit table from raw outcomes
- **Asked:** regenerate the circuit table from raw paired outcomes by script, with n10 and n01 separate.
- **Changed:**
  - `make_tables.py` rebuilds Table 12, and every other results table, from `results/*/…_failures.npz`. `build_v5.py` inserts the rows.
  - Table 12 now has separate n10 and n01 columns and Wilson intervals.
  - The values are unchanged: 1.35% vs 0.60%, n10 = 177, n01 = 28, p = 5×10⁻²⁵.
- **Open:** none.

### M14: clean-checkout reproducibility
- **Asked:**
  - Reproduce from a clean clone.
  - Pin the commit, environment, seeds, configurations and checkpoints.
- **Changed:**
  - `requirements-lock.txt` gives exact versions; all pins resolve from PyPI in a fresh venv.
  - `reproduce_all.sh` re-runs everything, then runs `make_tables.py --check`, then redraws the figures.
  - The availability statement now covers all results, including drift, and names tag `paper-v5`.
  - `audit_headline_v2.py --eval_only` and `reproduce_all.sh --stored-checkpoints` evaluate the committed GNN checkpoints instead of retraining. GNN training on CPU is not bit-reproducible across thread configurations: a clean-clone retraining matched the committed history at epoch 1 and diverged in the fourth digit of the loss at epoch 2.
  - Evaluation from a stored checkpoint reproduced the committed per-shot outcomes byte for byte (seeds 0 and 3, checked locally).
  - **Exact clean run (13 October):** a fresh clone of commit `676c8be` from GitHub was used, with every result and generated dataset deleted except the GNN checkpoints and their training histories. All 63 manuscript table rows were found by `make_tables.py --check`.
    - Of the 73 committed result and data files, 63 were regenerated identically. Five (`seed*.json`) differ only by the added `evaluated_from_stored_checkpoint` flag. The remaining five are the drift estimators' training histories, which only training produces.
    - All five data figures were pixel-identical.
    - Run time was 2 h 18 min on 4 CPU cores.
  - **Full retraining run (16 October), headline part:** in a second fresh clone, the five headline GNNs were retrained from scratch and every GNN-dependent analysis was redone. The non-GNN rows reproduced exactly, and every conclusion held, with the numbers moving within seed-to-seed variation:
    - Table 4 reduction: 12.3% ± 0.7% at p = 0.04 (paper 12.5% ± 0.6%) and 10.8% ± 0.3% at p = 0.06 (paper 10.6% ± 0.7%); McNemar p < 10⁻³³ for every seed.
    - Table 6: no seed better at any p; after Holm correction, 11 of 15 seed/p pairs at p ≥ 0.03 are significantly worse, the same as the paper.
    - Table 5: the Z-only reduction is 12.8% ± 0.9% (paper 13.3% ± 0.9%), and Δ on both components adds 6–48 X failures (paper 7–54).
    - Table 8: every retrained seed still fails all 57 weight-3 half-stabilizer errors at p = 0.04 (24 of 24 at p = 0.06), and BP-OSD fails none.
    - Gradient audit: ratio 0.54–1.10 (paper 0.63–2.19), mean |Δ| 0.71–0.89 (paper 0.54–1.01).
  - **Full retraining run, drift part:** the five drift estimators were retrained from scratch. The checkpoints, training histories, `drift.json` and per-shot outcomes came out byte-identical to the committed ones, so Table 10 and Fig. 6 reproduce exactly even after retraining.
  - **Why the two kinds of training differ:** `audit_drift.py` trains with a single CPU thread (`torch.set_num_threads(1)`), which is deterministic. The headline models were trained with PyTorch's default multithreading, which is not. The committed checkpoints remain the reference for Tables 4–8. Training them single-threaded in a future version would make them bit-reproducible too, but it would change the stored checkpoints and their numbers, so we did not do it here.
- **Open:**
  - The memo asks that someone other than the script author run the clean checkout; that still has to be a person.
  - The tag `paper-v5` must be created on GitHub, because tag pushes are blocked from this environment.

## Strongly recommended

### R1: failure mechanism
- **Done:** `audit_mechanism.py`, `results/mechanism/`, new Table 8 and the paragraph "Which failures the GNN repairs" in Sec. 5.4.
- **BP failure classes:** of 2,155 failures, 415 converge to the wrong logical class, 78 are transient, 1,030 are slow, 631 oscillate and 1 is stuck.
- **Early stopping:** per-shot early stopping would change BP's failures only from 2,155 to 2,146.
- **What the GNN repairs:** mostly slow shots (249–265 of 313–370).
- **Symmetric stabilizer configurations:**
  - They occur in 3.5% of shots but 28% of BP failures.
  - BP fails on 87% of shots containing one, against 8% of the rest.
  - On weight-3 such errors, BP, BP-100 and GNN-BP fail on 100% and BP-OSD on 0%.
  - The GNN lowers the failure rate on these configurations from 87% to 67–74%, only at weight ≥ 4.

### R2: trapping-set literature
- Raveendran & Vasić 2021, Chytas et al. 2025 and Pradhan et al. 2026 are cited in Sec. 1.1.
- Raveendran & Vasić is the frame for the symmetric configurations in Sec. 5.4.
- Raveendran et al. 2023 (soft syndrome) is cited in Sec. 6.4.

### R3: unevaluated architectures to an appendix
- FiLM, neural BP, interleaved training, Algorithm 1 and the interleaved-pipeline figure moved to Appendix A, which says they are not evaluated.
- Each item there is corrected to describe the code: the FiLM input is p, the neural-BP weight is exactly 1 at initialisation, and the interleaved variant's loss is on BP marginals.

### R4: oracle plots and tables
- Table 9 now gives absolute mean-prior and oracle LERs with Wilson intervals, the reduction, and n10/n01.
- Fig. 5 was redrawn as LER against σ per code, with no lines across codes. The caption says that code size and operating point co-vary.
- A same-p comparison was added to the text: at p = 0.04 and σ = 1 the reductions are 65%, 85% and 88%, preliminary because the larger codes have only 278 and 17 failures.

### R5: figures
- No figure was smoothed.
- Fig. 4 got correct error bars (see M12).
- Figs. 5 and 6 were redrawn from stored data with validated colour-blind-safe colours, distinct markers and complete captions.
- The drift figure (Fig. 6, previously Fig. 7) now reads `results/drift/drift.json`; it no longer uses hard-coded numbers.

### R6: shorter abstract and conclusion
- The abstract went from 303 to 243 words (−20%); the conclusion went from 266 to 211 words (−21%).
- Both now state the training and selection setup exactly and use the Holm and overlap numbers.

### R7: code names
- IBM uses lowercase: "gross code" for [[144,12,12]] and "two-gross code" for [[288,12,18]]. That is the usage in the "Tour de gross" paper and IBM's blog.
- The paper already used lowercase consistently. The Table 1 caption now cites Yoder et al. for the names. "Double-gross" appears only in the Error Correction Zoo.

## Other corrections found during this pass

- **Rounding:**
  - Table 4: BP-10 is 2,155/20,000 = 0.10775, now printed 0.1078 instead of 0.1077. BP-100 is 0.08635, now 0.0864 instead of 0.0863.
  - The reviewed text mixed float formatting and half-up rounding; all tables now round half-up from exact counts.
- **Proofreading fixes from October 1** that were missing from the reviewed PDF:
  - "comparable:" → "comparable.";
  - "checkpoint improves" → "checkpoints improve";
  - removed the redundant sentence in Sec. 5.4;
  - removed the duplicate CPU limitation;
  - Fig. 1 without the "(L = 3)" label.
- **Gradient audit (Sec. 5.3):** re-run on the five reported checkpoints. The ratio is now 0.63–2.19 and the mean |Δ| 0.54–1.01, instead of 0.22 and 0.854 from an older single-seed model.
- **Abstract sentence:** "These settings must therefore be specified…" was cut in Overleaf before review; it stays cut.
