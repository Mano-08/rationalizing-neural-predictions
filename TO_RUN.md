# Runs to do: MultiRC and FEVER

The paper evaluates only on USR Movie Review. Storek et al. (2023) also report
BERT-A2R and BERT-A2R+NI on **MultiRC** and **FEVER** (ERASER, sentence-level
rationales; their Table 2), and our Limitations names both as open. This file
lists the runs that add them, with the same models, seeds and protocol as the
USR runs in the paper.

## Round 3 (planned 2026-11-25): experiments 1, 2, 3 and 5

Round 2 (results_v27-v38) is done and in the paper. Round 3 was chosen after a
second literature review; every setting below was fixed before any run.

| # | Experiment | Stages | What it tests |
| --- | --- | --- | --- |
| 1 | The joint generator attends to document tokens only (`--document_attention`). In the released code, its attention also covers the query, [SEP] and padding, so on FEVER it can meet its loss by attending to the claim | MultiRC, FEVER: `a2r_dm`, `ni_dm`, `released_dm`, `frozen_dm` | whether this design of the released code is why every trained model is far below BM25 |
| 2 | Product of experts with a query-only BERT: the generator is trained on what the selected evidence adds to the query (`--poe_probs`) | MultiRC, FEVER: `query_only`, then `a2r_dm_poe`, `frozen_dm_poe`, `frozen_x_poe` | whether removing the query shortcut makes the generator find the evidence |
| 3 | HateXplain, GenSPP's split (rebuilt from the original release), selection rate 10% (training highlight rate), phrase window 3 (mean training span 2.2 words), special tokens never selected | `hatexplain`: `a2r`, `ni`, `frozen`, `frozen@w3` | whether the USR findings hold on a second token-level dataset; comparison with GenSPP (ACL 2025) and ReSR (EMNLP 2026) |
| 5 | USR seeds 4 and 5 | `usr`: `a2r`, `ni`, `released`, `frozen@w5` | five seeds, like the published numbers |

The query-only model is cross-fitted (two halves, each predicting the other),
with seed 1 in every session, so all PoE runs of a dataset use the same one.
`frozen@w3` / `frozen@w5` run the frozen generator with that phrase window, so it
can share a session with single-token runs. Tested on the Mac (small BERT, real
data): every stage runs; all attention falls on document tokens; with a uniform
query-only model the PoE loss equals the old loss, with a confident correct one
the generator gets no gradient; the HateXplain split has GenSPP's sizes (10,783 /
1,369 / 1,355 posts).

**Before the first session: push the code to `ani` and paste the new
`kaggle_run.py` into the notebook cell.**

### Session 0c: check (about 1 hour, first)

Main cell: `STAGES = ["setup", "smoke_r3"]`, `SEED = 1`, `BATCH = 16`,
`DATASET = "multirc"`, `TRUNCATION = "right"`. Then two more cells:

```python
DATASET = "fever"; run("setup", "smoke_r3")
DATASET = "hatexplain"; run("setup", "smoke_r3")
```

Expected: each smoke run ends with test metrics and no error. **Delete the two
extra cells afterwards.** Nothing to download.

### Wave 1 (after 0c): all of experiments 3 and 5, seed 1 of experiments 1 and 2

`BATCH = 16` everywhere. GPU T4 x2, Internet on, Save Version -> Save & Run All.

| # | DATASET | TRUNCATION | SEED | STAGES | est. time | save as |
| --- | --- | --- | --- | --- | --- | --- |
| U1 | `"usr"` | `"left"` | 4 | `["setup", ["a2r", "ni"], "probe", "pack"]` | 6-10 h | `results/results_v39` |
| U2 | `"usr"` | `"left"` | 4 | `["setup", ["released", "frozen@w5"], "probe", "pack"]` | 6-10 h | `results/results_v40` |
| U3 | `"usr"` | `"left"` | 5 | `["setup", ["a2r", "ni"], "probe", "pack"]` | 6-10 h | `results/results_v41` |
| U4 | `"usr"` | `"left"` | 5 | `["setup", ["released", "frozen@w5"], "probe", "pack"]` | 6-10 h | `results/results_v42` |
| H1 | `"hatexplain"` | `"right"` | 1 | `["setup", ["a2r", "ni"], ["frozen", "frozen@w3"], "probe", "pack"]` | 1-2 h | `results/results_v43` |
| H2 | `"hatexplain"` | `"right"` | 2 | same as H1 | 1-2 h | `results/results_v44` |
| H3 | `"hatexplain"` | `"right"` | 3 | same as H1 | 1-2 h | `results/results_v45` |
| M1 | `"multirc"` | `"right"` | 1 | `["setup", ["a2r_dm", "ni_dm"], ["released_dm", "frozen_dm"], "probe", "pack"]` | 9-10 h | `results/results_v46` |
| M2 | `"multirc"` | `"right"` | 1 | `["setup", "query_only", ["a2r_dm_poe", "frozen_dm_poe"], "frozen_x_poe", "probe", "pack"]` | 6-8 h | `results/results_v47` |
| F1 | `"fever"` | `"right"` | 1 | `["setup", ["a2r_dm", "ni_dm"], "probe", "pack"]` | 10.5 h | `results/results_v48` |
| F2 | `"fever"` | `"right"` | 1 | `["setup", ["released_dm", "frozen_dm"], "probe", "pack"]` | 10.5 h | `results/results_v49` |
| F3 | `"fever"` | `"right"` | 1 | `["setup", "query_only", ["a2r_dm_poe", "frozen_x_poe"], "probe", "pack"]` | 11 h | `results/results_v50` |
| F4 | `"fever"` | `"right"` | 1 | `["setup", "query_only", ["frozen_dm_poe"], "probe", "pack"]` | 7-8 h | `results/results_v51` |

USR sessions need `TRUNCATION = "left"`. F3 is the longest session (about 11 of
Kaggle's 12 hours: round-1 FEVER sessions took 10.1 h, plus the query-only model).

### Wave 2 (after I have checked M1, M2 and F1-F4): seeds 2 and 3

The same six MultiRC/FEVER sessions (M1, M2, F1-F4) with `SEED = 2` and `SEED = 3`:

| SEED | M1 | M2 | F1 | F2 | F3 | F4 |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | `results_v52` | `results_v53` | `results_v54` | `results_v55` | `results_v56` | `results_v57` |
| 3 | `results_v58` | `results_v59` | `results_v60` | `results_v61` | `results_v62` | `results_v63` |

About 330 GPU hours for both waves.

## Round 2 (planned 2026-11-23): experiments 1, 2, 4 and 5

**This replaces "No further runs are planned" below.** Four experiments chosen
after the literature review; every setting was fixed before any run.

| # | Experiment | Stage(s) | What it tests |
| --- | --- | --- | --- |
| 1 | Frozen generator that encodes the query with each sentence on its own (`--generator pairwise`) | `frozen_x` (MultiRC, FEVER) | whether the frozen generator fails on MultiRC/FEVER because it cannot relate a sentence to the query |
| 2 | Random vicinity of the selection: 20% of the selected tokens swapped for other tokens of the review during training (Liu et al., TACL 2025) | `vicinal` (USR) | whether scrambling *which* tokens are selected breaks the encoding where NI does not |
| 4 | Straight-through path from the predictor's loss to the generator, weight 1.0 | `ni_coupled`, `a2r_coupled` (USR) | whether NI steers the generator once a gradient reaches it (exposure below 1?) |
| 5 | DR: predictor learning rate = selection rate x generator's = 4e-6 (Liu et al., 2023) | `dr` (USR) | freezing vs. the closest published remedy |
| 5 | BM25 sentence selector, no training | done on the Mac: `results/bm25/` | MultiRC F1 51.3, FEVER F1 51.1 (every trained model: 17-39) |

USR runs pair by seed with the existing `a2r_tail_seed*` and `ni_tail_seed*`
runs (results_v8-v11): same batch (16), window (TAIL), single tokens and noise
level (0.2); only the tested option differs. Tested on the Mac (tiny BERT, real
data): every stage runs, the coupling passes gradient to the generator only
when on, the perturbation swaps exactly 20% of the selected word tokens, the
pairwise generator's sentence attentions reach the selector unchanged, and the
new 9.5 h time limit of USR runs stops training cleanly.

### Session 0b: check (about 40 minutes, first)

Push the code to `ani` first. Then, in the main cell:

```python
STAGES = ["setup", "smoke_new"]
SEED = 1
BATCH = 16
DATASET = "usr"
TRUNCATION = "left"
```

and in a second cell: `DATASET = "multirc"; run("setup", "smoke_new")`.
Expected: four USR smoke runs and one MultiRC smoke run, each ending with test
metrics and no error. **Delete the second cell afterwards.**

### Sessions R1-R12

Change only these lines (GPU T4 x2, Internet on, Save Version -> Save & Run All).
**USR sessions need `TRUNCATION = "left"`; MultiRC/FEVER sessions `TRUNCATION = "right"`.**

| # | DATASET | TRUNCATION | SEED | STAGES | est. time | save as |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | `"usr"` | `"left"` | 1 | `["setup", ["dr", "vicinal"], "probe", "pack"]` | 5-10.5 h | `results/results_v27` |
| R2 | `"usr"` | `"left"` | 2 | `["setup", ["dr", "vicinal"], "probe", "pack"]` | 5-10.5 h | `results/results_v28` |
| R3 | `"usr"` | `"left"` | 3 | `["setup", ["dr", "vicinal"], "probe", "pack"]` | 5-10.5 h | `results/results_v29` |
| R4 | `"usr"` | `"left"` | 1 | `["setup", ["ni_coupled", "a2r_coupled"], "probe", "pack"]` | 5-10.5 h | `results/results_v30` |
| R5 | `"usr"` | `"left"` | 2 | `["setup", ["ni_coupled", "a2r_coupled"], "probe", "pack"]` | 5-10.5 h | `results/results_v31` |
| R6 | `"usr"` | `"left"` | 3 | `["setup", ["ni_coupled", "a2r_coupled"], "probe", "pack"]` | 5-10.5 h | `results/results_v32` |
| R7 | `"multirc"` | `"right"` | 1 | `["setup", "frozen_x", "probe", "pack"]` | 2-3 h | `results/results_v33` |
| R8 | `"multirc"` | `"right"` | 2 | `["setup", "frozen_x", "probe", "pack"]` | 2-3 h | `results/results_v34` |
| R9 | `"multirc"` | `"right"` | 3 | `["setup", "frozen_x", "probe", "pack"]` | 2-3 h | `results/results_v35` |
| R10 | `"fever"` | `"right"` | 1 | `["setup", "frozen_x", "probe", "pack"]` | 6-9 h | `results/results_v36` |
| R11 | `"fever"` | `"right"` | 2 | `["setup", "frozen_x", "probe", "pack"]` | 6-9 h | `results/results_v37` |
| R12 | `"fever"` | `"right"` | 3 | `["setup", "frozen_x", "probe", "pack"]` | 6-9 h | `results/results_v38` |

`BATCH = 16` in all sessions. The sessions are independent: run them in any
order and in parallel on different accounts, each account its owner's. The
USR probe stage now runs only the `tail` and `tail_random` probes
(`USR_PROBES` in `kaggle_run.py`). About 115 GPU hours in total.

## Status (2026-11-23): round 1 done

All 24 runs are in `results/results_v15` to `results_v26` and in the paper
(Sec. 7.3, Table 3, App. I: Tables 13-14). Measured times differ from the
estimates in section 4: a fine-tuned FEVER epoch took 3.6-4.3 h (not
1.8-2.5 h), so all nine fine-tuned FEVER runs stopped at the 9.5 h limit
after 2.2-2.6 epochs (checkpoint: epoch 2 in seven runs, epoch 1 in two).
Analysis scripts: `~/noise_injection_cache/ce_analysis/` (`analyze.py`,
`make_tables.py`).

## Changes from the first version of this file (2026-11-17)

|                      | first version                              | now                        | why                                                                                                                                                                          |
| -------------------- | ------------------------------------------ | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| validation           | twice per epoch (released `run_ce.py`)     | once per epoch             | as in our USR runs (`run_words.py`), so "patience 2" means 2 epochs as in Table 11                                                                                           |
| MultiRC patience     | none (Storek et al.)                       | 2                          | Table 11 of the paper: one protocol for every run ("Epochs, patience: 5, 2")                                                                                                 |
| `released`           | released selection + released NI alignment | released NI alignment only | in the paper, NI_released differs from NI only in the training-time alignment defect (App. A item 3); selection fixes apply to every model, like the evaluation fixes on USR |
| probe `__fixed_eval` | re-scored `released` with fixed selection  | removed                    | no longer needed: `released` is evaluated like every other model                                                                                                             |

Session list, stages and results folders are unchanged.

## 1. What is run

| Stage      | Name in the paper     | Generator      | Noise                                                              |
| ---------- | --------------------- | -------------- | ------------------------------------------------------------------ |
| `a2r`      | A2R                   | fine-tuned     | none                                                               |
| `ni`       | A2R+NI (as described) | fine-tuned     | NI, alignment fixed                                                |
| `released` | A2R+NI_released       | fine-tuned     | NI with the alignment defect (`--legacy_alignment`, training only) |
| `frozen`   | FG                    | frozen encoder | none                                                               |

2 datasets x 4 models x seeds 1-3 = **24 training runs**, plus `probe`:
each trained predictor evaluated on k randomly chosen sentences
(`<run>__random`), the encoding diagnostic of Sec. 8 (accuracy on random
selections and share in one class, column `1cls%`).

Protocol (paper Table 11 unless noted): `bert-base-uncased`, Adam 2e-5,
batch 16, at most 5 epochs, one noise-free validation per epoch, patience 2,
checkpoint of the last epoch in which both validation losses improved,
lambda = 0.1, 20% selection, 512 wordpieces. Per dataset (Storek et al.):

|                               | MultiRC                                          | FEVER                  |
| ----------------------------- | ------------------------------------------------ | ---------------------- |
| train / valid / test          | 24,029 / 3,214 / 4,848                           | 97,957 / 6,122 / 6,111 |
| noise level p                 | 0.3                                              | 0.05                   |
| selection unit                | sentence: top 20% of sentences by mean attention | same                   |
| test docs over 512 wordpieces | 8.1%                                             | 16.8%                  |

**Not applicable:** phrase selection (the unit is already a sentence), so the
method here is FG, not FG+PS; HEAD/TAIL (the query comes first, documents
keep their beginning); the independent classifier; noise on FG.

**Comparing with the published rows.** Applying the paper's R/P test
(App. D) to Storek et al.'s Table 2: published R/P is 1.18-1.19 (MultiRC) and
1.21 (FEVER), while a 20% sentence budget on these test sets gives S/G =
1.22-1.23 and 1.38-1.49, whatever the rounding. As for USR, the published
numbers were not produced by the released evaluation, so the paired
comparisons among our own runs are the main result and the published rows are
shown for reference only:

| Dataset | Model       | Acc        | P          | R          | F1         |
| ------- | ----------- | ---------- | ---------- | ---------- | ---------- |
| MultiRC | BERT-A2R    | 66.1 (1.9) | 18.5 (1.6) | 21.9 (2.2) | 19.3 (1.8) |
| MultiRC | BERT-A2R+NI | 66.4 (0.8) | 22.6 (1.2) | 26.9 (1.8) | 23.8 (1.4) |
| FEVER   | BERT-A2R    | 82.1 (3.2) | 36.3 (0.6) | 44.0 (0.3) | 36.7 (0.5) |
| FEVER   | BERT-A2R+NI | 78.2 (1.9) | 39.0 (2.5) | 47.2 (2.9) | 39.5 (2.5) |

## 2. Code changes (commit bc97ac6 on `final_review`, pushed to `ani` as 2e5528e)

The released MultiRC/FEVER code (`model/run_ce.py`, `models_ce.py`,
`datasets_ce.py`) had never been run in this project. Defects found and
checked on the Mac with the real data:

1. **Sentence scores from the wrong words, every model** (training and
   evaluation): the selector subtracts a token position from word ids, so
   19.4% (MultiRC) and 8.4% (FEVER) of the test evidence tokens count towards
   the wrong sentence. Fixed for all runs (`--legacy_selection` restores it).
2. **NI alignment** (training only, the USR defect, larger here): the
   predictor gets the words (query wordpieces - query words + 1) positions
   before the selected ones, often another sentence. Example: selecting
   "dogs bark loudly at night ." gave the predictor "the cat sat on the mat".
   Fixed; `--legacy_alignment` restores it (`released`).
3. Sentences beyond 512 tokens got a NaN score, which `topk` ranks first.
   Fixed for all runs (`--legacy_selection` restores it).
4. The input for comprehensiveness held the query twice and could exceed 512
   tokens (crash). Fixed for all runs.
5. Training with noise crashed: validation/test asked for replacement
   statistics that are never written (and never used).
6. No seed and no per-example output.

Added: `--seed` (same-seed reruns identical), `config.json`,
`per_example.json` (tp/fp/fn, label, prediction per document, for the paired
bootstrap), `comp_suff_prob`, `--random_selection`, `majority_class_share`,
`--validations_per_epoch`, `--time_limit_hours`, `--*_subset`;
`kaggle_run.py` gets `DATASET`, downloads and preprocesses the data in
`setup`. Verified on CPU (tiny BERT, real data): preprocessing gives the
split sizes above, every stage runs on both datasets, same-seed reruns give
identical per-example results.

## 3. Kaggle sessions

Notebook: **GPU T4 x2**, Internet **On**. Paste `kaggle_run.py` into one cell,
change only these lines at its top, run. For long sessions use **Save
Version -> Save & Run All**. Download `results.tar.gz` and keep it as the
folder in the last column. Leave `TRUNCATION`, `WINDOW` and
`TIME_LIMIT_HOURS` as they are.

### Session 0: check (about 15 minutes, first)

```python
STAGES = ["setup", "smoke"]
SEED = 1
BATCH = 16
DATASET = "multirc"
```

then in a second cell: `DATASET = "fever"; run("setup", "smoke")`.
Expected: `wc -l` shows 24029 / 3214 / 4848 and 97957 / 6122 / 6111, and each
smoke run ends with "Sentence-level Micro-Averaged ...".

### Sessions 1-12 (`BATCH = 16` in all)

| #   | DATASET     | SEED | STAGES                                               | est. time | save as               |
| --- | ----------- | ---- | ---------------------------------------------------- | --------- | --------------------- |
| 1   | `"multirc"` | 1    | `["setup", ["a2r", "ni"], "probe", "pack"]`          | 3-6.5 h   | `results/results_v15` |
| 2   | `"multirc"` | 1    | `["setup", ["frozen", "released"], "probe", "pack"]` | 3-6.5 h   | `results/results_v16` |
| 3   | `"multirc"` | 2    | `["setup", ["a2r", "ni"], "probe", "pack"]`          | 3-6.5 h   | `results/results_v17` |
| 4   | `"multirc"` | 2    | `["setup", ["frozen", "released"], "probe", "pack"]` | 3-6.5 h   | `results/results_v18` |
| 5   | `"multirc"` | 3    | `["setup", ["a2r", "ni"], "probe", "pack"]`          | 3-6.5 h   | `results/results_v19` |
| 6   | `"multirc"` | 3    | `["setup", ["frozen", "released"], "probe", "pack"]` | 3-6.5 h   | `results/results_v20` |
| 7   | `"fever"`   | 1    | `["setup", ["a2r", "ni"], "probe", "pack"]`          | 5-10.5 h  | `results/results_v21` |
| 8   | `"fever"`   | 1    | `["setup", ["frozen", "released"], "probe", "pack"]` | 5-10.5 h  | `results/results_v22` |
| 9   | `"fever"`   | 2    | `["setup", ["a2r", "ni"], "probe", "pack"]`          | 5-10.5 h  | `results/results_v23` |
| 10  | `"fever"`   | 2    | `["setup", ["frozen", "released"], "probe", "pack"]` | 5-10.5 h  | `results/results_v24` |
| 11  | `"fever"`   | 3    | `["setup", ["a2r", "ni"], "probe", "pack"]`          | 5-10.5 h  | `results/results_v25` |
| 12  | `"fever"`   | 3    | `["setup", ["frozen", "released"], "probe", "pack"]` | 5-10.5 h  | `results/results_v26` |

Session 1, for example:

```python
STAGES = ["setup", ["a2r", "ni"], "probe", "pack"]
SEED = 1
BATCH = 16
DATASET = "multirc"
```

- `probe` must run in the same session as the training (`pack` keeps no
  checkpoints).
- If a session is cut off, run the same cell again: finished runs are skipped.
- Essential: `a2r`, `ni`, `frozen`. `released` is optional, but it shares the
  GPU-parallel session with `frozen`, so dropping it saves no session time.

## 4. Time and budget

**Measured, session 1 (MultiRC seed 1, `a2r` + `ni` in parallel):** about
56 min per epoch; both runs stopped by patience after epoch 3 (best
checkpoint: epoch 1); training 2.8 h, test evaluation 6 min, each random
probe 7 min; session about 3.3 h. The estimates below hold.

From Storek et al.'s per-epoch times (USR 110 min, MultiRC 70, FEVER 150) and
our USR epochs on T4 (fine-tuned 1.3-1.8 h, frozen 0.7-0.9 h):

- MultiRC: about 0.8-1.2 h per fine-tuned epoch; with patience 2, 3-5 epochs.
- FEVER: about 1.8-2.5 h per fine-tuned epoch. `TIME_LIMIT_HOURS = 9.5` stops
  training so the session ends within Kaggle's 12 hours; the best checkpoint
  so far is evaluated, the log says "Time limit of 9.5 h reached" and
  `metrics.json` holds `"stopped_by_time_limit": true`. Such a run trained
  fewer than 5 epochs and the paper must say so.
- Total: roughly 30 session hours for MultiRC and up to 65 for FEVER.

## 5. Afterwards

Bring the folders back; from the JSON files (never from log tables): a table
per dataset like Table 2 (Acc, sentence-level P/R/F1, comp/suff on both
scales, mean (sd) over 3 seeds; no IOU), random-selection accuracy and
one-class share per run, paired bootstrap per seed (FG - NI, FG - A2R,
NI - A2R, NI_released - NI), defects 1-4 in App. A, and the Limitations
"Scope" item. The paper ends on page 8, so the new table needs room.
