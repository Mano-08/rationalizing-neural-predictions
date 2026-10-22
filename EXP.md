# Experiments: current methodology and new approaches

This file describes the method we started from, what the experiments showed about it, and the new approaches that are now implemented in the code. Every result is labelled with how far it has been verified:

- **Full pipeline**: trained and scored on Kaggle with BERT-base, 40,000 training reviews, 2,000 annotated test reviews. One seed unless stated.
- **Small-scale test**: generator only, on cached BERT-base states of 2,000 training and 1,000 test reviews, 3 seeds. No rationale predictor is trained.
- **Untested**: implemented and unit-tested, never trained at scale.

Dataset throughout: USR Movie Review. The rationale is always the top 20% of tokens.

Full-pipeline numbers are computed from the result files in `results/` (one folder per Kaggle session, `results_v1` to `results_v14`), not from the rounded values printed in the Kaggle logs.

---

## 1. The current methodology

### 1.1 The pipeline (BERT-A2R, Storek et al. 2023)

```
review --> GENERATOR (BERT) --> one score per token (attention)
              |
              |--> ATTENTION PREDICTOR: classifies from the attention-weighted
              |    sum of the generator's token states
              |
              `--> top 20% of tokens = rationale
                        |
                   NOISE INJECTION (training only): replace some
                   rationale words by random vocabulary words
                        |
                   RATIONALE PREDICTOR (second BERT): classifies
                   from the rationale alone
```

- Loss of each predictor: cross-entropy on its own prediction, plus 0.1 x the Jensen-Shannon divergence between the two predictions.
- The generator is trained only through the attention predictor. **No gradient flows from the rationale predictor back to the generator.**
- Noise injection (NI): a selected word is replaced with probability `p x r`, where `r` is high for words with low TF*IDF in the review. The replacement is drawn from the vocabulary. The original paper uses a fixed `p = 0.2`.

### 1.2 Adaptive Noise Injection (the method of our draft)

The same pipeline, but `p` changes during training:

| Variant | Noise level |
| --- | --- |
| Exponential decay | `p_t = p_final + (p_0 - p_final) * exp(-gamma * t)` |
| Cosine decay | `p_s = p_final + (p_0 - p_final) * (1 + cos(pi * s)) / 2` |
| Closed loop (added later) | a PI controller raises `p` when the two predictors disagree on the clean rationale |

### 1.3 How it is scored

- **Accuracy**: of the rationale predictor.
- **Plausibility**: token-level precision, recall, F1 and IOU-F1 against human rationales.
- **Faithfulness**: comprehensiveness and sufficiency.

---

## 2. What the experiments showed about the current methodology

### 2.1 Full-pipeline results with a fine-tuned generator

Scored on the beginning of each review (what the released code does), single tokens:

| Run | Acc | P | R | F1 | IOU | Stopword share |
| --- | --- | --- | --- | --- | --- | --- |
| No noise | 88.1 | 11.2 | 30.6 | 16.4 | 17.8 | 0.64 |
| NI, fixed p = 0.2 | 85.1 | 13.4 | 36.7 | 19.6 | 20.5 | 0.56 |
| Closed loop (predictor disagreement) | 81.3 | 11.6 | 31.8 | 17.0 | 17.5 | 0.71 |
| NI as released (with its alignment bug) | 81.7 | 16.8 | 46.1 | 24.7 | 27.3 | 0.61 |
| Random selection of 20% of the tokens (expected scores) | - | 7.3 | 20.0 | 10.7 | - | 0.51 (measured on the end of the review) |

The paper reports F1 33.3 for its best model. None of these runs comes close, and all of them select more stopwords and punctuation than a random pick. All of them were also trained on the beginning of the review: trained and scored on the end, `ni` reaches F1 31.4 with seed 3 (4.3) but 22.5 and 19.5 with seeds 2 and 1 (4.5, 4.6), and NI as released reaches 23.3, 32.5 and 27.1 with seeds 1, 2 and 3 (4.7, 4.8).

### 2.2 Why changing the noise level does not help

1. **The noise cannot reach the generator.** The generator gets no gradient from the rationale predictor, so noise on the rationale does not tell it which tokens were bad choices. Measured: in every run the share of replaced rationale words equals `p`, which is what a random rationale would get. If noise injection steered the generator, that share would fall below `p`.
2. **The closed-loop controller had nothing to react to.** The two predictors agree even when 71% of the rationale is stopwords, so its signal stayed at the target and the noise stayed near 0.18.
3. **A fine-tuned generator hides the label in arbitrary tokens.** Small-scale test: with attention restricted to stopwords and punctuation, a classifier on fine-tuned BERT states is as accurate as with free attention (90.3% vs 89.1%; the raw output of this test was not preserved, see 6, item 8). Nothing forces the attention onto the words that carry the evidence.

Note (sessions 5-7): NI against no noise does raise a fine-tuned generator's F1 (+2.4, +4.8 and +8.1 on the end of the review with seeds 1, 2 and 3, +3.3 on the beginning with seed 1). It does so not by steering the selection, but apparently by breaking a code between generator and predictor, and it broke it in only one of three seeds (4.4-4.6).

### 2.3 What the published numbers correspond to

The paper's numbers match the **end** of each review, while the released code keeps the beginning when a review exceeds 512 tokens. With a frozen generator:

| | P | R | F1 | IOU |
| --- | --- | --- | --- | --- |
| Paper, "fixed gen. weights" | 21.9 | 47.4 | 30.0 | 29.9 |
| Ours, end of review (full pipeline, mean of three seeds) | 23.0 | 49.8 | 31.5 | 30.5 |
| Ours, beginning of review (full pipeline, mean of three seeds) | 19.0 | 52.0 | 27.8 | 27.0 |

The end of a review holds more of the human rationales: 68% of annotated words survive the 512-token limit there, against 54% at the beginning.

### 2.4 Bugs fixed in the released code

- The statistics script crashed, and training with noise asked for a file that was never created.
- During noisy training the rationale predictor received the token *before* each selected token. Fixed by default; `--legacy_alignment` restores the old behaviour. The defect matters, but not in one direction: with the same seed it changes the F1 of NI from 19.6 to 24.7 (beginning, seed 1) and, on the end of the review, by +3.8, +10.0 and -4.3 in seeds 1, 2 and 3 (4.8).
- Comprehensiveness and sufficiency applied a sigmoid to values that were already probabilities. Both versions are now reported.

---

## 3. The new approaches

### 3.1 Frozen generator

**Idea.** Keep the generator's BERT at its pretrained weights. Train only its two linear layers: the token scorer and the classifier of the attention predictor. The rationale predictor is still fine-tuned in full.

**Why.** Fine-tuning is what lets the generator spread the label over arbitrary tokens (2.2, point 3). Pretrained states keep the sentiment signal on the sentiment words.

**Switch.** `--freeze_encoder_bb` (Kaggle stage `frozen`).

**Status: full pipeline.** F1 27.8 on the beginning of the review, against 16.4 to 24.7 for the fine-tuned runs trained there. A fine-tuned NI run trained on the end of the review gets 28.3 on the beginning and 31.4 on the end with seed 3, as much as the frozen generator with single tokens (4.3), but only 19.5 and 22.5 with seed 2 and 16.6 and 19.5 with seed 1, 8.7 and 11.9 F1 below the frozen generator (4.5, 4.6). The frozen generator varies little between seeds (single tokens, end of review: F1 31.3 to 31.7 over three seeds). Training is also about twice as fast.

### 3.2 Phrase selection

**Idea.** Before taking the top 20%, average each token's attention over a window of 5 tokens. High-attention tokens then pull in their neighbours, and the rationale becomes a set of phrases.

**Why.** Humans mark phrases ("is cast perfectly"), not isolated words. Single-token selection gave spans of about 2 tokens; humans mark about 5 spans per review.

**Switch.** `--selection_window 5` (Kaggle setting `WINDOW = 5`). It can also be applied at evaluation only.

**Status: full pipeline.** On the same frozen models (mean of three seeds): F1 31.5 -> 34.2 and IOU 30.5 -> 50.9. A window of 9 gives IOU 60.4 but F1 33.3.

### 3.3 End-of-review window

**Idea.** For reviews longer than 512 tokens, keep the end instead of the beginning.

**Why.** Reviewers state their verdict at the end. More human rationales fall inside the window (68% vs 54%), and this is the setting the published numbers correspond to.

**Switch.** `--truncation_side left` (Kaggle setting `TRUNCATION = "left"`). It can also be applied at evaluation only.

**Status: full pipeline.** On the same frozen models (mean of three seeds): F1 +3.3 to +3.6 and accuracy +3.4 to +3.8 points. It applies to every model, though not equally (F1 +1.7 for `a2r`, +3.0 to +3.6 for NI and the frozen generator; audit 2026-10-22), so it is a protocol choice, not a method gain. Both windows should be reported.

### 3.4 Noise in the generator's own evidence

**Idea.** Move the noise from the rationale to the place where the generator has a gradient. The attention predictor classifies from `sum_i a_i * h_i`. During training, the state `h_i` in that sum is replaced, with probability `p x r_i`, by the state of a random token of the batch. The attention scores `a_i` are still computed on the clean input.

**Why.** Attending to a word that noise targets now gives the generator's own predictor unreliable evidence, and that shows up in the generator's own loss.

**Switch.** `--inject_noise --noise_target evidence --noise_p 0.3` (Kaggle stage `frozen_en`). `--noise_target both` also keeps the noise on the rationale.

**Status: small-scale test only.** On a frozen generator, F1 on the beginning of the review:

| Noise level p | 0 | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | 26.1 | 26.4 | 26.9 | 27.3 | 27.3 | 25.1 | 14.1 |
| Attention-predictor accuracy | 81.8 | 83.5 | 84.2 | 85.0 | 84.8 | 83.0 | 72.6 |

A mild level adds about 1.2 F1 and 3 accuracy points; a strong level collapses the model.

*Full pipeline (trained on the end of the review with a 5-token window, p = 0.3):* the noise works as intended on the attention, but it does not improve the rationale.

| Run | Seed | Acc | P | R | F1 | IOU | Stopword share (end of review, single tokens) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `frozen` (no noise) | 1 | 92.25 | 25.13 | 54.41 | 34.38 | 51.12 | 0.42 |
| `frozen` (no noise) | 2 | 91.75 | 24.99 | 54.12 | 34.20 | 50.55 | 0.38 |
| `frozen_en` (evidence noise, p = 0.3) | 1 | 91.45 | 25.33 | 54.84 | 34.65 | 50.91 | 0.38 |
| `frozen_en` (evidence noise, p = 0.3) | 2 | 90.80 | 24.67 | 53.42 | 33.76 | 49.21 | 0.33 |
| `frozen_ni` (noise on the rationale, p = 0.2) | 1 | 91.40 | 25.14 | 54.43 | 34.39 | 51.05 | 0.42 |
| `frozen_ni` (noise on the rationale, p = 0.2) | 2 | 91.70 | 25.02 | 54.18 | 34.23 | 50.49 | not probed |
| **Mean, `frozen`** | 1-2 | 92.0 | 25.1 | 54.3 | 34.3 | 50.8 | 0.40 |
| **Mean, `frozen_en`** | 1-2 | 91.1 | 25.0 | 54.1 | 34.2 | 50.1 | 0.35 |
| **Mean, `frozen_ni`** | 1-2 | 91.6 | 25.1 | 54.3 | 34.3 | 50.8 | |

The noise runs have two seeds, so this table compares seeds 1-2 only. `frozen` also has a third seed (4.1).

The generator moved its attention away from the words that the noise targets: the attention-weighted exposure `E = sum_i a_i * r_i` fell from 0.89-0.92 (`frozen_closed`, whose controller kept p at 0.04-0.06) to 0.83 in both seeds, and in both seeds the stopword share of single-token rationales is lower than without noise (0.38 vs 0.42, 0.33 vs 0.38). (The training log of `frozen_ni` prints a different exposure, the mean `r` of the selected words, 1.02; it is not comparable with the attention-weighted one.) But F1 changed by +0.3 in seed 1 and -0.4 in seed 2, and accuracy fell by 0.80 and 0.95 points: **evidence noise does not help, it slightly hurts accuracy.**

**Training is deterministic given the seed.** `frozen` seed 2 was trained twice, in two different Kaggle sessions (once next to `frozen_en`, once next to `frozen_ni`), and the two runs are bit-identical: the same training losses, metrics and per-review results (`results_v5` and `results_v6`). So two runs with the same seed differ only by what we change.

Noise on the rationale (`frozen_ni`) changes nothing for a frozen generator: in both seeds its F1 is within 0.04 points of `frozen` with the same seed (+0.01 in seed 1, +0.04 in seed 2) and its IOU within 0.07, much closer than two seeds of `frozen` are to each other (0.2 and 0.6 points apart). Its only effect is on the rationale predictor's accuracy: -0.85 in seed 1, -0.05 in seed 2.

### 3.5 Exposure-controlled noise level (the adaptive part)

**Idea.** *Exposure* is the share of the generator's attention that sits on the words noise targets: `E = sum_i a_i * r_i`. It is 1 for uniform attention. A PI controller raises `p` while the smoothed exposure is above a target and lowers it when it falls below:

```
p_t = clip(p_min + K_P * (E_t - E*) + K_I * sum(E - E*), p_min, p_max)
```

**Why.** The useful range of `p` is narrow (table in 3.4). Exposure drops sharply when the noise becomes too strong, so it is a signal the controller can hold on to. Predictor disagreement, the first signal we tried, does not move.

**Switch.** `--noise_schedule closed_loop --ctrl_signal exposure --ctrl_target 0.9` (Kaggle stage `frozen_closed`).

**Status.**
- *Small-scale test:* with target 0.9 the controller matched the best fixed level without a sweep (F1 27.3, accuracy 85.2). Targets 0.8 and 0.7 were clearly worse (21.9 and 18.3).
- *Full pipeline:* **the controller kept the noise almost off.** In the first epoch the exposure was 0.70, far below the target of 0.9, so `p` fell from 0.2 to about 0. From the second epoch on, the exposure was 0.89 to 0.92, close to the target, and `p` stayed at 0.04 to 0.06. The run is therefore practically the frozen run without noise (F1 34.38 in both, IOU 51.10 against 51.12, accuracy 91.70 against 92.25). Fixed evidence noise was tested separately (3.4).
- *Conclusion:* an absolute target is brittle, because the natural exposure differs between settings and drifts during training.

### 3.6 Other options implemented

| Option | What it does | Status |
| --- | --- | --- |
| `--noise_schedule exponential / cosine / linear` | The decay schedules of our draft | Untested in this code |
| `--ctrl_signal jsd` / `entropy` | Closed loop on predictor disagreement or attention entropy | Disagreement: full pipeline, no effect |
| `--noise_source mlm` | Replace words by in-context substitutes from BERT's masked-language-model head | Untested |
| `--replacement_probs saliency` | Choose words to replace by gradient x input instead of TF*IDF | Untested |
| `--coupling_weight` | Let the rationale predictor's loss reach the generator (straight-through) | Untested |
| `--generator_layers N` | Use only the first N layers of the generator | Small-scale: lower layers are worse on a frozen generator |
| `--mask_special_tokens` | Keep `[SEP]` and `[PAD]` out of the attention and the rationale | Used in controls |

### 3.7 Diagnostics and tooling added

- **Floor and ceiling.** Every evaluation reports what a random and a perfect selection of the same size would score.
- **Random-mask control.** `--selection_method random` scores a trained model with random rationales.
- **Degeneracy measures.** Share of stopwords and punctuation, span length, realized selection rate, exposure.
- **Truncation report.** Share of human-annotated words that survive the 512-token limit.
- **Probe stage** (Kaggle stage `probe`). Re-scores trained models under other settings (beginning or end of the review, single tokens or phrases, random selection) without retraining.
- **Small-scale test.** BERT-base states are cached once; generator variants then train in under a minute on a laptop. Its predictions matched the full pipeline (heads trained with the pipeline's learning rate, 2e-5, and number of updates, 12,500; `lr_check.py`, one seed): predicted F1 31.1 and 34.4, measured 31.5 and 34.2 (mean of three seeds); predicted IOU 30.3 and 51.0, measured 30.5 and 50.9.

---

## 4. Results so far

### 4.1 Full pipeline: frozen generator with phrase selection

Three seeds (Kaggle sessions of 2026-10-08, 2026-10-11 and 2026-10-12). The model was trained on the end of each review with a 5-token window, then scored under each setting by the `probe` stage. Mean of the three seeds, standard deviation in brackets (from the exact values in `results/`):

| Setting | Acc | P | R | F1 | IOU | Stopword share | Span (words) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **End of review, phrases (w = 5)** | **91.8** (0.4) | **25.0** | **54.2** | **34.2** (0.1) | **50.9** (0.3) | 0.52 | 5.9 |
| End of review, longer phrases (w = 9) | 91.1 (0.2) | 24.3 | 52.6 | 33.3 (0.2) | 60.4 (0.2) | 0.53 | 9.1 |
| End of review, single tokens | 90.9 (1.1) | 23.0 | 49.8 | 31.5 (0.2) | 30.5 (0.5) | 0.42 | 1.9 |
| Beginning of review, phrases | 88.4 (1.0) | 21.1 | 57.8 | 31.0 (0.1) | 46.8 (0.2) | 0.51 | 5.9 |
| Beginning of review, single tokens | 87.1 (1.1) | 19.0 | 52.0 | 27.8 (0.2) | 27.0 (0.5) | 0.41 | 1.9 |
| End of review, random selection | 74.1 (0.6) | 9.2 | 20.0 | 12.6 (0.1) | 7.6 (0.1) | 0.51 | 1.3 |
| Paper, best model (tuned + NI) | 87.9 (1.8) | 24.4 | 52.7 | 33.3 (0.8) | 38.4 (1.9) | | |
| Paper, fixed generator | 85.0 (0.8) | 21.9 | 47.4 | 30.0 (0.5) | 29.9 (0.6) | | |

The three seeds of the main setting:

| Seed | Acc | P | R | F1 | IOU |
| --- | --- | --- | --- | --- | --- |
| 1 | 92.25 | 25.13 | 54.41 | 34.38 | 51.12 |
| 2 | 91.75 | 24.99 | 54.12 | 34.20 | 50.55 |
| 3 | 91.40 | 24.96 | 54.05 | 34.15 | 51.18 |

- Against the paper's best model: accuracy +3.9, F1 +0.9, IOU +12.5. Welch's t-test against the published mean and standard deviation (5 seeds): accuracy p = 0.007, IOU p < 0.001, **F1 p = 0.06, not significant**. The F1 gain is about one published standard deviation. The stronger test is the paired bootstrap against our own `ni` trained on the end of the review (session 4).
- The rationale matters to the predictor: accuracy is 91.8% with the generator's rationale and 74.1% with a random one.
- Faithfulness measured with the rationale predictor: comprehensiveness 0.034 and sufficiency -0.007 on the paper's scale (paper: 0.02 to 0.04 and -0.01 to -0.04); 0.157 and -0.037 on real probabilities.
- Faithfulness measured with an independent classifier (`judge`: BERT-base trained on full reviews, end of review, seed 3, test accuracy 94.0%), for `frozen` seed 3: comprehensiveness 0.175 and sufficiency 0.045; normalized (Carton et al. 2020) 0.35 and 0.88, defined for 97.2% of the reviews; AOPC 0.135 and 0.089. There is nothing to compare these with yet: session 4 scores `ni` with the same classifier.

  | Selection rate | 1% | 5% | 10% | 20% | 50% |
  | --- | --- | --- | --- | --- | --- |
  | Comprehensiveness | 0.018 | 0.060 | 0.106 | 0.175 | 0.315 |
  | Sufficiency | 0.206 | 0.101 | 0.072 | 0.045 | 0.020 |

- The stopword share of phrases (0.51-0.52) is that of running text, because phrases include the small words between content words, as human rationales do (0.46). Where the attention peaks (single tokens), the frozen generator picks 0.41-0.42 stopwords on average (seed 3: 0.45-0.46); fine-tuned generators pick 0.56-0.71.
- Seed 3 has the lowest single-token accuracy (89.60, against 91.35 and 91.60). A predictor trained on phrases and scored on single tokens varies more between seeds (standard deviation 1.1, against 0.4 with phrases).
- Validation loss was still falling after 5 epochs (in all three seeds, the last epoch was the best).
- A session with two frozen runs and the probe takes about 4 hours on Kaggle; two fine-tuned runs took 6 to 8.

### 4.3 Full pipeline: fine-tuned generator with NI on the end of the review (session 4)

`ni` (fine-tuned generator, NI with p = 0.2, alignment fixed) trained and scored on the end of the review with single tokens, seed 3 (`results_v8`). This is the paper's best configuration in the paper's own protocol. Early stopping kept epoch 1: both validation losses rose after it.

| Setting (same model) | Acc | P | R | F1 | IOU | Stopword share | Span (words) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| End of review, single tokens (as trained) | 92.30 | 22.94 | 49.67 | 31.39 | 35.31 | 0.63 | 2.4 |
| End of review, phrases (w = 5, at evaluation only) | 92.15 | 25.32 | 54.83 | 34.65 | 52.71 | 0.55 | 5.8 |
| End of review, longer phrases (w = 9) | 92.10 | 25.33 | 54.84 | 34.65 | 60.36 | 0.55 | 8.2 |
| Beginning of review, single tokens | 87.85 | 19.34 | 52.90 | 28.33 | 32.35 | 0.63 | 2.5 |
| Beginning of review, phrases | 87.70 | 21.28 | 58.20 | 31.17 | 49.53 | 0.56 | 6.2 |
| End of review, random selection | 71.10 | 9.21 | 19.92 | 12.59 | 7.55 | 0.51 | 1.3 |
| Paper, tuned + NI | 87.9 (1.8) | 24.4 | 52.7 | 33.3 (0.8) | 38.4 (1.9) | | |

- **Trained on the end of the review, the published result is close to reproducible**: F1 31.4 vs 33.3, IOU 35.3 vs 38.4, accuracy 92.3 vs 87.9. Our fine-tuned runs trained on the beginning got 16.4-24.7 F1 there; this one gets 28.3 on the beginning. With one run each, the training window and the seed are confounded.
- **Against FG+PS** (paired bootstrap over the 2,000 test reviews, 10,000 resamples, 95% intervals):
  - FG+PS seed 3 vs `ni` seed 3: F1 +2.76 (+2.32 to +3.21), IOU +15.9, accuracy -0.90 (-2.15 to +0.40, not significant). FG+PS seeds 1 and 2 vs `ni` seed 3: F1 +3.00 and +2.81.
  - **The gain comes from phrase selection, not from freezing.** With single tokens, the frozen generator (seed 3, `tail` probe) vs `ni`: F1 +0.28 (-0.11 to +0.67, not significant). Phrases on `ni` at evaluation: F1 +3.26 (+3.05 to +3.47). FG+PS vs `ni` with phrases: F1 -0.50 (-0.98 to -0.01).
- NI does not steer this generator either: the replacement rate is 1.05-1.06 times p (exposure above 1: the selection holds slightly more noise targets than a random one), and 63% of its single-token selections are stopwords or punctuation (frozen generator: 42%, human rationales: 46%).
- **The judge is not exactly repeatable.** The session-4 `judge` (same settings and seed) reached 93.7% and loss 0.1575, against 94.0% and 0.1514 in session 3. `run_full_text.py` seeds everything, so the cause is probably the GPU or the Kaggle image. `ni` and FG+PS are therefore scored by two classifiers of the same recipe.
- Faithfulness with the independent classifier, `ni` (FG+PS seed 3 in brackets): comprehensiveness 0.335 (0.175), sufficiency 0.029 (0.045), normalized 0.56 (0.35) and 0.90 (0.88), defined for 98.3% of the reviews; AOPC 0.228 (0.135) and 0.117 (0.089). The fine-tuned rationale is twice as comprehensive for a fine-tuned classifier; only its AOPC sufficiency is worse, because of its smallest selections:

  | Selection rate | 1% | 5% | 10% | 20% | 50% |
  | --- | --- | --- | --- | --- | --- |
  | `ni`, comprehensiveness | 0.012 | 0.077 | 0.169 | 0.335 | 0.547 |
  | `ni`, sufficiency | 0.331 | 0.136 | 0.073 | 0.029 | 0.016 |
  | FG+PS, comprehensiveness | 0.018 | 0.060 | 0.106 | 0.175 | 0.315 |
  | FG+PS, sufficiency | 0.206 | 0.101 | 0.072 | 0.045 | 0.020 |

- Faithfulness with its own rationale predictor: comprehensiveness 0.074 and sufficiency -0.035 on the paper's scale; 0.321 and -0.168 on probabilities.
- **What this means for the paper:** freezing does not make the rationale more plausible than a fine-tuned generator with NI trained on the end of the review; phrase selection carries the gain over it. Freezing matches that generator with 2,307 trainable generator parameters, fewer stopwords and less training time. This rests on one seed of the fine-tuned model: sessions 5-7 add `a2r` and two more seeds. Session 5 (4.4) shows the same generator without noise. **Seeds 1 and 2 (sessions 6-7, 4.5-4.6) do not repeat this: there, the frozen generator is 11.9 and 8.7 F1 above `ni`.**

### 4.4 Full pipeline: fine-tuned generator without noise, and frozen generator trained on single tokens (session 5)

Seed 3, trained and scored on the end of the review with single tokens (`results_v9`): `a2r` (fine-tuned, no noise; early stopping kept epoch 5) and `frozen` with `WINDOW = 1`. The other rows are from sessions 3 and 4, same seed.

| Model (end of review, seed 3) | Acc | P | R | F1 | IOU | Stopword share | Acc with a random selection |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `a2r` (fine-tuned, no noise) | 90.45 | 17.03 | 36.87 | 23.30 | 22.19 | 0.58 | 50.00 |
| `a2r`, phrases at evaluation | 89.20 | 18.40 | 39.85 | 25.18 | 37.29 | 0.54 | |
| `ni` (fine-tuned + NI) | 92.30 | 22.94 | 49.67 | 31.39 | 35.31 | 0.63 | 71.10 |
| `ni`, phrases at evaluation | 92.15 | 25.32 | 54.83 | 34.65 | 52.71 | 0.55 | |
| `frozen`, trained on single tokens | 90.80 | 23.14 | 50.10 | 31.66 | 30.94 | 0.45 | 75.15 |
| `frozen`, trained on single tokens, phrases at evaluation | 90.25 | 25.00 | 54.14 | 34.21 | 51.12 | 0.52 | |
| FG+PS (`frozen` trained with w = 5) | 91.40 | 24.96 | 54.05 | 34.15 | 51.18 | 0.52 | 74.20 |
| Paper, tuned (no NI) | 82.4 (8.6) | 20.2 | 43.7 | 27.6 (3.0) | 29.1 (5.3) | | |

Paired bootstrap (seed 3, 2,000 reviews, 10,000 resamples, 95% intervals):
- **NI on a fine-tuned generator:** `ni` - `a2r`: F1 +8.09 (+7.34 to +8.85), accuracy +1.85 (+0.55 to +3.15). With phrases on both: F1 +9.47.
- **Freezing against fine-tuning without noise:** `frozen` (single tokens) - `a2r`: F1 +8.36 (+7.68 to +9.05), accuracy +0.35 (not significant).
- **Freezing against fine-tuning with NI:** `frozen` (single tokens) - `ni`: F1 +0.27 (-0.12 to +0.66, not significant), accuracy -1.50 (-2.85 to -0.20).
- **Phrases at evaluation:** +1.89 F1 on `a2r`, +2.55 on `frozen`, +3.26 on `ni`.
- **Training window of the frozen generator** (single tokens vs phrases during training): the same selection (F1 -0.01 with single tokens, -0.06 with phrases), and the predictor is 1.2 points more accurate when training and evaluation use the same selection.

What it means:
- NI does help a fine-tuned generator: +8.1 F1 here and +3.3 in the seed-1 pair on the beginning of the review (2.1), in line with the paper (+5.7). It does not do so by steering the selection away from the words it replaces: the exposure stays at 1.05-1.06, and `ni` selects more stopwords than `a2r` (0.63 vs 0.58).
- Given a random selection, the `a2r` predictor is at exactly 50.00%, chance on this balanced test set: it reads a code in which tokens are selected, not their content. With NI it keeps 71%, frozen 75%. This fits Storek et al.'s explanation (the noise breaks the code between generator and predictor) rather than token-level steering.
- Freezing reaches the plausibility of NI without noise. Phrase selection adds 2-3 F1 points to any generator, least to the degenerate one.
- Confound: early stopping kept epoch 5 for `a2r` and epoch 1 for `ni`. In the seed-1 pair on the beginning of the review it was the other way round (`a2r` epoch 1, `ni` epoch 3), and NI still helped.
- Seeds 1 and 2 (sessions 6-7, 4.5-4.6) confirm that NI helps, but not that freezing only matches NI: there, freezing is 11.9 and 8.7 F1 above `ni`.

### 4.5 Full pipeline: second seed of both fine-tuned runs (session 7)

Seed 2, trained and scored on the end of the review with single tokens (`results_v10`, Kaggle run of 2026-10-18): `ni` (fine-tuned + NI, p = 0.2, alignment fixed) and `a2r` (fine-tuned, no noise). The configs differ from sessions 4-5 only in the seed. As in seed 3, early stopping kept epoch 1 for `ni` (both validation losses rose after it) and epoch 5 for `a2r`.

| Model (end of review, seed 2) | Acc | P | R | F1 | IOU | Stopword share | Acc with a random selection |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `a2r` (fine-tuned, no noise) | 87.05 | 12.94 | 28.01 | 17.70 | 17.34 | 0.65 | 51.30 |
| `a2r`, phrases at evaluation | 85.25 | 14.82 | 32.09 | 20.27 | 30.26 | 0.54 | |
| `ni` (fine-tuned + NI) | 85.55 | 16.48 | 35.69 | 22.55 | 23.63 | 0.60 | 50.60 |
| `ni`, phrases at evaluation | 83.30 | 18.02 | 39.02 | 24.66 | 35.89 | 0.54 | |
| FG+PS (`frozen` seed 2, `results_v5`) | 91.75 | 24.99 | 54.12 | 34.20 | 50.55 | 0.51 | 73.40 |
| FG+PS model, single tokens at evaluation | 91.60 | 22.86 | 49.51 | 31.28 | 29.98 | 0.38 | |

On the beginning of the review (same models): `a2r` 85.35 / 10.55 / 28.85 / F1 15.45 / IOU 15.15 (phrases: F1 17.40), `ni` 83.10 / 13.32 / 36.43 / F1 19.51 / IOU 20.47 (phrases: F1 21.35). Longer phrases (w = 9, end): `a2r` F1 20.65, `ni` 24.67.

Two seeds of the fine-tuned runs (end of review, single tokens; mean and standard deviation):

| Model | F1 seed 2 | F1 seed 3 | Mean F1 | Mean Acc | Mean IOU |
| --- | --- | --- | --- | --- | --- |
| `a2r` | 17.70 | 23.30 | 20.5 (4.0) | 88.8 (2.4) | 19.8 (3.4) |
| `ni` | 22.55 | 31.39 | 27.0 (6.2) | 88.9 (4.8) | 29.5 (8.3) |
| FG+PS model, single tokens (3 seeds) | 31.28 | 31.67 | 31.5 (0.2) | 90.9 (1.1) | 30.5 (0.5) |
| FG+PS (3 seeds) | 34.20 | 34.15 | 34.2 (0.1) | 91.8 (0.4) | 50.9 (0.3) |
| Paper, tuned + NI (5 seeds) | | | 33.3 (0.8) | 87.9 (1.8) | 38.4 (1.9) |
| Paper, tuned (5 seeds) | | | 27.6 (3.0) | 82.4 (8.6) | 29.1 (5.3) |

Paired bootstrap, seed 2 (2,000 reviews, 10,000 resamples, 95% intervals; seed 3 in brackets). "FG" is the FG+PS model scored with single tokens: no frozen generator was trained with single tokens in seed 2 (in seed 3 the two select the same tokens, F1 31.67 vs 31.66).
- **NI on a fine-tuned generator:** `ni` - `a2r`: F1 +4.85 (+4.51 to +5.20) [+8.09], accuracy -1.50 (-3.10 to +0.10, not significant) [+1.85].
- **Freezing against fine-tuning without noise:** FG - `a2r`: F1 +13.58 (+13.00 to +14.18) [+8.37].
- **Freezing against fine-tuning with NI:** FG - `ni`: F1 +8.73 (+8.13 to +9.36) [+0.28, not significant].
- **FG+PS against `ni`:** F1 +11.65 (+10.98 to +12.32) [+2.76], accuracy +6.20 (+4.70 to +7.75) [-0.90, not significant]. With phrases on `ni` too: F1 +9.54 (+8.80 to +10.30) [-0.50].
- **Phrases at evaluation:** +2.57 F1 on `a2r` [+1.89], +2.11 on `ni` [+3.26], +2.91 on FG [+2.48].

Accuracy on random selections, and the share of reviews the predictor puts in a single class:

| Predictor | Seed 2 | Seed 3 |
| --- | --- | --- |
| `a2r` | 51.30% (98.6% in one class) | 50.00% (99.7% in one class) |
| `ni` | 50.60% (99.0% in one class) | 71.10% |
| FG+PS | 73.40% | 74.20% (seed 1: 74.55%) |

The exposure of `ni` (mean `r` of the selected words, training log) is 1.04, 1.05 and 1.06 in its three epochs (seed 3: 1.05-1.06): NI does not steer this seed either. Stopword share: `ni` 0.60, `a2r` 0.65 (seed 3: 0.63, 0.58), all above a random selection (0.51).

What it means:
- **The fine-tuned runs vary strongly between seeds; the frozen generator does not.** `ni`: F1 31.4 (seed 3) and 22.5 (seed 2); `a2r`: 23.3 and 17.7. The FG+PS model with single tokens: 31.3 to 31.7 over three seeds. The published tuned + NI has a standard deviation of 0.8 over 5 seeds, so seed 2 of our `ni` is far outside the published spread.
- **NI helps in every pair** (+3.3 on the beginning with seed 1, +4.8 and +8.1 on the end with seeds 2 and 3), by varying amounts.
- **The code explanation gets stronger.** In seed 2, NI did not break the code: like `a2r`, its predictor puts 99% of random selections in one class, and `ni` stays 8.7 F1 below the frozen generator. In seed 3, NI broke the code (71.1%) and matched the frozen generator.
- **"The gain comes from phrase selection, not from freezing" (4.3) holds only for seed 3.** In seed 2, freezing adds 8.7 F1 over `ni` and phrase selection 2.9 more. Across seeds 2-3: the frozen generator is at least as plausible as `ni` in each seed, and phrase selection adds 2-3 F1 to every generator. This supports the title "Freeze, Don't Perturb" better than seed 3 alone.
- Confound, unchanged: early stopping kept epoch 1 for `ni` and epoch 5 for `a2r` in both seeds.
- Open: our `ni` fixes the alignment defect (2.4), which on the beginning of the review made NI more plausible (seed 1: F1 24.7 with the defect, 19.6 without). Whether the published runs had the defect, and whether it explains their small seed variation, is untested on the end of the review (6, item 11).

Paper (updated 2026-10-19 after session 7, superseded by the three-seed update in 4.6): reproduced rows of Tables 1, 2 and 5 were means over seeds 2-3; new appendix table of paired differences per seed (App. B, Table 6); abstract, introduction, contributions, Section 7.1, Discussion, Conclusion and Limitations rewritten for the two seeds; R/P counts in App. C now 28 (beginning) and 49 (end). The conclusion still ends on page 8, with `\drafttrue` and `\draftfalse`.

### 4.6 Full pipeline: third seed of both fine-tuned runs (session 6), and the three-seed picture

Seed 1, trained and scored on the end of the review with single tokens (`results_v11`, Kaggle run of 2026-10-13, added to the repo on 2026-10-19): `ni` and `a2r`, configs identical to sessions 4-5 and 7 except the seed. Early stopping kept epoch 1 for both (`a2r` kept epoch 5 in seeds 2 and 3).

| Model (end of review, seed 1) | Acc | P | R | F1 | IOU | Stopword share | Acc with a random selection |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `a2r` (fine-tuned, no noise) | 89.90 | 12.54 | 27.15 | 17.16 | 17.50 | 0.70 | 50.70 (99.1% in one class) |
| `a2r`, phrases at evaluation | 66.40 | 15.94 | 34.50 | 21.80 | 34.26 | 0.56 | |
| `ni` (fine-tuned + NI) | 87.95 | 14.27 | 30.90 | 19.52 | 22.22 | 0.64 | 50.45 (99.2% in one class) |
| `ni`, phrases at evaluation | 83.65 | 15.93 | 34.50 | 21.80 | 35.68 | 0.55 | |
| FG+PS (`frozen` seed 1, `results_v3`) | 92.25 | 25.13 | 54.41 | 34.38 | 51.12 | 0.52 | 74.55 |
| FG+PS model, single tokens at evaluation | 91.35 | 22.97 | 49.74 | 31.43 | 30.50 | 0.42 | |

On the beginning of the review: `a2r` 86.10 / 10.64 / 29.09 / F1 15.58 / IOU 16.00; `ni` 84.00 / 11.32 / 30.95 / F1 16.58 / IOU 19.03. Exposure of `ni`: 1.02, 1.06 and 1.08 in its three epochs. Faithfulness with its own predictor is near zero: comprehensiveness 0.0006 (`a2r`) and 0.006 (`ni`) on the paper's scale.

Phrases at evaluation cost the `a2r` predictor 23.5 accuracy points (89.90 -> 66.40; on the beginning 86.10 -> 60.10): it was trained on single tokens and reads a code in them.

Paired bootstrap, seed 1 (95% intervals):
- `ni` - `a2r`: F1 +2.37 (+1.93 to +2.82), accuracy -1.95 (-3.30 to -0.60).
- FG - `a2r`: F1 +14.27 (+13.67 to +14.88). FG - `ni`: F1 +11.91 (+11.16 to +12.65), accuracy +3.40.
- FG+PS - `ni`: F1 +14.86 (+14.07 to +15.65), accuracy +4.30 (+2.85 to +5.75). FG+PS - `ni` with phrases: F1 +12.58.
- Phrases at evaluation: +4.65 F1 on `a2r`, +2.27 on `ni`, +2.95 on FG.

**Three seeds of the fine-tuned runs** (end of review, single tokens; mean and standard deviation; the paper's Table 2):

| Model | F1 seed 1 | F1 seed 2 | F1 seed 3 | Mean F1 | Mean Acc | Mean IOU |
| --- | --- | --- | --- | --- | --- | --- |
| `a2r` | 17.16 | 17.70 | 23.30 | 19.4 (3.4) | 89.1 (1.8) | 19.0 (2.8) |
| `ni` | 19.52 | 22.55 | 31.39 | 24.5 (6.2) | 88.6 (3.4) | 27.1 (7.2) |
| FG+PS model, single tokens | 31.43 | 31.28 | 31.67 | 31.5 (0.2) | 90.9 (1.1) | 30.5 (0.5) |
| FG+PS | 34.38 | 34.20 | 34.15 | 34.2 (0.1) | 91.8 (0.4) | 50.9 (0.3) |
| Paper, tuned + NI (5 seeds) | | | | 33.3 (0.8) | 87.9 (1.8) | 38.4 (1.9) |
| Paper, tuned (5 seeds) | | | | 27.6 (3.0) | 82.4 (8.6) | 29.1 (5.3) |

Paired differences, mean over the three seeds (per seed with intervals: paper Table 6): `ni` - `a2r` F1 +5.10; FG - `a2r` +12.08; FG - `ni` +6.97 (+11.91, +8.73, +0.28); FG+PS - `ni` +9.76, accuracy +3.20; phrases at evaluation +3.03 (`a2r`), +2.55 (`ni`), +2.78 (FG).

What it means:
- **Our reproduction of the published best model (fine-tuned + NI) reaches the published F1 in one of three seeds.** Mean F1 24.5 (sd 6.2) against 33.3 (sd 0.8). Seed 3 was the outlier.
- **NI helps in every pair** (+2.4, +4.8, +8.1 on the end; +3.3 on the beginning), but **it broke the code in only one of three seeds**: the `ni` predictor is at chance on random selections in seeds 1 and 2 (99% in one class, like `a2r` in all three seeds) and at 71.1% in seed 3, the only seed in which `ni` matches the frozen generator.
- **The frozen generator is at least as plausible as `ni` in every seed** (+11.9, +8.7, +0.3 F1) and far more stable. This is the paper's main claim now, and it supports the title "Freeze, Don't Perturb".
- Phrase selection adds about 3 F1 to every generator (2.5-3.0 on average), but on a fine-tuned predictor that was trained on single tokens it can cost a lot of accuracy (`a2r` seed 1: -23.5).
- Confound: in seed 1, both fine-tuned runs kept epoch 1 and NI still helped (+2.4, the smallest effect); in seeds 2-3, `a2r` kept epoch 5.
- Open, more relevant now: our `ni` fixes the alignment defect. If the published runs had it, it might explain why they are both better and far more stable (6, item 11).

Paper (updated 2026-10-19 after session 6): reproduced rows of Tables 1, 2 and 5 are three-seed means; Table 6 (App. B) has all three seeds plus the mean; abstract unchanged; introduction (four paired runs, two of three seeds keep the code), Section 4.1 (exposure 1.02-1.08), Section 7.1 (rewritten around three-seed means), Discussion, Limitations (reproduction falls short of the published mean) and App. C (R/P counts: 32 beginning, 57 end) updated; all session-6/7 TODOs removed. The conclusion still ends on page 8 in both draft and final mode.

### 4.7 Full pipeline: NI as released on the end of the review (session 8)

**Note (2026-10-22): seeds 1 and 3 (4.8) show that seed 2 was the best case for NI as released. The conclusions below that rest on seed 2 alone ("the defect explains most of the gap", "NI as released beats the frozen generator") no longer hold.**

Seed 2, trained and scored on the end of the review with single tokens (`results_v12`, Kaggle run of 2026-10-19/20): `released`, the fine-tuned generator with NI (p = 0.2) and the alignment defect of the released code (`--legacy_alignment`). Its config differs from `ni` seed 2 (`results_v10`) only in `legacy_alignment`, and training is deterministic given the seed, so the pair isolates the defect. Early stopping kept epoch 1, as for `ni` (stopped after epoch 3; 5.0 h including evaluation). Exposure 1.025-1.028 (measured on the words the noise replaces, which with the defect are the words before the selected tokens).

The defect: during noisy training, the word ids are computed with `[CLS]` still in place, so the rationale predictor receives the word of the token *before* each selected token. Validation and test use the selected tokens, in the released code too (checked in its original commit, 334e888).

| Model (end of review, seed 2) | Acc | P | R | F1 | IOU | Stopword share | Acc with a random selection |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `released` (NI with the defect) | 92.00 | 23.78 | 51.50 | 32.54 | 36.17 | 0.56 | 67.85 (65.9% in one class) |
| `released`, phrases at evaluation (w = 5) | 91.40 | 25.87 | 56.02 | 35.40 | 53.21 | 0.53 | |
| `released`, w = 9 | 91.45 | 25.79 | 55.84 | 35.28 | 60.82 | 0.53 | |
| `ni` (NI with the fix, `results_v10`) | 85.55 | 16.48 | 35.69 | 22.55 | 23.63 | 0.60 | 50.60 (99.0% in one class) |
| `a2r` (`results_v10`) | 87.05 | 12.94 | 28.01 | 17.70 | 17.34 | 0.65 | 51.30 (98.6% in one class) |
| FG+PS model, single tokens (`results_v5`) | 91.60 | 22.86 | 49.51 | 31.28 | 29.98 | 0.38 | |
| FG+PS (`results_v5`) | 91.75 | 24.99 | 54.12 | 34.20 | 50.55 | 0.51 | 73.40 |
| Paper, tuned + NI (5 seeds) | 87.9 (1.8) | 24.4 | 52.7 | 33.3 (0.8) | 38.4 (1.9) | | |

On the beginning of the review (same model): 88.20 / 20.01 / 54.74 / F1 29.31 / IOU 32.92, stopword share 0.56; with phrases F1 31.79 / IOU 49.49 (FG+PS seed 2 there: 31.00 / 46.65). Faithfulness with its own predictor: comprehensiveness 0.052 and sufficiency -0.032 on the paper's sigmoid scale, 0.225 and -0.150 on probabilities. No `judge` in this session.

Paired bootstrap, seed 2 (95% intervals; paper Table 6):
- **The defect:** `released` - `ni`: F1 +9.99 (+9.39 to +10.59), accuracy +6.45 (+4.90 to +8.05).
- `released` - `a2r`: F1 +14.84 (+14.27 to +15.40), accuracy +4.95 (+3.30 to +6.60).
- FG - `released`: F1 -1.26 (-1.60 to -0.90), accuracy -0.40 (not significant).
- FG+PS - `released`: F1 +1.66 (+1.24 to +2.07), accuracy -0.25 (not significant).
- Phrases at evaluation on `released`: F1 +2.86 (+2.67 to +3.05), accuracy -0.60 (not significant).
- FG+PS - `released` with phrases: F1 -1.20 (-1.64 to -0.76), accuracy +0.35 (not significant).
- On the beginning of the review, seed 1 (both trained there, `results_v1`): `ni_released` - `ni` F1 +5.03 (+4.64 to +5.42), accuracy -3.40 (-5.20 to -1.55).

What it means:
- **The defect explains most of the gap to the published model.** With it, NI reaches F1 32.5 (published 33.3 +- 0.8), IOU 36.2 (38.4 +- 1.9) and accuracy 92.0 (87.9) in the seed in which NI with the fix reaches only 22.5. The published runs used the released code, so they presumably had the defect. In both pairs we have, the defect adds more F1 than the noise itself does over no noise (beginning, seed 1: +5.0 vs +3.3; end, seed 2: +10.0 vs +4.9).
- **It breaks the code where the noise alone did not.** The `released` predictor is 67.9% accurate on random selections and predicts both classes; `ni` and `a2r` of the same seed put 99% of the reviews in one class. Trained on the words before the selected tokens, the predictor cannot learn which tokens the generator selects. NI still does not steer the generator (exposure 1.03; stopword share 0.56, above a random selection's 0.51).
- **NI as released beats the frozen generator in this seed:** +1.3 F1 with single tokens, +1.2 with phrases on both (F1 35.4 vs 34.2, IOU 53.2 vs 50.6), at equal accuracy. FG+PS stays 1.7 above `released` with single tokens, and above the published numbers. "The frozen generator is at least as plausible as NI" now holds for NI as described (with the fix) only.
- Open: one seed. Whether NI as released is as stable across seeds as the published sd of 0.8 suggests, and whether the defect helps without noise, are untested (6, item 11).

Paper (updated 2026-10-21 after session 8): the terms "NI as described" (+NI, defect fixed) and "NI as released" (+NI_released) are used throughout. Abstract (new defect sentence; the frozen generator is "more plausible and far more stable than NI without the defect, and nearly as plausible as with it"; the sentence on evidence noise and the "3-4 F1 points" clause were dropped to keep the length), introduction paragraphs 3-4, contributions, Related Work, Sections 4.2, 5.1, 6 (Models, Implementation), 7.1 (new paragraph), Discussion, Conclusion, Limitations, App. A (item 3) updated; new rows in Table 1 (`released` evaluated on the beginning) and Table 2 (`released`, with and without phrases); new App. B Table 6 (paired differences, seed 2). To keep the conclusion on page 8, the noise-on-FG table and its detailed text moved to a new App. C ("Noise on the Frozen Generator"; Section 7.3 keeps a short summary), the determinism statement moved to Section 6 (Implementation), and a few sentences were shortened (faithfulness: the top-1% detail was cut). Run times corrected from the result-file timestamps: an epoch takes 0.7-0.9 h with the frozen generator and 1.3-1.8 h fine-tuned (1.4-1.8 until the audit of 2026-10-22: the fastest fine-tuned run took 1.3495 h); a run takes 3.5-4.6 h and 4.2-8.2 h (the paper said 4-4.6 h and 6-8 h). App. D R/P counts: 34 beginning, 61 end. The conclusion still ends on page 8 in draft and final mode.

### 4.8 Full pipeline: NI as released, seeds 1 and 3 (sessions 9-10), and the three-seed picture

Seeds 1 and 3 with the settings of session 8 (`results_v13` = seed 1, `results_v14` = seed 3; Kaggle runs started 2026-10-21). The configs differ from `released` seed 2 only in the seed, and from `ni` of the same seed only in `legacy_alignment`. Early stopping kept epoch 1 in all three seeds (4.9-5.2 h per run). Exposure 1.01-1.03.

| Model (end of review, single tokens) | F1 seed 1 | F1 seed 2 | F1 seed 3 | Mean F1 (sd) | Mean Acc (sd) | Mean IOU (sd) | Acc with a random selection (seeds 1 / 2 / 3) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `a2r` | 17.16 | 17.70 | 23.30 | 19.4 (3.4) | 89.1 (1.8) | 19.0 (2.8) | 50.70 / 51.30 / 50.00 |
| `ni` (NI as described) | 19.52 | 22.55 | 31.39 | 24.5 (6.2) | 88.6 (3.4) | 27.1 (7.2) | 50.45 / 50.60 / 71.10 |
| `released` (NI as released) | 23.31 | 32.54 | 27.07 | 27.6 (4.6) | 87.1 (4.4) | 30.9 (4.8) | 56.50 / 67.85 / 67.85 |
| FG+PS model, single tokens | 31.43 | 31.28 | 31.67 | 31.5 (0.2) | 90.9 (1.1) | 30.5 (0.5) | |
| FG+PS | 34.38 | 34.20 | 34.15 | 34.2 (0.1) | 91.8 (0.4) | 50.9 (0.3) | 74.55 / 73.40 / 74.20 |
| `released`, phrases at evaluation | 25.29 | 35.40 | 30.17 | 30.3 (5.1) | 88.4 (2.7) | 46.6 (6.5) | |
| Paper, tuned + NI (5 seeds) | | | | 33.3 (0.8) | 87.9 (1.8) | 38.4 (1.9) | |

Per seed, `released`: seed 1 86.00 / 17.04 / 36.90 / F1 23.31 / IOU 26.89, stopword share 0.63, predictor on random selections 56.50% (91.6% in one class); seed 3 83.35 / 19.78 / 42.84 / F1 27.07 / IOU 29.52, stopword share 0.67, random selections 67.85% (65.7% in one class). On the beginning of the review: F1 20.24, 29.31 and 24.10 in seeds 1-3; mean 84.7 / 16.8 / 45.8 / F1 24.5 / IOU 27.6, stopword share 0.63 (paper Table 1). Faithfulness with its own predictor (sigmoid scale): comprehensiveness 0.016 / 0.052 / 0.040, sufficiency -0.058 / -0.032 / -0.015.

Paired bootstrap per seed (95% intervals in paper Table 6), seeds 1 / 2 / 3 and mean:
- **The defect** (`released` - `ni`): F1 +3.79 (+3.46 to +4.13) / +9.99 / -4.32 (-4.74 to -3.90), mean +3.15; accuracy -1.95 / +6.45 / -8.95, mean -1.48.
- `released` - `a2r`: F1 +6.16 / +14.84 / +3.77, mean +8.26.
- FG - `released`: F1 +8.12 (+7.47 to +8.76) / -1.26 / +4.60 (+4.15 to +5.05), mean +3.82; accuracy +5.35 / -0.40 / +6.25.
- FG+PS - `released`: F1 +11.07 / +1.66 / +7.08, mean +6.60; accuracy +6.25 / -0.25 / +8.05.
- Phrases at evaluation on `released`: F1 +1.97 / +2.86 / +3.11, mean +2.65.
- FG+PS - `released` with phrases: F1 +9.09 / -1.20 / +3.97, mean +3.95.

What it means:
- **Seed 2 was the best case.** NI as released varies almost as much between seeds as NI as described (F1 23.3-32.5, sd 4.6, against 19.5-31.4, sd 6.2) and falls short of the published 33.3 (sd 0.8) on average. The defect explains neither the published level nor its small spread; we do not know what else differed in the published runs.
- **The defect has no consistent sign:** F1 +3.8, +10.0 and -4.3 in seeds 1-3 (+3.2 on average); accuracy -2.0, +6.5 and -9.0.
- **Plausibility follows the code.** Of the nine fine-tuned runs on the end of the review, the six whose predictor puts more than 90% of random selections in one class (`a2r` in all seeds, `ni` seeds 1-2, `released` seed 1) reach F1 17.2-23.3; the three whose predictor reads no code (`ni` seed 3, `released` seeds 2-3) reach 27.1-32.5. The frozen generator reads no code in any seed (73.4-74.6%) and reaches 31.3-31.7.
- **The frozen generator is more plausible than NI as released in seeds 1 and 3 and by 3.8 F1 on average, and FG+PS beats it in every seed** (+1.7 to +11.1). The session-8 reading (NI as released 1.3 F1 above the frozen generator) was the exception. The title "Freeze, Don't Perturb" fits again.
- Phrase selection adds 2.0-3.1 F1 to NI as released, as to every other generator.
- Every fine-tuned run still selects more stopwords than a random selection (`released`: 0.63, 0.56, 0.67 against 0.51).

Paper (updated 2026-10-22 after sessions 9-10): NI as released is a three-seed mean in Tables 1 and 2 (with sd); App. B Table 6 has all three seeds plus the mean, in the format of Table 5; abstract ("Its benefit varies strongly between seeds, with or without a defect of the released code ... Rationales are plausible only in the runs in which the predictor no longer reads a code ... the frozen generator is more plausible on average than noise injection, with or without the defect"), introduction paragraphs 3-4, contributions (the defect changes F1 by -4 to +10), Related Work, Sections 4.2, 5.1, 6 (Models), 7.1, Discussion (new: plausibility follows the code across the nine fine-tuned runs), Recommendations (also report the predictor's accuracy on random selections), Conclusion, Limitations, App. A and App. D (R/P counts: 38 beginning, 69 end). A few sentences were shortened to keep the conclusion on page 8 in draft and final mode (Discussion, 7.1, faithfulness, "Window versus model").

### 4.2 Supporting small-scale results

- A standard IMDB classifier reaches 95.4% from the human rationale alone (about 35 tokens) and 89.3% from the full review. Better rationales should raise accuracy further. (Raw output not preserved; see 6, item 8.)
- Attention restricted to stopwords and punctuation classifies as well as free attention on the top layer of BERT, pretrained (81.8% vs 81.8%) or fine-tuned (90.3% vs 89.1%). (Raw output not preserved; see 6, item 8.)

---

## 5. What did not work

- **Changing the noise level on the rationale** (fixed or closed loop): the selected words stay as exposed to noise as random ones. (Fixed NI against no noise does make a fine-tuned generator more plausible: 4.4.)
- **Predictor disagreement as a control signal**: it does not move when the rationale degenerates.
- **Pooling token-level embeddings with a fine-tuned scorer**: the scorer codes the label into its choice of tokens. The attention predictor reached 89.3% while selecting 57% stopwords; F1 18.2 (beginning of the review).
- **Evidence noise on a fine-tuned encoder**: no gain at low levels, collapse at high levels.
- **Strong TF*IDF-guided noise**: it pushes the attention to rare words, not to evidence.
- **An absolute exposure target** in the full pipeline: the controller kept the noise almost off (p at most 0.06).

---

## 6. Open questions and next runs

1. ~~Does evidence noise help in the full pipeline?~~ Answered (3.4): no. Seed 1 gave +0.3 F1, seed 2 gave -0.4, and accuracy fell by about 1 point in both.
2. **A self-calibrating controller.** Proposed, not implemented: train a twin pair of linear layers on clean evidence in the same run and let the controller hold the *ratio* of the two exposures. This removes the dependence on an absolute target.
3. **Seeds.** `frozen` and the fine-tuned runs on the end of the review (`ni`, `a2r`) have three seeds; `frozen_en` and `frozen_ni` have two; the runs on the beginning have one. The paper reports 5.
4. **Faithfulness with an independent classifier** (`judge` and `rescore` stages). Done for `frozen` seed 3 (4.1). Session 4 scores `ni`, trained on the end of the review, with the same classifier.
5. **A second dataset** (ERASER Movies; MultiRC and FEVER use a separate code path that has not been updated).
6. ~~A fine-tuned generator on the end of the review.~~ Done in session 4 (4.3): it is as plausible as the frozen generator with single tokens, and with phrases it matches FG+PS.
7. **More epochs** for the frozen generator: the paper trains its fixed generator for 10.
8. **Provenance of the small-scale results.** Their scripts and result files (`explore2.json`, `explore3.json`, `explore3.py`, `lr_check.py`) are only in a temporary folder (`/private/tmp/claude-501/-Users-wmano-github-noise-injection/1756dbf0-3a41-4afc-997f-19c8edf513d2/scratchpad/probe/`) and should be copied into the repo. The evidence-noise sweep (3.4; paper Table 9) matches them exactly. The restricted-attention test (2.2 and 4.2; paper Table 8) and the 95.4% vs 89.3% comparison have no surviving raw output, and the cached BERT states are deleted: re-run them before submission or drop them.
9. **Fine-tuned against frozen on the end of the review.** Seed 3 (sessions 4-5, 4.3-4.4): NI raises the fine-tuned generator's F1 by 8.1, and freezing matches NI without noise. Seed 2 (session 7, 4.5): NI raises F1 by 4.8, and freezing is 8.7 above NI. Seed 1 (session 6, 4.6): NI +2.4, freezing 11.9 above NI. Answered: freezing is at least as plausible as NI in every seed.
10. **Framing of the paper** (updated 2026-10-22 after sessions 9-10): NI helps a fine-tuned generator, but not by steering it, and by an amount that varies strongly between seeds, with or without the alignment defect of the released code; plausibility follows whether the predictor still reads a code in the selection (4.8); neither version reproduces the published 33.3 +- 0.8 on average (24.5 as described, 27.6 as released); the frozen generator is more plausible on average than either version and far more stable; phrase selection adds about 3 F1 to every generator. The title "Freeze, Don't Perturb" fits.
11. ~~**Alignment defect on the end of the review.**~~ Answered (sessions 8-10, 4.7-4.8): the defect changes F1 by +3.8, +10.0 and -4.3 in seeds 1-3 and does not explain the published results. The optional run without noise (session 11) was dropped: with effects this variable between seeds, one seed of it would not be conclusive.

---

## 7. How to run

Phrase selection on the end of the review is the best setting so far, with a frozen generator (three seeds); a fine-tuned generator with NI and phrases at evaluation matched it in seed 3 only (4.5, 4.6), and NI as released with phrases at evaluation beat it by 1.2 F1 in seed 2 but fell 9.1 and 4.0 F1 below it in seeds 1 and 3 (4.8). On Kaggle, at the top of the pasted `kaggle_run.py` cell:

```python
STAGES = ["setup", "frozen", "probe", "pack"]
BATCH = 16
TRUNCATION = "left"
WINDOW = 5
```

For another seed, set `SEED = 2`. The remaining sessions for the paper:

| Session | Settings | Gives |
| --- | --- | --- |
| 3 (done, 2026-10-12) | `SEED = 3`, `WINDOW = 5`, `STAGES = ["setup", ["frozen", "judge"], "probe", "rescore", "pack"]` | third seed of the method; faithfulness from an independent classifier (`judge`) |
| 4 (done, 2026-10-13) | `SEED = 3`, `WINDOW = 1`, `STAGES = ["setup", ["ni", "judge"], "probe", "rescore", "pack"]` | the paper's best configuration (fine-tuned + NI) trained on the end of the review, with faithfulness from a second `judge` (4.3) |
| 5 (done, 2026-10-13) | `SEED = 3`, `WINDOW = 1`, `STAGES = ["setup", ["a2r", "frozen"], "probe", "pack"]` | `a2r` on the end of the review, paired with `ni` seed 3: does NI help a fine-tuned generator there? Frozen generator trained on single tokens |
| 6 (done, run 2026-10-13, `results_v11`) | `SEED = 1`, `WINDOW = 1`, `STAGES = ["setup", ["ni", "a2r"], "probe", "pack"]` | first seed of both fine-tuned runs on the end of the review (4.6) |
| 7 (done, run 2026-10-18, `results_v10`) | `SEED = 2`, `WINDOW = 1`, `STAGES = ["setup", ["ni", "a2r"], "probe", "pack"]` | second seed of both (4.5) |
| 8 (done, run 2026-10-19, `results_v12`) | `SEED = 2`, `WINDOW = 1`, `STAGES = ["setup", "released", "probe", "pack"]` | NI with the alignment defect on the end of the review, paired with `ni` seed 2 (4.7) |
| 9 (done, run 2026-10-21, `results_v13`) | `SEED = 1`, `WINDOW = 1`, `STAGES = ["setup", "released", "probe", "pack"]` | NI as released, seed 1, paired with `ni`, `a2r` and `frozen` seed 1 (4.8) |
| 10 (done, run 2026-10-21, `results_v14`) | `SEED = 3`, `WINDOW = 1`, `STAGES = ["setup", "released", "probe", "pack"]` | the same for seed 3 (4.8) |
| 11 (dropped) | `SEED = 2`, `WINDOW = 1`, `STAGES = ["setup"]`, then `train_run("shift", "--inject_noise", "--noise_p", 0, "--legacy_alignment")` and `run("probe", "pack")` | the defect without noise; dropped because one seed of it would not be conclusive (6, item 11) |

Dropped: a third seed of `frozen_en` and `frozen_ni` (noise on the frozen generator is settled; the open question is now the fine-tuned baseline).

All keep `BATCH = 16` and `TRUNCATION = "left"`. `frozen_ni` and `frozen_en` are ablations. The `judge` must train in the same session as the models it scores, because `rescore` only sees that session's models, and its training is not exactly repeatable across sessions.

Directly:

```
python run_words.py --train --evaluate --freeze_encoder_bb \
    --truncation_side left --selection_window 5 \
    --data_path ../usr_movie_review --save_path trained/frozen_seed1 --seed 1
```

Add `--inject_noise --noise_target evidence --noise_p 0.3` for noise in the generator's evidence.
