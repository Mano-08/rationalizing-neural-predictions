# Neurosymbolic AI for explainability in this project

Research notes, 22 October 2026. The question: can neurosymbolic (NeSy) AI improve the explainability of our unsupervised rationalization models (BERT-A2R, frozen generator, phrase selection) on USR Movie Review?

This file combines a literature review with five offline experiments on our own saved test outputs (`results/`). The experiments need no GPU and no new training; their scripts are in `nesy_probes/` (Section 5).

---

## TL;DR

**Yes, but not as a way to raise plausibility.** NeSy is useful here in two roles: reading the rationale transparently, and detecting coded rationales without human annotations.

- **Symbolic knowledge does not improve the frozen generator's rationales.** Multiplying its attention by a sentiment-lexicon prior changes token F1 by −0.09 to +0.13 points (prior strength chosen on half of the test reviews, measured on the other half). Its selections already contain about three times the review's share of lexicon words (22% vs 8%), so the lexicon adds nothing.
- **A purely symbolic rationale is weak.** Lexicon words plus five-token phrases reach F1 19–22, about A2R's level (19.4) and far below FG+PS (34.2). Sentiment-lexicon words make up only 17–23% of the wordpieces humans highlight.
- **Linguistic structure trades F1 for IOU.** Selecting whole clauses instead of a five-token window lowers F1 by about 1.3 points and raises IOU-F1 by about 9, the same trade-off as a wider window.
- **A transparent reader works well on good rationales.** A Naive Bayes model, whose decision is a readable sum of word weights, classifies 88.0% of reviews correctly from the frozen generator's single-token rationales. That beats reading the whole review window (82.2%) and is 2.8 points below the BERT predictor. So "neural selection, symbolic reading" is viable: the explanation becomes the rationale plus the word weights that decided it.
- **The same reader detects codes without human rationales.** On coded rationales, the transparent reader falls to 59–72%, below a random 20% selection (77.3%), while the run's own BERT predictor still reads 86–90%. The gap between the two (the **code gap**) is 16.7–28.2 points for all six runs whose predictor reads a code and −1.5 to 3.4 for all six that do not. Across the 12 runs it correlates with token F1 at r = −0.86, and the reader's accuracy at r = 0.91. It needs no human rationales, which USR's development set lacks (see the paper's Limitations).

**Recommendation.** Don't add NeSy machinery to raise plausibility. Do consider (1) reporting the transparent reader and the code gap as an annotation-free diagnostic (an appendix table in the current paper; no new runs needed), and (2) a follow-up study of a frozen neural selector with a transparent symbolic reader as the predictor (Section 4).

---

## 1. The question in this project's terms

### 1.1 What we have

BERT-A2R has three parts: a generator that scores tokens, top-k selection (20% of the tokens), and a rationale predictor that classifies from the selected tokens only. Plausibility is the agreement of the selection with human rationales (token F1, IOU-F1). Our paper's findings that matter here:

- **Fine-tuned generators form codes.** The predictor reads the label from *which* tokens are selected, not from their content. On random selections, the A2R predictor puts more than 98% of reviews into one class.
- **Plausibility follows the code.** Across the nine fine-tuned runs trained on the review endings (the TAIL window), the six with a code reach F1 17.2–23.3, the three without one 27.1–32.5.
- **Freezing prevents the code.** The frozen generator (FG) reaches F1 31.3–31.7 in every seed.
- **Phrase selection (PS) adds about three F1 points to any generator.**

### 1.2 Where neurosymbolic AI could enter

| Entry point | Example techniques | Kautz type ([Garcez & Lamb, 2023](https://arxiv.org/abs/2012.05876)) |
|---|---|---|
| Symbolic knowledge as a training constraint on the generator | logic-rule distillation, semantic loss, posterior regularization, lexicon priors | Neuro:Symbolic → Neuro |
| Symbolic structure in the selection | constrained structured inference, clause or constituent units | Neuro<sub>Symbolic</sub> (structure compiled into a layer) |
| A symbolic or transparent predictor that reads the rationale | lexicon rules, sparse linear models, logic rule learners | Neuro \| Symbolic (neural perception, symbolic decision) |
| Symbolic explanations of the trained model | logic explained networks, rule extraction | post-hoc, model-level |
| Symbolic checks for evaluation | simple "student" readers, leakage tests | evaluation |

**An architectural detail makes the predictor swap easy.** In BERT-A2R the rationale predictor reaches the generator only through the stop-gradient term sg[ŷ<sup>r</sup>] (Eq. 1 of the paper). Replacing the BERT predictor with a non-differentiable symbolic reader therefore leaves the generator's training unchanged in form. For the frozen generator, whose selection is shaped mainly by the attention predictor's cross-entropy, the choice of reader barely affects what is selected.

---

## 2. What the literature says

### 2.1 Knowledge as training constraints

- **Logic-rule distillation.** [Hu et al. (ACL 2016)](https://arxiv.org/abs/1603.06318) distill first-order rules into CNN/RNN classifiers through a rule-regularized teacher (posterior regularization). Their sentiment rule is "A-but-B takes the sentiment of B"; they report gains on sentiment and NER.
- **Caution for contextual encoders.** [Krishna, Jyothi & Iyyer (EMNLP 2018)](https://aclanthology.org/D18-1505/) re-examined this with many more random seeds. The rule-distillation model performs poorly, and contextual ELMo embeddings capture A-but-B behaviour implicitly. A BERT-based generator is likely to show the same.
- **Logic compiled into the network.** [Li & Srikumar (ACL 2019)](https://aclanthology.org/P19-1028/) compile first-order statements into extra computation-graph nodes, including attention layers, without new parameters. Gains are largest in low-data regimes. This is the closest existing mechanism for constraining a generator's attention with rules.
- **Differentiable constraints.** The [semantic loss (Xu et al., ICML 2018)](https://proceedings.mlr.press/v80/xu18h.html) turns propositional constraints into a training loss. [Logic Tensor Networks (Badreddine et al., AIJ 2022)](https://arxiv.org/abs/2012.13635) ground a fuzzy first-order logic in neural networks.
- **Lexicon priors.** [Zeng & Song (EACL 2021)](https://aclanthology.org/2021.eacl-main.285) use an opinion lexicon with posterior regularization for weakly supervised sentiment analysis.
- **Knowledge bases mostly help weak models.** [Kocoń et al. (ICCS 2022)](https://www.iccs-meeting.org/archive/iccs2022/papers/133510628.pdf) find that external knowledge bases help simple models; with strong transformers the gains are limited and concentrated on hard examples.
- **Constraining explanations needs masks.** [Ross et al. (IJCAI 2017)](https://arxiv.org/abs/1703.03717) penalize input gradients outside annotated regions, which requires annotation masks we don't have (USR is unsupervised).
- **Gap.** We found no work that puts logic constraints directly on a rationale generator's selection mask. Our experiment 3.3 suggests why: at least for lexicon knowledge, the frozen generator already knows what the constraint would add.

### 2.2 Structure in the selection

- **Structured rationale selection.** [SPECTRA (Guerreiro & Martins, EMNLP 2021)](https://aclanthology.org/2021.emnlp-main.525) extracts rationales by constrained inference on a factor graph, used as a differentiable layer, so budget and contiguity constraints become part of the model. Our phrase selection is a parameter-free, soft version of a contiguity constraint.
- **Our dataset began with structured inference.** [Pang & Lee (ACL 2004)](https://aclanthology.org/P04-1035/), who built the movie-review polarity data, classified only the subjective sentences of each review, found with minimum-cut graph inference over sentence-level scores.
- **Compositional sentiment over parse trees.** [SentiBERT (Yin et al., ACL 2020)](https://aclanthology.org/2020.acl-main.341) composes BERT representations over binary constituency trees and handles negation and contrast better than baselines. It shows symbolic syntax helping sentiment composition, but it is not a rationale extractor.

### 2.3 Symbolic or transparent predictors (interpretable by design)

- **Neurosymbolic sentiment.** [SenticNet 7 (Cambria et al., LREC 2022)](https://aclanthology.org/2022.lrec-1.408) builds a commonsense knowledge base with subsymbolic tools and extracts polarity from it in a fully interpretable way. It is a complete sentiment system with its own resources, not a rationale extractor.
- **Rule learners for text.** [RuleNN (Sen et al., EMNLP 2020)](https://aclanthology.org/2020.emnlp-main.345) learns first-order rules over linguistic predicates for sentence classification, roughly matching recurrent networks. [Logical Neural Networks for ILP (Sen et al., AAAI 2022)](https://ojs.aaai.org/index.php/AAAI/article/view/20795) learn readable first-order rules by gradient descent.
- **Concept bottlenecks.** [Text Bottleneck Models (Ludan et al.)](https://arxiv.org/abs/2310.19660) predict from a small set of LLM-generated concepts through a linear layer. [CB-LLM (Sun et al., ICLR 2025)](https://arxiv.org/abs/2412.07992) adds a concept bottleneck layer to an LLM. Both need an LLM to propose and score concepts.
- **LLMs at training time only.** [Aug-imodels (Singh et al., Nature Communications 2023)](https://www.nature.com/articles/s41467-023-43713-1) use an LLM only during fitting to build transparent linear models and trees for text classification. [LLM-based Symbolic Programs (Wang et al., ICLR 2025)](https://arxiv.org/abs/2406.17224) combine LLM-defined natural-language concepts into a symbolic decision program.
- **The classic transparent baseline.** [Naive Bayes log-count ratios (Wang & Manning, ACL 2012)](https://aclanthology.org/P12-2018/) are strong for sentiment, and every decision is a sum of per-word weights. We use this reader in experiment 3.5.

### 2.4 Logic explanations of trained models

- **Logic Explained Networks.** [Barbiero et al. (AAAI 2022)](https://ojs.aaai.org/index.php/AAAI/article/view/20551) extract first-order logic rules over concepts from concept-based networks. [LENp (Jain et al., EMNLP 2022)](https://aclanthology.org/2022.emnlp-main.604) extends them to text classification, with local explanations judged better than LIME in faithfulness and in a user survey. Applied to our rationales, a LEN could give rules such as "positive ← 'superb' ∧ ¬'boring'".
- **Rules from transformer layers.** [NeuroLogic (2025)](https://arxiv.org/abs/2501.08281) extracts global logic rules from hidden predicates of a fine-tuned transformer (DistilBERT on emotion data).

### 2.5 Faithful by construction through symbolic reasoning

- **Proof-based decisions.** [ProoFVer (Krishna, Riedel & Vlachos, TACL 2022)](https://aclanthology.org/2022.tacl-1.59/) decides fact verification only through a natural-logic proof, so the proof is a faithful explanation. Its proofs overlap human rationales better than attention highlights.
- **LLM plus solver.** [Faithful CoT (Lyu et al., 2023)](https://aclanthology.org/2023.ijcnlp-main.20.pdf) and [Logic-LM (Pan et al., 2023)](https://arxiv.org/abs/2305.12295) let an LLM translate a problem into a symbolic program that a deterministic solver executes.
- **The shared principle.** If the final decision is computed only by an inspectable symbolic step from the extracted evidence, the explanation cannot hide a code. That is the motivation for option B in Section 4.

### 2.6 Evaluating explanations with simple readers (closest to our code gap)

- **Trojan explanations.** [Jacovi & Goldberg (TACL 2021)](https://aclanthology.org/2021.tacl-1.18) describe exactly our failure mode in Section 4.1: a select-predict model can encode the predicted class in the highlight (its location, its mask, or a mapping from highlighted tokens to labels). A classifier that sees only the mask predicts the decisions above chance. They propose predict-select-verify instead.
- **Leakage.** [Leakage-Adjusted Simulatability (Hase et al., Findings of EMNLP 2020)](https://arxiv.org/abs/2010.04119) scores natural-language explanations by how much they help an observer model predict the output, while adjusting for explanations that leak the label.
- **Student readers.** [Pruthi et al. (TACL 2022)](https://aclanthology.org/2022.tacl-1.21) score explanations by how much they help a student model simulate a teacher. [The Explanation Game (Treviso & Martins, BlackboxNLP 2020)](https://arxiv.org/abs/2004.13876) evaluates explanations by communication success between an explainer and a layperson.
- **Annotation-free tuning.** [Complement control (Yu et al., EMNLP 2019)](https://aclanthology.org/D19-1420.pdf) adds a player that predicts the label from the unselected words, and tunes without rationale annotations.
- **What our code gap adds.** It is related to all of these but simpler. A *fixed, transparent* reader, trained once on full reviews and never on rationales, reads each rationale's content. Comparing its accuracy with the trained predictor's on the same rationale isolates what the predictor gets from the selection pattern rather than the content. We found no prior work that uses it this way.

### 2.7 Reality checks

- **NeSy's promises in NLP are uneven.** [Hamilton et al.'s structured review](https://arxiv.org/abs/2202.12205) asks whether NeSy NLP papers meet five promised goals (reasoning, out-of-distribution generalization, interpretability, learning from small data, transfer). Systems that compile logic into the network meet the most goals, and definitions, especially of reasoning, are inconsistent across studies.
- **Rules add little to contextual models** for sentiment ([Krishna et al., 2018](https://aclanthology.org/D18-1505/)), which our experiment 3.3 reproduces for lexicon knowledge on frozen BERT states.
- **Recent fixes for degeneration are not NeSy.** Guidance modules ([G-RAT, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29783)), token pruning ([PLMR, KDD 2025](https://arxiv.org/abs/2501.03182)), shortcut discovery ([SSR, ICLR 2024](https://arxiv.org/abs/2403.07955)), and rationale–input alignment ([DAR](https://arxiv.org/abs/2312.04103)) all target the same degeneration we study without symbolic components.
- **Heavy NeSy frameworks fit poorly.** DeepProbLog, Scallop and DomiKnowS ([comparative analysis, NeSy 2025](https://proceedings.mlr.press/v284/sinha25a.html)) are general frameworks for relational reasoning. Binary review sentiment has little relational structure for them to exploit.

---

## 3. Offline experiments on our saved outputs

### Setup

- **Data.** The `probe` stage saved, for every test review, the frozen generator's attention per wordpiece, the word index, whether the word is in the human rationale, and the selection (`attention.npz`, TAIL window). We use the FG+PS models of seeds 1–3 (`results_v3`, `v5`, `v7`) and the fine-tuned runs (`results_v8`–`v14`).
- **Reproduction check.** We re-implemented top-k selection and phrase averaging (`TopkSelector`) and both metrics. This reproduces the paper's F1 and IOU-F1 within 0.03 for all three seeds (e.g. seed 1: F1 31.45 vs 31.43, IOU 30.52 vs 30.50; with phrases F1 34.41 vs 34.38). Selections are identical in 92.5–93.1% of reviews. The rest differ because the released evaluation counts `[CLS]`/`[SEP]` in the budget and can select `[SEP]` (48 reviews in seed 1); our re-implementation selects real tokens only.
- **Lexicons.** The Hu & Liu opinion lexicon as packaged in [NLTK data](https://github.com/nltk/nltk_data) (6,786 entries) and [VADER](https://github.com/cjhutto/vaderSentiment) (words with |valence| ≥ 1: 5,666), plus 14 negation cues.
- **Protocol.** Anything tuned (lexicon choice, prior strength, clause rules) was chosen on the even-numbered test reviews and is reported on the odd-numbered ones. These are exploratory test-set analyses, like the paper's cached-state diagnostics.

### 3.1 How much of a human rationale is sentiment vocabulary?

| Lexicon | Gold wordpieces in lexicon | Other wordpieces in lexicon | Lexicon words that are gold (base rate 9.2%) |
|---|---|---|---|
| Hu & Liu | 17.2% | 6.8% | 20.5% |
| VADER | 13.1% | 6.1% | 17.9% |
| Hu & Liu + VADER | 20.4% | 9.1% | 18.6% |
| + negation cues | 22.7% | 10.8% | 17.6% |

Lexicon words are about twice as likely to be part of a human rationale as other words. Still, four in five rationale wordpieces are not sentiment words: they are function words (46% of rationale words), aspect nouns, and the rest of the phrase.

### 3.2 A purely symbolic rationale

Each review's 20% budget is filled with lexicon words first, then at random.

| Selection | P | R | F1 | IOU-F1 |
|---|---|---|---|---|
| Hu & Liu, single tokens | 13.0 | 28.1 | 17.7 | 12.8 |
| Hu & Liu, five-token phrases | 16.0 | 34.6 | 21.9 | 27.3 |
| Hu & Liu + VADER + negation, single tokens | 13.7 | 29.6 | 18.7 | 13.7 |
| *Reference (paper, TAIL):* random / A2R / NI / FG / FG+PS | | | 12.6 / 19.4 / 24.5 / 31.5 / 34.2 | 7.6 / 19.0 / 27.1 / 30.5 / 50.9 |

Without any training, a lexicon plus phrase expansion matches the fine-tuned A2R generator but stays 12 F1 points below the frozen generator with phrases.

### 3.3 A lexicon prior on the frozen generator (neural × symbolic)

The score becomes attention × (1 + α · lexicon<sub>i</sub>). Choices on the development half: α = 0.5 (larger α was worse), Hu & Liu for single tokens, Hu & Liu + VADER + negation for phrases. Eval half (1,000 reviews), change in token F1 with a 95% paired-bootstrap interval:

| Seed | Single tokens: FG → +prior | Phrases: FG+PS → +prior |
|---|---|---|
| 1 | 31.05 → 30.98 (−0.07; −0.13 to −0.01) | 34.03 → 34.12 (+0.09; −0.02 to +0.20) |
| 2 | 30.84 → 30.75 (−0.09; −0.15 to −0.02) | 33.86 → 33.78 (−0.08; −0.19 to +0.02) |
| 3 | 31.18 → 31.16 (−0.02; −0.08 to +0.04) | 33.83 → 33.97 (+0.13; +0.02 to +0.26) |

The symbolic knowledge is redundant. In seed 1, 22.4% of the frozen generator's selected wordpieces are lexicon words, against 7.8% in the window, so a lexicon prior cannot steer it anywhere new. (A2R's selections contain 7.8%, exactly the base rate: they ignore sentiment vocabulary, as a code would.) IOU-F1 also does not improve.

### 3.4 Clause structure instead of a fixed window

Reviews are split into clauses at punctuation and discourse markers ("but", "however", "although", "though", "yet", "while", "whereas", "and"). Clauses are ranked by mean attention and filled up to the same budget; the variant was chosen on the dev half.

| Seed | PS (w = 5): F1 / IOU | Clauses: F1 / IOU |
|---|---|---|
| 1 | 34.03 / 50.22 | 32.78 / 59.52 |
| 2 | 33.86 / 49.69 | 32.51 / 58.98 |
| 3 | 33.83 / 50.25 | 32.57 / 58.83 |

Clause units behave like a wider window (paper Table 3, w = 9: F1 33.3, IOU 60.4). Longer, human-like spans help IOU-F1 and cost token F1. Linguistic structure is a different point on the same trade-off, not a free gain.

### 3.5 Symbolic readers on the rationales, and the code gap

Two transparent readers read only the words of a rationale:

- a **lexicon reader**: Hu & Liu polarity counts, with a sign flip within three words after a negation cue (no training);
- a **Naive Bayes reader**: log-count ratios over unigrams, trained once on the 40,000 training reviews (last 400 words), never on rationales. Its decision is the sign of a sum of word weights.

Whole TAIL window: 69.7% (lexicon reader) and 82.2% (Naive Bayes reader). Random 20% selection: 62.0% and 77.3%.

| Model | Seed | Token F1 | BERT predictor acc. | NB reader acc. (tokens / phrases) | Lexicon reader acc. | Code gap | One class on random selections |
|---|---|---|---|---|---|---|---|
| A2R | 1 | 17.2 | 89.9 | 72.2 / 70.0 | 65.0 | 17.7 | 99.1% |
| A2R | 2 | 17.7 | 87.1 | 59.4 / 61.2 | 58.1 | 27.7 | 98.6% |
| A2R | 3 | 23.3 | 90.5 | 62.3 / 72.9 | 65.4 | 28.2 | 99.7% |
| NI as described | 1 | 19.5 | 87.9 | 61.6 / 62.0 | 59.1 | 26.3 | 99.2% |
| NI as described | 2 | 22.5 | 85.5 | 68.9 / 70.2 | 62.4 | 16.7 | 99.0% |
| NI as described | 3 | 31.4 | 92.3 | 91.8 / 91.8 | 82.0 | 0.5 | 58.7% |
| NI as released | 1 | 23.3 | 86.0 | 69.0 / 69.5 | 64.5 | 17.0 | 91.6% |
| NI as released | 2 | 32.5 | 92.0 | 90.9 / 90.9 | 85.9 | 1.1 | 66.0% |
| NI as released | 3 | 27.1 | 83.4 | 84.9 / 86.4 | 72.4 | −1.5 | 65.8% |
| FG (frozen) | 1 | 31.4 | 91.3 | 88.1 / 86.8 | 74.5 | 3.2 | 50.9% |
| FG (frozen) | 2 | 31.3 | 91.6 | 88.2 / 86.8 | 75.0 | 3.4 | 62.2% |
| FG (frozen) | 3 | 31.7 | 89.6 | 87.8 / 86.1 | 74.9 | 1.8 | 51.9% |

Code gap = BERT predictor accuracy minus NB reader accuracy on the same single-token rationales. "One class" is the share of reviews the run's predictor assigns to its majority class on random selections, the paper's code diagnostic.

Findings:

1. **Coded rationales are unreadable to a transparent reader.** A2R's rationales and those of NI's code-carrying seeds are read *worse than random selections* (59–72% vs 77.3%), even though their own BERT predictors read them at 86–90%. The selection carries the label for the predictor that co-adapted with it, not as evidence a reader can use: the "Trojan explanations" of Jacovi & Goldberg (2021), now measured.
2. **Good rationales are more readable than the full review.** The frozen generator's 20% selection gives the transparent reader 88.0% against 82.2% for the whole window, because it filters out plot description, which carries misleading sentiment words. NI's best seeds reach 90.9–91.8%.
3. **The code gap separates the two groups perfectly.** It is 16.7–28.2 points for all six runs with a code and −1.5 to 3.4 for all six without one. It correlates with token F1 at r = −0.86 and with the code diagnostic at r = 0.91; the NB reader's accuracy correlates with token F1 at r = 0.91 (lexicon reader: r = 0.90). None of this uses human rationales.
4. **The global weights expose dataset shortcuts.** The reader's largest weights are rating strings ("8/10", "7/10" vs "3/10", "1/10") and names ("uwe", "manos", "ponyo"). This is the usual benefit and cost of a transparent model: shortcuts become visible, and must be filtered before its explanations are shown to people.

**Caveats.** These are 12 runs from four methods, analysed on the test set. The readers use external resources (a lexicon; the training labels). The code gap is a diagnostic, not a training objective: optimizing a generator for a fixed reader invites it to select high-weight words, which experiment 3.3 suggests would not make rationales more human-like.

---

## 4. What this means for the project

### 4.1 Answer

| Goal | Does NeSy help? | Evidence |
|---|---|---|
| More plausible rationales (token F1, IOU) | No measurable gain from lexicon knowledge; structure trades F1 for IOU | 3.2–3.4; Krishna et al. 2018 |
| A transparent decision (why this label) | Yes. A frozen neural selector plus a transparent reader keeps 88% accuracy (2.8 points below BERT on the same single-token rationales; 5.3 below FG+PS with phrases), and its decision is a readable sum of word weights over the rationale | 3.5; Wang & Manning 2012; LEN / RuleNN for rule-based readers |
| Detecting degenerate rationales without human annotations | Yes. The code gap separates coded from content rationales in all 12 runs | 3.5; Jacovi & Goldberg 2021; Hase et al. 2020 |
| Logic-style explanations for users | Possible (LENp on rationale words), untested here | Jain et al. 2022 |

### 4.2 Options, ranked

| Option | What | Cost | Value | Risk |
|---|---|---|---|---|
| **A. Diagnostic in the current paper** | Add the transparent reader and code gap (table in 3.5) as an appendix section, plus one sentence pointing to it. Uses existing outputs only. | Writing only; no runs. The appendix has no page limit, but the main text has no slack. | An independent, annotation-free confirmation that plausibility follows the code; answers "how would we select models without annotations?" | Low; must be presented as exploratory |
| **B. Neural selection, symbolic reading** (follow-up) | Replace or accompany the BERT rationale predictor with a transparent reader trained on rationales (sparse logistic regression, NB, or a LEN over rationale words). Train FG with it in the stop-gradient term. Report accuracy, comprehensiveness and sufficiency of the reader, explanation size, and a small user study. | Code change in `run_words.py`, about 1 Kaggle session per seed | A fully inspectable decision, faithful by construction; directly addresses the "code" problem | The accuracy cost (2.8 points with single tokens, 5.3 with phrases, before training the reader on rationales) may not shrink enough |
| **C. Annotation-free model selection** | Use the code gap to pick seeds, checkpoints or hyperparameters, and check that it picks the high-F1 runs | Offline, cheap | Fixes a stated limitation (no annotated development set) | Small sample (12 runs) so far |
| **D. Rule constraints on selection** | Semantic loss or posterior regularization with negation scope, A-but-B contrast, or "no isolated punctuation" rules on the attention | Moderate | Probably small for F1 (3.3); maybe more human-like spans | Krishna et al. 2018 suggests little gain with BERT |
| **E. LLM concept bottlenecks / symbolic programs** | TBM, CB-LLM, LSP | API cost; a different setting | Rich concept-level explanations | Leaves the unsupervised rationalization setting |

Not recommended here: SenticNet-style knowledge-base pipelines (a different task framing, no rationale extraction), and general NeSy frameworks such as DeepProbLog or Scallop (no relational structure to exploit in binary review sentiment).

### 4.3 If we pursue option B: a minimal plan

1. **Reader.** Add `--rationale_reader {bert,nb,logreg}`. A non-BERT reader trains on the selected words of training batches (bag of words) and supplies ŷ<sup>r</sup> for the stop-gradient JSD term. Since that term carries no gradient, any reader works.
2. **Runs.** FG+PS, TAIL, w = 5, seeds 1–3; compare accuracy and faithfulness with the BERT reader of the same seeds (the paired design is deterministic given the seed).
3. **Explanations.** Report the top word weights per decision, and the global weights after filtering rating strings and names.
4. **Diagnostics.** Report the code gap of every run, and test model selection by code gap (option C).

---

## 5. Reproducing the experiments

All three scripts run on CPU in about a minute and read only `results/` and `usr_movie_review/`.

```bash
mkdir -p /tmp/lex && cd /tmp/lex
curl -sSfLO https://raw.githubusercontent.com/cjhutto/vaderSentiment/master/vaderSentiment/vader_lexicon.txt
curl -sSfLO https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/opinion_lexicon.zip
cd /Users/wmano/github/noise_injection
python3 -B nesy_probes/lexicon_prior.py /tmp/lex/vader_lexicon.txt /tmp/lex/opinion_lexicon.zip     # 3.1-3.3
python3 -B nesy_probes/symbolic_reader_and_clauses.py /tmp/lex/opinion_lexicon.zip                   # reproduction check, lexicon reader, 3.4
python3 -B nesy_probes/transparent_reader.py                                                         # 3.5, code gap
```

They need NumPy only. The lexicons are third-party resources, downloaded at run time and not stored in the repository: VADER is MIT-licensed; check the Hu & Liu lexicon's terms before redistributing it.

---

## References

- Badreddine, d'Avila Garcez, Serafini, Spranger (2022). Logic Tensor Networks. *Artificial Intelligence* 303. [arXiv:2012.13635](https://arxiv.org/abs/2012.13635)
- Barbiero, Ciravegna, Giannini, Lió, Gori, Melacci (2022). Entropy-Based Logic Explanations of Neural Networks. *AAAI*. [link](https://ojs.aaai.org/index.php/AAAI/article/view/20551)
- Cambria, Liu, Decherchi, Xing, Kwok (2022). SenticNet 7: A Commonsense-based Neurosymbolic AI Framework for Explainable Sentiment Analysis. *LREC*. [link](https://aclanthology.org/2022.lrec-1.408)
- d'Avila Garcez, Lamb (2023). Neurosymbolic AI: The 3rd Wave (includes Kautz's taxonomy). [arXiv:2012.05876](https://arxiv.org/abs/2012.05876)
- Guerreiro, Martins (2021). SPECTRA: Sparse Structured Text Rationalization. *EMNLP*. [link](https://aclanthology.org/2021.emnlp-main.525)
- Hamilton, Nayak, Božić, Longo. Is Neuro-Symbolic AI Meeting its Promise in Natural Language Processing? A Structured Review. *Semantic Web journal*. [arXiv:2202.12205](https://arxiv.org/abs/2202.12205)
- Hase, Zhang, Xie, Bansal (2020). Leakage-Adjusted Simulatability. *Findings of EMNLP*. [arXiv:2010.04119](https://arxiv.org/abs/2010.04119)
- Hu, Liu (2004). Mining and Summarizing Customer Reviews. *KDD* (opinion lexicon; used here as packaged in [NLTK data](https://github.com/nltk/nltk_data))
- Hu, Ma, Liu, Hovy, Xing (2016). Harnessing Deep Neural Networks with Logic Rules. *ACL*. [arXiv:1603.06318](https://arxiv.org/abs/1603.06318)
- Hu, Yu, et al. (2024). Learning Robust Rationales for Model Explainability: A Guidance-Based Approach (G-RAT). *AAAI*. [link](https://ojs.aaai.org/index.php/AAAI/article/view/29783)
- Hutto, Gilbert (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. *ICWSM*. [code and lexicon](https://github.com/cjhutto/vaderSentiment)
- Jacovi, Goldberg (2021). Aligning Faithful Interpretations with their Social Attribution. *TACL* 9. [link](https://aclanthology.org/2021.tacl-1.18)
- Jain, Ciravegna, Barbiero, Giannini, Buffelli, Lio (2022). Extending Logic Explained Networks to Text Classification. *EMNLP*. [link](https://aclanthology.org/2022.emnlp-main.604)
- Kocoń et al. (2022). Neuro-symbolic Models for Sentiment Analysis. *ICCS*. [pdf](https://www.iccs-meeting.org/archive/iccs2022/papers/133510628.pdf)
- Krishna, Jyothi, Iyyer (2018). Revisiting the Importance of Encoding Logic Rules in Sentiment Classification. *EMNLP*. [link](https://aclanthology.org/D18-1505/)
- Krishna, Riedel, Vlachos (2022). ProoFVer: Natural Logic Theorem Proving for Fact Verification. *TACL* 10. [link](https://aclanthology.org/2022.tacl-1.59/)
- Li, Srikumar (2019). Augmenting Neural Networks with First-order Logic. *ACL*. [link](https://aclanthology.org/P19-1028/)
- Liu, Lu, Mou (2023). Weakly Supervised Reasoning by Neuro-Symbolic Approaches (incl. Jumper). [arXiv:2309.13072](https://arxiv.org/abs/2309.13072)
- Enhancing the Rationale-Input Alignment for Self-explaining Rationalization (DAR). [arXiv:2312.04103](https://arxiv.org/abs/2312.04103)
- Ludan et al. Interpretable-by-Design Text Understanding with Iteratively Generated Concept Bottleneck (Text Bottleneck Models). [arXiv:2310.19660](https://arxiv.org/abs/2310.19660)
- Lyu et al. (2023). Faithful Chain-of-Thought Reasoning. *IJCNLP-AACL*. [pdf](https://aclanthology.org/2023.ijcnlp-main.20.pdf)
- Pan et al. (2023). Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning. [arXiv:2305.12295](https://arxiv.org/abs/2305.12295)
- Pang, Lee (2004). A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts. *ACL*. [link](https://aclanthology.org/P04-1035/)
- Pruthi et al. (2022). Evaluating Explanations: How Much Do Explanations from the Teacher Aid Students? *TACL* 10. [link](https://aclanthology.org/2022.tacl-1.21)
- Ross, Hughes, Doshi-Velez (2017). Right for the Right Reasons. *IJCAI*. [arXiv:1703.03717](https://arxiv.org/abs/1703.03717)
- Sen et al. (2020). Learning Explainable Linguistic Expressions with Neural Inductive Logic Programming for Sentence Classification (RuleNN). *EMNLP*. [link](https://aclanthology.org/2020.emnlp-main.345)
- Sen, de Carvalho, Riegel, Gray (2022). Neuro-Symbolic Inductive Logic Programming with Logical Neural Networks. *AAAI*. [link](https://ojs.aaai.org/index.php/AAAI/article/view/20795)
- Singh, Askari, Caruana, Gao (2023). Augmenting Interpretable Models with Large Language Models during Training. *Nature Communications* 14. [link](https://www.nature.com/articles/s41467-023-43713-1)
- Sinha et al. (2025). Toward a Clearer Characterization of Neuro-Symbolic Frameworks: A Brief Comparative Analysis. *PMLR* 284. [link](https://proceedings.mlr.press/v284/sinha25a.html)
- Sun, Oikarinen, Ustun, Weng (2025). Concept Bottleneck Large Language Models. *ICLR*. [arXiv:2412.07992](https://arxiv.org/abs/2412.07992)
- Treviso, Martins (2020). The Explanation Game: Towards Prediction Explainability through Sparse Communication. *BlackboxNLP*. [arXiv:2004.13876](https://arxiv.org/abs/2004.13876)
- Wang, Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification. *ACL*. [link](https://aclanthology.org/P12-2018/)
- Wang et al. (2025). Large Language Models are Interpretable Learners (LLM-based Symbolic Programs). *ICLR*. [arXiv:2406.17224](https://arxiv.org/abs/2406.17224)
- Xu, Zhang, Friedman, Liang, Van den Broeck (2018). A Semantic Loss Function for Deep Learning with Symbolic Knowledge. *ICML*. [link](https://proceedings.mlr.press/v80/xu18h.html)
- Yin, Meng, Chang (2020). SentiBERT: A Transferable Transformer-Based Architecture for Compositional Sentiment Semantics. *ACL*. [link](https://aclanthology.org/2020.acl-main.341)
- Yu, Chang, Zhang, Jaakkola (2019). Rethinking Cooperative Rationalization: Introspective Extraction and Complement Control. *EMNLP*. [pdf](https://aclanthology.org/D19-1420.pdf)
- Yuan, Hu, Yu, Wu (2025). Boosting Explainability through Selective Rationalization in Pre-trained Language Models (PLMR). *KDD*. [arXiv:2501.03182](https://arxiv.org/abs/2501.03182)
- Yue et al. (2024). Towards Faithful Explanations: Boosting Rationalization with Shortcuts Discovery (SSR). *ICLR*. [arXiv:2403.07955](https://arxiv.org/abs/2403.07955)
- Zeng, Song (2021). Variational Weakly Supervised Sentiment Analysis with Posterior Regularization. *EACL*. [link](https://aclanthology.org/2021.eacl-main.285)
- Geng, Xing, Zhang, Zhao, Jiang, Si (2025). NeuroLogic: From Neural Representations to Interpretable Logic Rules. [arXiv:2501.08281](https://arxiv.org/abs/2501.08281)
