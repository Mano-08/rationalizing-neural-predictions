# Unsupervised Selective Rationalization with Noise Injection

This repository contains code for [Unsupervised Selective Rationalization with Noise Injection](https://doi.org/10.48550/arXiv.2305.17534) to appear in ACL 2023. It includes the PyTorch implementation of BERT-A2R (+ NI) model, preprocessing scripts for each dataset used in our evaluation, as well as the USR Movie Review dataset.

## Model

Please note that to train our model, you will need a GPU with 24+ GB of VRAM.
To reproduce our results, please:

1. Set up a conda environment as specified in `spec-file.txt`.
2. Download the original dataset files for MultiRC, FEVER, or ERASER Movie Reviews from [ERASER Benchmark](http://www.eraserbenchmark.com).
3. To preprocess each dataset in question (except for USR Movie Reviews, which are already in the correct format in `usr_movie_review`), navigate to `data_preprocessing/dataset_in_question` and run the first script

    `python prepare_dataset.py --data_path=path_to_the_dataset_in_question --save_path=path_to_the_preprocessed_dataset`
4. To generate the token statistics needed for noise injection for each dataset in question, run the second script

    `python generate_word_statistics.py --data_path=path_to_the_preprocessed_dataset`

4. Train and evaluate our model by running the corresponding script in `model` using the training parameters found in Appendix B of our paper (The default parameters of the script are not always the best for a specific
dataset).

    - `run_ce.py --train --evaluate --dataset=multirc_or_fever --data_path=path_to_the_preprocessed_dataset`  for Claim/Evidence datasets with sentence-level rationales (MultiRC, FEVER)

    - `run_words.py --train --evaluate --data_path=path_to_the_preprocessed_dataset`  for datasets with token-level rationales (USR Movies, ERASER Movies)

## Noise schedules, noise sources and evaluation controls

`run_words.py` (USR Movies, ERASER Movies) accepts the following on top of the original options. All commands are run from `model`.

**Alignment fix.** The released noise injection paired the selection mask (which has no `[CLS]` position) with word indices that include it, so during training with `--inject_noise` the rationale predictor received the token *before* every selected token. This is fixed by default. Pass `--legacy_alignment` to train exactly as released, e.g. to reproduce published numbers. Models trained without `--inject_noise` are unaffected. `models_ce.py` (MultiRC, FEVER) pairs the mask with word indices the same way; it has not been checked or changed.

### Training

| Option | Effect |
| --- | --- |
| `--seed` | Seeds initialization, batch order and noise. |
| `--legacy_alignment` | Trains with the released (shifted) alignment between selection and predictor input, see above. |
| `--noise_schedule=constant` | Fixed noise level `--noise_p` (original NI, the default). |
| `--noise_schedule=exponential\|cosine\|linear` | Open-loop decay from `--noise_p0` to `--noise_p`. `--noise_gamma` is the exponential decay rate **per epoch**. |
| `--noise_schedule=closed_loop` | PI controller on a degeneracy signal, starting at `--noise_p` and kept within `--noise_p_min`/`--noise_p_max`. `--ctrl_signal=jsd` raises noise when the two predictors disagree on the clean rationale (probed every `--ctrl_probe_every` steps), `--ctrl_signal=entropy` when the generator's attention collapses, `--ctrl_signal=exposure` when the rationale holds many of the words noise injection targets. `--ctrl_target`, `--ctrl_kp`, `--ctrl_ki` and `--ctrl_ema` set the target, gains and smoothing. |
| `--noise_target=evidence` | Puts the noise into the evidence the generator's own (attention-based) predictor pools instead of into the rationale: the state pooled at a token is replaced by the state of a random token. `both` does both. Noise on the rationale alone never reaches the generator, which receives no gradient from the rationale predictor. |
| `--noise_source=mlm` | Replaces tokens with in-context substitutes instead of words drawn from the vocabulary. Requires `build_mlm_replacements.py`. |
| `--replacement_probs=saliency` | Chooses tokens to replace by contextual saliency instead of TF*IDF. Requires `build_saliency_probs.py`. |
| `--mask_special_tokens` | Keeps `[SEP]` and `[PAD]` out of the attention and of the rationale. Use it for all models of a comparison or for none. |
| `--truncation_side=left` | Keeps the end of texts longer than `--max_length` instead of the beginning. On USR Movies the end holds more of the human rationales (68% of them survive instead of 54%). |
| `--selection_window=5` | Ranks tokens by their attention averaged over 5 tokens (any odd number), which selects phrases instead of scattered tokens. |
| `--freeze_encoder_bb` | Original option: keeps the generator's encoder at its pretrained weights ("fixed gen. weights" in the paper). |
| `--generator_layers=N` | Uses only the first N layers of the generator's encoder. |
| `--coupling_weight=W` | Lets the loss of the rationale predictor reach the generator (straight-through), which the original model never does. Requires `--inject_noise`; use `--noise_p=0` to couple without noise. Experimental. |
| `--train_subset`, `--valid_subset`, `--test_subset` | Use the first documents only, for a fast debugging loop. |

The noise level realized between validations is logged to `checkpoints/metrics.json` (`noise.p_mean`), together with `noise.exposure`: the mean replacement probability of the selected words at p = 1. It is 1 for a random rationale and falls below 1 only if the generator avoids the words noise injection targets. Validation is always noise-free.

One-time preparation for the options above:

    python run_full_text.py --train --evaluate --data_path=path_to_the_preprocessed_dataset --save_path=trained/full_text
    python build_mlm_replacements.py --data_path=path_to_the_preprocessed_dataset
    python build_saliency_probs.py --data_path=path_to_the_preprocessed_dataset --classifier_path=trained/full_text

`run_full_text.py` trains a classifier on the full input: the accuracy ceiling, the source of saliency, and the independent judge of faithfulness below.

### Evaluation

`results.json` keeps the original scores under their original keys and adds:

| Key | Content |
| --- | --- |
| `rationales_word_level` | Plausibility over words instead of wordpieces. |
| `reference` | Expected scores of a random selection and scores of an oracle selection at the same number of selected tokens: the floor and the ceiling of `rationales.micro`. |
| `plausibility_at_k` | Plausibility of the top-k tokens at every rate in `--eval_sparsities` (default 10%, 20%, 30%). |
| `selection` | Realized selection rate, share of the selection spent on `[SEP]`/`[PAD]`, rationale length, number and length of spans, share of stopwords and punctuation. |
| `truncation` | Share of documents cut by `--max_length` and share of human-annotated words that survive it. Annotated words that are cut off are not part of the recall denominator. |
| `comp_suff_prob` | Comprehensiveness and sufficiency of the rationale predictor computed on probabilities. `comp_suff` applies a sigmoid to probabilities, which confines each term to [0.5, 0.73]. |
| `faithfulness_judge` | With `--faithfulness_model=trained/full_text/checkpoints/full_text.pt`: comprehensiveness and sufficiency according to the independent classifier, their normalized variants (Carton et al., 2020), and AOPC curves over `--aopc_bins`. |

`--selection_method=random` evaluates a checkpoint with tokens selected uniformly at random (random-mask control). `per_example.json` holds per-document counts. `--save_attention` writes the attention of every test token to `attention.npz`.

`--truncation_side` and `--selection_window` can also be given at evaluation only, to score a trained model on the other end of long texts or with phrases.

    python aggregate_results.py --runs ni="trained/ni_seed*" ours="trained/ours_seed*" --baseline=ni

prints mean (std) over runs and a paired bootstrap of the F1 difference against the baseline.

### Tests

    python -m pytest tests

runs offline on CPU with a tiny randomly initialized BERT.

## USR Movie Review Dataset

The dataset can be found in `usr_movie_review`. It contains the training, validation,
and test splits as used in our evaluation. The training and validation documents are represented as (list_of_tokens, label). The test documents are represented as (list_of_tokens, label, list_of_rationale_ranges).

## Citation

If you use our code and/or dataset, please cite:
```
@inproceedings{storek-etal-2023-unsupervised,
    title = "Unsupervised Selective Rationalization with Noise Injection",
    author = "Storek, Adam  and
      Subbiah, Melanie  and
      McKeown, Kathleen",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.707",
    pages = "12647--12659",
    abstract = "A major issue with using deep learning models in sensitive applications is that they provide no explanation for their output. To address this problem, unsupervised selective rationalization produces rationales alongside predictions by chaining two jointly-trained components, a rationale generator and a predictor. Although this architecture guarantees that the prediction relies solely on the rationale, it does not ensure that the rationale contains a plausible explanation for the prediction. We introduce a novel training technique that effectively limits generation of implausible rationales by injecting noise between the generator and the predictor. Furthermore, we propose a new benchmark for evaluating unsupervised selective rationalization models using movie reviews from existing datasets. We achieve sizeable improvements in rationale plausibility and task accuracy over the state-of-the-art across a variety of tasks, including our new benchmark, while maintaining or improving model faithfulness.",
}
```
