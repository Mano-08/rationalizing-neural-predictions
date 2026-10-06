# What the code does, and what changed

## The pipeline, end to end

The task: classify a movie review as positive or negative, and also highlight the words that justify the answer (the "rationale").

Two BERT models are trained together:

```
review ──> GENERATOR (BERT) ──> a score for every token
                │
                ├──> ATTENTION PREDICTOR: classifies from the whole review,
                │    weighted by those scores
                │
                └──> keep the top 20% of tokens = the rationale
                           │
                     NOISE INJECTION: swap some rationale words
                     for other words (training only)
                           │
                     RATIONALE PREDICTOR (second BERT):
                     classifies from the rationale alone
```

The loss is: both predictors should get the label right, and the two predictors should agree with each other.

Why the noise? If the rationale is full of filler words, swapping them for junk hurts nothing, so the generator has no reason to avoid them. Noise makes filler words costly, which pushes the generator toward words that actually carry the sentiment.

At test time there is no noise. We measure accuracy, and how well the highlighted words match the words humans highlighted (precision, recall, F1).

## What it was before

- **How much noise:** one fixed number `p` (for example 0.2) for the whole of training.
- **Which words get swapped:** words with low TF·IDF in the review (common, uninformative words).
- **What they are swapped for:** a random word from the whole vocabulary.

## What changed

### 1. A bug fix (the most important change)

During noisy training, the rationale predictor was being given the word *before* each selected word. If the generator picked "dull" and "superb", the predictor saw "was" and "was". Validation and testing were correct, so training and testing did not match.

This is fixed. `--legacy_alignment` brings the old behaviour back, only for reproducing published numbers.

Two crashes that stopped noise injection from running at all are also fixed.

### 2. How much noise: it can now change during training

| Option | What it does |
| --- | --- |
| constant | Fixed `p`, as before. |
| exponential / cosine / linear | Starts high, decays to a final value. This is the "ANI" idea from your paper. |
| closed loop | Measures the model while it trains and adjusts `p` itself. |

The closed loop watches one of two warning signs and raises the noise when it sees them:

- the two predictors start disagreeing (the generator and predictor are drifting apart), or
- the generator's attention collapses onto a handful of tokens.

This is what makes the method genuinely *adaptive*: a decay schedule only follows the clock, the closed loop reacts to the model.

### 3. What words are swapped in: realistic ones

Random vocabulary words produce word salad, which BERT spots easily and learns to ignore. The new option asks BERT's own masked-language-model head "what word could plausibly go here?" and uses that. Words strongly tied to one label (like "great" or "awful") are excluded, so the substitute is fluent but carries no sentiment.

These substitutes are computed once before training and stored, so training is not slower.

### 4. Which words get swapped: context-aware

TF·IDF does not know context, so it treats "not" as an unimportant word. The new option asks a trained classifier which words mattered for *this* review (gradient × input) and protects those. Also computed once, before training.

### 5. Token selection: no wasted picks

The generator could spend part of its 20% budget on `[SEP]` and `[PAD]` tokens. A new option keeps those out.

### 6. Evaluation: more numbers, same old ones

All original scores are still reported, unchanged. Added on top:

- **Random floor and oracle ceiling.** On this test set a random 20% selection gets F1 ≈ 0.11 and a perfect one gets ≈ 0.52. Every result now has that context.
- **Random-mask control.** Run any trained model with random selection to confirm the floor.
- **How much is lost to truncation.** BERT reads 512 tokens; about 46% of human-highlighted words fall beyond that and can never be found.
- **Is the rationale trivial?** Share of stopwords and punctuation, number and length of spans, the real selection rate.
- **Faithfulness judged by a separate model.** Before, the rationale predictor graded its own rationales. Now an independent classifier trained on full reviews can do it, at several selection sizes (1% to 50%).

### 7. Tools for a proper results table

- `--seed`, so every run can be repeated.
- A plain full-text BERT classifier (the accuracy ceiling, and the independent judge above).
- A script that averages runs across seeds (mean ± std) and tests whether a difference is statistically real.

## What did not change

- The two-BERT architecture and the loss.
- The top-20% selection rule.
- With no new options, training without noise is identical to the original code, and the original scores come out identical on the same model.
- The MultiRC / FEVER code path was not touched.

## One-line summary

Before: fixed amount of random-word noise, on a pipeline with a hidden alignment bug, scored with a few numbers.
Now: the bug is fixed, the noise can adapt to the model and look like real text, and the evaluation says whether a result is actually better than chance.
