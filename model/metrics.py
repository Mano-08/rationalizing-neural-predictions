import string

import numpy as np

# Function words used to measure how much of a rationale is trivial.
# Negations are left out on purpose, they carry sentiment.
STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been
before being below between both but by can could did do does doing down during
each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself
now of off on once only or other our ours ourselves out over own s
same she should so some such t than that the their theirs them themselves then
there these they this those through to too under until up very was we were
what when where which while who whom why will with would you your yours
yourself yourselves 's 're 've 'm 'll 'd `` ''
""".split())


def is_trivial(word):
    return word in STOPWORDS or all(char in string.punctuation for char in word)


def safe_div(numerator, denominator):
    return numerator/denominator if denominator > 0 else 0.0


def prf(tp, fp, fn):
    return {
        "prec": safe_div(tp, tp + fp),
        "rec": safe_div(tp, tp + fn),
        "F1": safe_div(tp, tp + (fp + fn)/2)
    }


def to_gold_words(rationale_ranges):
    return {word for low, high in rationale_ranges for word in range(low, high)}


def count_spans(indices):
    indices = sorted(indices)
    return sum(1 for i, index in enumerate(indices) if i == 0 or index != indices[i - 1] + 1)


def selection_counts(word_ids, selected, gold_words):
    """Compares one generated rationale with the human one.

    word_ids: word index of every wordpiece, None for CLS/SEP/PAD
    selected: whether every wordpiece was selected
    gold_words: indices of human-annotated words in the whole document

    Wordpiece-level counts ignore special tokens (reported separately as
    "special"). A word counts as selected if any of its wordpieces is, and
    only words that survived truncation can be recovered.
    """
    tp = fp = fn = special = wordpieces = 0
    selected_words = set()
    window_words = set()
    for word_id, is_selected in zip(word_ids, selected):
        if word_id is None:
            special += int(is_selected)
            continue
        wordpieces += 1
        window_words.add(word_id)
        is_gold = word_id in gold_words
        tp += int(is_selected and is_gold)
        fp += int(is_selected and not is_gold)
        fn += int(not is_selected and is_gold)
        if is_selected:
            selected_words.add(word_id)
    gold_in_window = gold_words & window_words
    tp_word = len(selected_words & gold_in_window)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "tp_word": tp_word,
        "fp_word": len(selected_words) - tp_word,
        "fn_word": len(gold_in_window) - tp_word,
        "special": special,
        "wordpieces": wordpieces,
        "words": len(window_words),
        "gold_words": len(gold_words),
        "gold_words_in_window": len(gold_in_window),
        "selected_words": sorted(selected_words)
    }


def sum_counts(counts, keys):
    return {key: sum(count[key] for count in counts) for key in keys}


def plausibility(counts):
    """Micro- and macro-averaged precision/recall/F1 from per-document counts."""
    report = {}
    for level, suffix in (("wordpiece", ""), ("word", "_word")):
        tp, fp, fn = (np.array([count[key + suffix] for count in counts]) for key in ("tp", "fp", "fn"))
        per_doc = [prf(a, b, c) for a, b, c in zip(tp, fp, fn)]
        report[level] = {
            "micro": prf(tp.sum(), fp.sum(), fn.sum()),
            "macro": {key: float(np.mean([doc[key] for doc in per_doc])) for key in ("prec", "rec", "F1")}
        }
    return report


def reference_points(counts):
    """Plausibility floor and ceiling at the realized number of selected tokens.

    random: expected micro scores of selecting the same number of wordpieces
    uniformly at random. oracle: selecting human-annotated wordpieces first.
    """
    selected = np.array([count["tp"] + count["fp"] for count in counts], dtype = float)
    gold = np.array([count["tp"] + count["fn"] for count in counts], dtype = float)
    wordpieces = np.array([count["wordpieces"] for count in counts], dtype = float)
    random_tp = (selected * gold/np.maximum(wordpieces, 1)).sum()
    oracle_tp = np.minimum(selected, gold).sum()
    return {
        "random": prf(random_tp, selected.sum() - random_tp, gold.sum() - random_tp),
        "oracle": prf(oracle_tp, selected.sum() - oracle_tp, gold.sum() - oracle_tp)
    }


def selection_report(counts, reviews):
    """Degeneracy diagnostics of the generated rationales."""
    totals = sum_counts(counts, ["tp", "fp", "special", "wordpieces", "words", "gold_words", "gold_words_in_window"])
    selected_wordpieces = totals["tp"] + totals["fp"]
    num_selected_words = sum(len(count["selected_words"]) for count in counts)
    num_spans = sum(count_spans(count["selected_words"]) for count in counts)
    num_trivial = sum(is_trivial(review[word]) for count, review in zip(counts, reviews) for word in count["selected_words"])
    return {
        "selection": {
            # share of real wordpieces / words in the (truncated) input that were selected
            "realized_rate_wordpiece": safe_div(selected_wordpieces, totals["wordpieces"]),
            "realized_rate_word": safe_div(num_selected_words, totals["words"]),
            # share of the selection budget spent on SEP/PAD tokens
            "special_token_share": safe_div(totals["special"], selected_wordpieces + totals["special"]),
            "mean_rationale_words": safe_div(num_selected_words, len(counts)),
            "mean_spans": safe_div(num_spans, len(counts)),
            "mean_span_length": safe_div(num_selected_words, num_spans),
            # share of selected words that are stopwords or punctuation
            "trivial_word_share": safe_div(num_trivial, num_selected_words)
        },
        "truncation": {
            "docs_truncated": float(np.mean([len(review) > count["words"] for count, review in zip(counts, reviews)])),
            # human-annotated words cut off by max_length can never be selected
            # and are NOT part of the recall denominator
            "gold_words_surviving": safe_div(totals["gold_words_in_window"], totals["gold_words"]),
            "gold_share_in_window": safe_div(totals["gold_words_in_window"], totals["words"])
        }
    }


def normalized_faithfulness(p_full, p_rationale, p_remainder, p_null, eps = 1e-6):
    """Normalized sufficiency and comprehensiveness (Carton et al., 2020).

    All arguments are probabilities of the predicted class. Both scores are
    in [0, 1] and higher is better. Documents whose null difference
    max(0, p_full - p_null) is zero cannot be normalized and are left out.
    """
    p_full, p_rationale, p_remainder, p_null = (np.asarray(p, dtype = float) for p in (p_full, p_rationale, p_remainder, p_null))
    null_diff = np.maximum(0, p_full - p_null)
    valid = null_diff > eps
    safe_null_diff = np.where(valid, null_diff, 1.0)
    norm_suff = np.clip((null_diff - np.maximum(0, p_full - p_rationale))/safe_null_diff, 0, 1)
    norm_comp = np.clip(np.maximum(0, p_full - p_remainder)/safe_null_diff, 0, 1)
    return norm_suff, norm_comp, valid


def paired_bootstrap(counts_a, counts_b, num_samples = 10000, seed = 0):
    """Paired bootstrap over test documents of the micro F1 difference A - B.

    counts_*: per-document tp/fp/fn arrays of shape (num_docs, 3), summed over
    seeds. Returns the observed difference, its 95% interval and the share of
    resamples in which A does not beat B (one-sided p-value).
    """
    counts_a, counts_b = np.asarray(counts_a, dtype = float), np.asarray(counts_b, dtype = float)
    if counts_a.shape != counts_b.shape:
        raise ValueError("Both arms must be evaluated on the same documents")
    f1 = lambda totals: totals[..., 0]/np.maximum(totals[..., 0] + (totals[..., 1] + totals[..., 2])/2, 1e-12)
    rng = np.random.default_rng(seed)
    num_docs = counts_a.shape[0]
    diffs = np.empty(num_samples)
    for start in range(0, num_samples, 500):
        sample = rng.integers(0, num_docs, size = (min(500, num_samples - start), num_docs))
        diffs[start:start + sample.shape[0]] = f1(counts_a[sample].sum(1)) - f1(counts_b[sample].sum(1))
    return {
        "diff": float(f1(counts_a.sum(0)) - f1(counts_b.sum(0))),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "p_value": float(np.mean(diffs <= 0))
    }
