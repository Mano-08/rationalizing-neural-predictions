import numpy as np
import pytest

from metrics import (count_spans, is_trivial, normalized_faithfulness,
                     paired_bootstrap, plausibility, reference_points,
                     selection_counts, selection_report, to_gold_words)

# CLS, word 0, word 1 (two wordpieces), word 2, word 3, SEP, PAD
WORD_IDS = [None, 0, 1, 1, 2, 3, None, None]


def test_gold_ranges_are_half_open():
    assert to_gold_words([[1, 3], [5, 6]]) == {1, 2, 5}


def test_selection_counts_on_wordpieces_and_words():
    selected = [False, True, False, True, False, False, False, True]
    counts = selection_counts(WORD_IDS, selected, gold_words = {1, 2, 7})
    # wordpieces: word 0 is a false positive, word 1 has one piece selected
    # and one missed, word 2 is missed
    assert (counts["tp"], counts["fp"], counts["fn"]) == (1, 1, 2)
    # words: word 1 counts as selected, as one of its wordpieces is
    assert (counts["tp_word"], counts["fp_word"], counts["fn_word"]) == (1, 1, 1)
    # the selected PAD token is reported separately
    assert counts["special"] == 1
    assert counts["wordpieces"] == 5 and counts["words"] == 4
    # word 7 was cut off by truncation: annotated, but not recoverable
    assert counts["gold_words"] == 3 and counts["gold_words_in_window"] == 2
    assert counts["selected_words"] == [0, 1]


def test_plausibility_micro_and_macro():
    counts = [
        selection_counts(WORD_IDS, [False, True, True, True, False, False, False, False], {0, 1}),
        selection_counts(WORD_IDS, [False, False, False, False, False, True, False, False], {0}),
    ]
    report = plausibility(counts)
    assert report["wordpiece"]["micro"] == {"prec": pytest.approx(3/4), "rec": pytest.approx(3/4), "F1": pytest.approx(3/4)}
    assert report["wordpiece"]["macro"]["prec"] == pytest.approx(0.5)
    assert report["word"]["micro"]["prec"] == pytest.approx(2/3)
    assert report["word"]["micro"]["rec"] == pytest.approx(2/3)


def test_reference_points_bracket_any_selection():
    rng = np.random.default_rng(0)
    counts = []
    for _ in range(200):
        selected = [False] + (rng.random(5) < 0.4).tolist() + [False, False]
        counts.append(selection_counts(WORD_IDS, selected, {1, 2}))
    reference = reference_points(counts)
    observed = plausibility(counts)["wordpiece"]["micro"]
    # 3 of 5 wordpieces are annotated
    assert reference["random"]["prec"] == pytest.approx(0.6)
    assert observed["prec"] == pytest.approx(0.6, abs = 0.08)
    assert reference["oracle"]["F1"] >= observed["F1"]
    assert reference["oracle"]["rec"] >= observed["rec"]


def test_selection_report():
    reviews = [["the", "great", ",", "film", "extra"]]
    counts = [selection_counts(WORD_IDS, [False, True, True, True, False, True, False, True], {1, 4})]
    report = selection_report(counts, reviews)
    assert report["selection"]["realized_rate_wordpiece"] == pytest.approx(4/5)
    assert report["selection"]["realized_rate_word"] == pytest.approx(3/4)
    assert report["selection"]["special_token_share"] == pytest.approx(1/5)
    # selected words 0, 1, 3 form two spans, "the" is a stopword
    assert report["selection"]["mean_spans"] == 2
    assert report["selection"]["mean_span_length"] == pytest.approx(1.5)
    assert report["selection"]["trivial_word_share"] == pytest.approx(1/3)
    assert report["truncation"]["docs_truncated"] == 1.0
    assert report["truncation"]["gold_words_surviving"] == pytest.approx(0.5)


def test_trivial_words_and_spans():
    assert is_trivial("the") and is_trivial("...") and is_trivial("'s")
    assert not is_trivial("great") and not is_trivial("not") and not is_trivial("n't")
    assert count_spans([]) == 0
    assert count_spans([3, 1, 2, 7, 9, 8]) == 2


def test_normalized_faithfulness_bounds():
    # p_full, p_rationale, p_remainder, p_null
    norm_suff, norm_comp, valid = normalized_faithfulness(
        [0.9, 0.9, 0.9, 0.5],
        [0.9, 0.5, 0.7, 0.5],
        [0.5, 0.9, 0.7, 0.5],
        [0.5, 0.5, 0.5, 0.5]
    )
    # a rationale as good as the full input and whose removal is as bad as
    # removing everything is perfectly sufficient and comprehensive
    assert (norm_suff[0], norm_comp[0]) == (pytest.approx(1.0), pytest.approx(1.0))
    # a rationale no better than the empty input scores zero on both
    assert (norm_suff[1], norm_comp[1]) == (pytest.approx(0.0), pytest.approx(0.0))
    assert (norm_suff[2], norm_comp[2]) == (pytest.approx(0.5), pytest.approx(0.5))
    # no gap between full and empty input: nothing to normalize by
    assert valid.tolist() == [True, True, True, False]


def test_normalized_faithfulness_is_clipped():
    # rationale better than the full input, remainder worse than the empty one
    norm_suff, norm_comp, _ = normalized_faithfulness([0.8], [0.95], [0.1], [0.5])
    assert norm_suff[0] == 1.0 and norm_comp[0] == 1.0


def test_paired_bootstrap_detects_a_real_difference_only():
    rng = np.random.default_rng(0)
    num_docs = 300
    gold = rng.integers(5, 40, size = num_docs)
    def run(recall):
        tp = rng.binomial(gold, recall)
        return np.stack([tp, 50 - tp, gold - tp], axis = 1)
    better = paired_bootstrap(run(0.6), run(0.4), num_samples = 2000)
    assert better["diff"] > 0 and better["ci95"][0] > 0 and better["p_value"] < 0.01
    same = paired_bootstrap(run(0.5), run(0.5), num_samples = 2000)
    assert same["ci95"][0] < 0 < same["ci95"][1] and same["p_value"] > 0.05
    with pytest.raises(ValueError):
        paired_bootstrap(run(0.5), run(0.5)[:-1])
