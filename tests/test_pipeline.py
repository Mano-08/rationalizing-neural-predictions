import json
import os
import shutil
import sys

import numpy as np
import pytest

import aggregate_results
import build_mlm_replacements
import build_saliency_probs
import run_full_text
import run_words

from conftest import MAX_LENGTH


def run(module, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", [module.__name__] + [str(arg) for arg in argv])
    module.main(module.parse_args())


def common(model_path, data_path, save_path):
    return ["--device", "cpu", "--model", model_path, "--data_path", data_path, "--save_path", save_path,
            "--max_length", MAX_LENGTH, "--batch_size", 8]


def load(save_path, name):
    with open(os.path.join(save_path, name)) as f:
        return json.load(f)


@pytest.fixture(scope = "module")
def work_path(tmp_path_factory, data_path):
    # private copy of the data, as the offline builders write next to it
    path = tmp_path_factory.mktemp("work")
    shutil.copytree(data_path, path / "data")
    return path


@pytest.fixture(scope = "module")
def classifier_path(work_path, model_path):
    save_path = str(work_path / "full_text")
    with pytest.MonkeyPatch.context() as monkeypatch:
        run(run_full_text, monkeypatch, "--train", "--evaluate", "--num_epochs", 2, "--seed", 0,
            *common(model_path, str(work_path / "data"), save_path))
    return save_path


def train_and_evaluate(monkeypatch, work_path, model_path, name, *argv):
    save_path = str(work_path / name)
    run(run_words, monkeypatch, "--train", "--evaluate", "--num_epochs", 2, "--patience", 5, "--seed", 0,
        *common(model_path, str(work_path / "data"), save_path), *argv)
    return save_path, load(save_path, "results.json"), load(os.path.join(save_path, "checkpoints"), "metrics.json")


def test_full_text_classifier(classifier_path):
    assert 0 <= load(classifier_path, "results.json")["accuracy"] <= 1
    assert os.path.exists(run_full_text.get_checkpoint_path(classifier_path))


def test_fixed_noise_and_reported_metrics(monkeypatch, work_path, model_path, classifier_path):
    save_path, results, metrics = train_and_evaluate(
        monkeypatch, work_path, model_path, "ni", "--inject_noise", "--noise_p", 0.2, "--mask_special_tokens",
        "--faithfulness_model", run_full_text.get_checkpoint_path(classifier_path))
    # the realized noise level is logged at every validation
    assert [entry["noise"]["p_mean"] for entry in metrics] == pytest.approx([0.2, 0.2])
    assert all(0.1 < entry["replace_ratio"]["replace_train_ratio"] < 0.3 for entry in metrics)
    # validation is noise-free
    assert all(entry["replace_ratio"]["replace_valid_ratio"] == 0 for entry in metrics)

    # no selection budget is spent on SEP/PAD, the realized rate is the requested one
    assert results["selection"]["special_token_share"] == 0
    assert results["selection"]["realized_rate_wordpiece"] == pytest.approx(0.2, abs = 0.02)
    assert results["truncation"]["docs_truncated"] > 0
    assert 0 < results["truncation"]["gold_words_surviving"] < 1
    # with special tokens masked, both wordpiece-level reports agree
    assert results["rationales"]["micro"] == pytest.approx(results["plausibility_at_k"]["0.2"]["wordpiece"]["micro"])
    assert results["reference"]["random"]["F1"] <= results["reference"]["oracle"]["F1"]
    rates = [results["plausibility_at_k"][k]["realized_rate_wordpiece"] for k in ("0.1", "0.2", "0.3")]
    assert rates == pytest.approx([0.1, 0.2, 0.3], abs = 0.02)
    recalls = [results["plausibility_at_k"][k]["wordpiece"]["micro"]["rec"] for k in ("0.1", "0.2", "0.3")]
    assert recalls[0] <= recalls[1] <= recalls[2]

    # the original score is the same difference after a sigmoid, which
    # can only shrink it
    assert abs(results["comp_suff"]["sufficiency"]) <= abs(results["comp_suff_prob"]["sufficiency"]) + 1e-9
    judge = results["faithfulness_judge"]
    assert set(judge["curve"]) == {"0.01", "0.05", "0.1", "0.2", "0.5"}
    assert judge["curve"]["0.2"]["comprehensiveness"] == pytest.approx(judge["comprehensiveness"], abs = 1e-6)
    assert judge["aopc_sufficiency"] == pytest.approx(np.mean([point["sufficiency"] for point in judge["curve"].values()]))
    assert judge["accuracy"] == pytest.approx(load(classifier_path, "results.json")["accuracy"])

    per_example = load(save_path, "per_example.json")
    micro_f1 = sum(per_example["tp"])/(sum(per_example["tp"]) + (sum(per_example["fp"]) + sum(per_example["fn"]))/2)
    assert micro_f1 == pytest.approx(results["rationales"]["micro"]["F1"])
    assert np.mean(per_example["correct"]) == pytest.approx(results["accuracy"])
    assert load(save_path, "config.json")["noise_p"] == 0.2


@pytest.mark.parametrize("name,argv,check", [
    ("exponential", ["--noise_schedule", "exponential", "--noise_p0", 0.5, "--noise_p", 0.1, "--noise_gamma", 1.0],
        lambda p: 0.1 < p[1] < p[0] < 0.5),
    ("cosine", ["--noise_schedule", "cosine", "--noise_p0", 0.5, "--noise_p", 0.1],
        lambda p: 0.1 < p[1] < 0.3 < p[0] < 0.5),
    ("linear", ["--noise_schedule", "linear", "--noise_p0", 0.5, "--noise_p", 0.1],
        lambda p: p == pytest.approx([0.5 - 0.2 * 5/12, 0.3 - 0.2 * 5/12])),
    ("closed_loop_jsd", ["--noise_schedule", "closed_loop", "--ctrl_signal", "jsd", "--noise_p", 0.2, "--ctrl_probe_every", 2],
        lambda p: all(0.05 - 1e-9 <= value <= 0.5 + 1e-9 for value in p)),
    ("closed_loop_entropy", ["--noise_schedule", "closed_loop", "--ctrl_signal", "entropy", "--noise_p", 0.2],
        lambda p: all(0.05 - 1e-9 <= value <= 0.5 + 1e-9 for value in p)),
])
def test_noise_schedules(monkeypatch, work_path, model_path, name, argv, check):
    _, results, metrics = train_and_evaluate(monkeypatch, work_path, model_path, name, "--inject_noise", *argv)
    assert check([entry["noise"]["p_mean"] for entry in metrics])
    if name.startswith("closed_loop"):
        assert all(0 <= entry["noise"]["signal_ema"] <= 1 for entry in metrics)
    # the amount of replaced words follows the noise level
    for entry in metrics:
        assert entry["replace_ratio"]["replace_train_ratio"] == pytest.approx(entry["noise"]["p_mean"], abs = 0.1)
    assert 0 <= results["accuracy"] <= 1


def test_decaying_schedule_requires_p0(monkeypatch, work_path, model_path):
    with pytest.raises(ValueError):
        train_and_evaluate(monkeypatch, work_path, model_path, "no_p0", "--inject_noise", "--noise_schedule", "cosine")


def test_without_noise_and_random_selection(monkeypatch, work_path, model_path):
    save_path, results, metrics = train_and_evaluate(monkeypatch, work_path, model_path, "plain")
    assert all(entry["noise"]["p_mean"] == 0 for entry in metrics)
    assert all(entry["replace_ratio"]["replace_train_ratio"] == 0 for entry in metrics)
    # random-mask control on the same checkpoint: plausibility at the floor
    run(run_words, monkeypatch, "--evaluate", "--selection_method", "random", "--mask_special_tokens", "--seed", 0,
        *common(model_path, str(work_path / "data"), save_path))
    control = load(save_path, "results.json")
    assert control["rationales"]["micro"]["prec"] == pytest.approx(control["reference"]["random"]["prec"], abs = 0.1)
    assert control["selection"]["realized_rate_wordpiece"] == pytest.approx(0.2, abs = 0.02)


def test_mlm_noise(monkeypatch, work_path, model_path):
    data_path = str(work_path / "data")
    run(build_mlm_replacements, monkeypatch, "--device", "cpu", "--model", model_path, "--data_path", data_path,
        "--max_length", MAX_LENGTH, "--batch_size", 8, "--num_candidates", 4, "--top_k", 8, "--min_prob", 0.0,
        "--min_z", 1.0)
    candidates = np.load(os.path.join(data_path, "word_statistics", "train_mlm_candidates.npy"))
    offsets = np.load(os.path.join(data_path, "word_statistics", "train_mlm_offsets.npy"))
    with open(os.path.join(data_path, "train.jsonl")) as f:
        data = [json.loads(line) for line in f.read().splitlines()]
    assert offsets[-1] == candidates.shape[0] == sum(len(tokens) for tokens, _ in data)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokens = tokenizer.convert_ids_to_tokens(np.arange(len(tokenizer)))
    informative = build_mlm_replacements.get_label_informative_tokens(data, 0.4, 1.0)
    # the cue words of the synthetic task are recognized as label-informative
    assert {"good", "great", "wonderful", "bad", "awful", "boring"} <= informative
    assert "the" not in informative
    for (review, _), offset in zip(data, offsets):
        encoded = tokenizer(review, is_split_into_words = True, truncation = True, max_length = MAX_LENGTH)
        in_window = {word_id for word_id in encoded.word_ids() if word_id is not None}
        for word, word_candidates in enumerate(candidates[offset:offset + len(review)]):
            proposed = [tokens[token_id] for token_id in word_candidates if token_id != 0]
            if word not in in_window:
                # cut off by truncation: can never be selected, no substitutes
                assert proposed == []
                continue
            assert len(proposed) == len(set(proposed)) == 4
            assert review[word] not in proposed
            assert not any(token.startswith("##") or token.startswith("[") or token in informative for token in proposed)

    _, results, metrics = train_and_evaluate(monkeypatch, work_path, model_path, "mlm", "--inject_noise", "--noise_p", 0.3, "--noise_source", "mlm")
    assert all(0.15 < entry["replace_ratio"]["replace_train_ratio"] < 0.45 for entry in metrics)


def test_saliency_replacement_probs(monkeypatch, work_path, model_path, classifier_path):
    data_path = str(work_path / "data")
    run(build_saliency_probs, monkeypatch, "--device", "cpu", "--model", model_path, "--data_path", data_path,
        "--classifier_path", classifier_path, "--max_length", MAX_LENGTH, "--batch_size", 8)
    import pickle
    with open(os.path.join(data_path, "word_statistics", "train_replacement_probs_saliency.pkl"), "rb") as f:
        saliency_probs = pickle.load(f)
    with open(os.path.join(data_path, "word_statistics", "train_replacement_probs.pkl"), "rb") as f:
        tfidf_probs = pickle.load(f)
    assert [len(probs) for probs in saliency_probs] == [len(probs) for probs in tfidf_probs]
    for probs in saliency_probs:
        # same scale as TF*IDF probabilities: mean 1 over the document, so
        # that p means the same share of replaced words for both
        assert probs.mean() == pytest.approx(1.0)
        assert probs.min() == pytest.approx(0.0) and probs.max() <= 2.0 + 1e-9
    _, results, metrics = train_and_evaluate(monkeypatch, work_path, model_path, "saliency", "--inject_noise", "--noise_p", 0.2, "--replacement_probs", "saliency")
    assert all(0.1 < entry["replace_ratio"]["replace_train_ratio"] < 0.3 for entry in metrics)


def test_aggregation(monkeypatch, work_path, model_path, capsys):
    for name, argv in (("agg_ni_seed", ["--inject_noise", "--noise_p", 0.2]), ("agg_plain_seed", [])):
        for seed in (1, 2):
            run(run_words, monkeypatch, "--train", "--evaluate", "--num_epochs", 1, "--seed", seed,
                *common(model_path, str(work_path / "data"), str(work_path / f"{name}{seed}")), *argv)
    capsys.readouterr()
    run(aggregate_results, monkeypatch, "--runs", f"ni={work_path}/agg_ni_seed*", f"plain={work_path}/agg_plain_seed*",
        "--baseline", "plain", "--num_samples", 200)
    output = capsys.readouterr().out
    rows = [line for line in output.splitlines() if line.startswith("| ni") or line.startswith("| plain")]
    assert len(rows) == 2 and all("| 2 |" in row for row in rows)
    assert "ni (wordpiece):" in output and "ni (word):" in output
    with pytest.raises(ValueError):
        run(aggregate_results, monkeypatch, "--runs", f"missing={work_path}/nothing*")
