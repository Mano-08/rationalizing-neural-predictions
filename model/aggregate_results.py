import glob
import json
import os
from argparse import ArgumentParser

import numpy as np

from metrics import paired_bootstrap

# Aggregates results.json of several runs (seeds) per model into mean (std)
# and tests every model against a baseline with a paired bootstrap, e.g.
#   python aggregate_results.py --runs ni="trained/ni_seed*" ani="trained/ani_seed*" --baseline ni

COLUMNS = {
    "Acc": lambda r: 100 * r["accuracy"],
    "P": lambda r: 100 * r["rationales"]["micro"]["prec"],
    "R": lambda r: 100 * r["rationales"]["micro"]["rec"],
    "F1": lambda r: 100 * r["rationales"]["micro"]["F1"],
    "IOU": lambda r: 100 * r["micro_iou"]["0.1"]["f1"],
    "Word F1": lambda r: 100 * r["rationales_word_level"]["micro"]["F1"],
    "Rate": lambda r: 100 * r["selection"]["realized_rate_wordpiece"],
    "Trivial": lambda r: 100 * r["selection"]["trivial_word_share"],
    "Com": lambda r: r["comp_suff"]["comprehensiveness"],
    "Suf": lambda r: r["comp_suff"]["sufficiency"],
    "Com*": lambda r: r["faithfulness_judge"]["comprehensiveness"],
    "Suf*": lambda r: r["faithfulness_judge"]["sufficiency"],
    "NormCom*": lambda r: r["faithfulness_judge"]["normalized_comprehensiveness"],
    "NormSuf*": lambda r: r["faithfulness_judge"]["normalized_sufficiency"],
    "AOPC Com*": lambda r: r["faithfulness_judge"]["aopc_comprehensiveness"],
    "AOPC Suf*": lambda r: r["faithfulness_judge"]["aopc_sufficiency"],
}


def parse_args():
    parser = ArgumentParser()
    # name=glob pairs, every match of a glob is one run of that model
    parser.add_argument("--runs", type = str, nargs = "+", required = True)
    parser.add_argument("--baseline", type = str, default = None)
    parser.add_argument("--num_samples", type = int, default = 10000)
    parser.add_argument("--seed", type = int, default = 0)
    return parser.parse_args()


def load_runs(pattern):
    runs = []
    for path in sorted(glob.glob(pattern)):
        if not os.path.exists(os.path.join(path, "results.json")):
            continue
        with open(os.path.join(path, "results.json"), "r") as f:
            results = json.load(f)
        with open(os.path.join(path, "per_example.json"), "r") as f:
            per_example = json.load(f)
        runs.append((path, results, per_example))
    return runs


def get_value(column, results):
    try:
        return COLUMNS[column](results)
    except (KeyError, TypeError):
        # metric not computed for this run (e.g. no --faithfulness_model)
        return None


def format_cell(values):
    values = [value for value in values if value is not None]
    if len(values) == 0:
        return "-"
    std = np.std(values, ddof = 1) if len(values) > 1 else 0.0
    digits = 1 if abs(np.mean(values)) >= 1 else 3
    return f"{np.mean(values):.{digits}f} ({std:.{digits}f})"


def sum_counts(runs, suffix = ""):
    # per-document tp/fp/fn summed over the runs of a model
    return sum(np.stack([per_example[key + suffix] for key in ("tp", "fp", "fn")], axis = 1) for _, _, per_example in runs)


def main(args):
    models = {}
    for run in args.runs:
        name, pattern = run.split("=", 1)
        models[name] = load_runs(pattern)
        if len(models[name]) == 0:
            raise ValueError(f"No finished runs match {pattern}")

    print("| Model | Runs | " + " | ".join(COLUMNS) + " |")
    print("|" + " --- |" * (len(COLUMNS) + 2))
    for name, runs in models.items():
        cells = [format_cell([get_value(column, results) for _, results, _ in runs]) for column in COLUMNS]
        print(f"| {name} | {len(runs)} | " + " | ".join(cells) + " |")
    print()
    print("Mean (std over runs). P/R/F1/IOU: wordpiece-level, micro-averaged. Rate: realized selection rate.")
    print("Trivial: share of selected words that are stopwords or punctuation.")
    print("Com/Suf: rationale predictor, as in Storek et al. (2023). *: independent full-text classifier.")

    if args.baseline is not None:
        print()
        print(f"Paired bootstrap over test documents against {args.baseline} (micro F1 difference, 95% interval, one-sided p):")
        for name, runs in models.items():
            if name == args.baseline:
                continue
            for level, suffix in (("wordpiece", ""), ("word", "_word")):
                # counts are averaged over runs, so models may differ in their number of runs
                test = paired_bootstrap(
                    sum_counts(runs, suffix)/len(runs),
                    sum_counts(models[args.baseline], suffix)/len(models[args.baseline]),
                    num_samples = args.num_samples,
                    seed = args.seed
                )
                print(f"{name} ({level}): {100 * test['diff']:+.2f} [{100 * test['ci95'][0]:+.2f}, {100 * test['ci95'][1]:+.2f}] p = {test['p_value']:.4f}")
        print("The bootstrap covers the choice of test documents only, not the variance across seeds.")


if __name__ == "__main__":
    main(parse_args())
