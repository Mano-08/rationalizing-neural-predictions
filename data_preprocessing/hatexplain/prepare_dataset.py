"""HateXplain in the format of the USR files, one JSON list per line.

Follows the split of GenSPP (Ruggeri and Signorelli, ACL 2025), built here from the
original release of HateXplain (Mathew et al., AAAI 2021: Data/dataset.json and
Data/post_id_divisions.json):
  - posts with at most 30 tokens;
  - label: majority over the annotators after merging hate speech and offensive;
    written as 1 = hate speech or offensive, 0 = normal (GenSPP: the reverse);
  - highlight: the tokens marked by more than half of the annotators who gave a
    rationale, or all marks if only one annotator did; normal posts have none.
train.jsonl and valid.jsonl hold [tokens, label] and test.jsonl [tokens, label,
ranges] with the highlighted word ranges [start, end), as for USR. valid_rationales.jsonl
keeps the validation highlights (unused in training). stats.json reports the sizes
and the highlight statistics of the training posts, which set the selection rate.

    python prepare_dataset.py --raw_path RAW --data_path OUT
"""
import json
import os
import statistics
from argparse import ArgumentParser

MAX_TOKENS = 30
# GenSPP's label ids (those of the Hugging Face release): hatespeech 0, normal 1,
# offensive 2, with offensive merged into hate speech
GENSPP_LABEL = {"hatespeech": 0, "normal": 1, "offensive": 0}


def majority_label(annotators):
    # statistics.mode as in GenSPP: the label of most annotators
    return 1 - statistics.mode(GENSPP_LABEL[annotator["label"]] for annotator in annotators)


def majority_highlight(rationales, num_tokens):
    if len(rationales) == 0:
        return [0] * num_tokens
    if len(rationales) == 1:
        return [int(x) for x in rationales[0][:num_tokens]]
    votes = [sum(rationale[i] for rationale in rationales) for i in range(num_tokens)]
    return [int(vote > len(rationales) / 2.0) for vote in votes]


def to_ranges(highlight):
    ranges, start = [], None
    for i, marked in enumerate(highlight + [0]):
        if marked and start is None:
            start = i
        elif not marked and start is not None:
            ranges.append([start, i])
            start = None
    return ranges


def main():
    parser = ArgumentParser()
    parser.add_argument("--raw_path", required = True)
    parser.add_argument("--data_path", required = True)
    args = parser.parse_args()

    with open(os.path.join(args.raw_path, "dataset.json")) as f:
        posts = json.load(f)
    with open(os.path.join(args.raw_path, "post_id_divisions.json")) as f:
        divisions = json.load(f)

    os.makedirs(args.data_path, exist_ok = True)
    stats = {}
    for split, target in (("train", "train"), ("val", "valid"), ("test", "test")):
        rows = []
        for post_id in divisions[split]:
            post = posts[post_id]
            tokens = post["post_tokens"]
            if len(tokens) > MAX_TOKENS:
                continue
            label = majority_label(post["annotators"])
            highlight = majority_highlight(post["rationales"], len(tokens))
            rows.append((tokens, label, highlight))
        with open(os.path.join(args.data_path, f"{target}.jsonl"), "w") as f:
            for tokens, label, highlight in rows:
                line = [tokens, label, to_ranges(highlight)] if target == "test" else [tokens, label]
                f.write(json.dumps(line) + "\n")
        if target == "valid":
            with open(os.path.join(args.data_path, "valid_rationales.jsonl"), "w") as f:
                for tokens, label, highlight in rows:
                    f.write(json.dumps([tokens, label, to_ranges(highlight)]) + "\n")
        num_tokens = sum(len(tokens) for tokens, _, _ in rows)
        marked = sum(sum(highlight) for _, _, highlight in rows)
        spans = [high - low for _, _, highlight in rows for low, high in to_ranges(highlight)]
        toxic = [row for row in rows if row[1] == 1]
        stats[target] = {
            "posts": len(rows),
            "toxic_share": len(toxic) / len(rows),
            "mean_tokens": num_tokens / len(rows),
            "highlight_rate": marked / num_tokens,
            "highlight_rate_toxic_posts": sum(sum(h) for _, _, h in toxic) / sum(len(t) for t, _, _ in toxic),
            "toxic_posts_without_highlight": sum(1 for _, _, h in toxic if sum(h) == 0),
            "normal_posts_with_highlight": sum(1 for _, label, h in rows if label == 0 and sum(h) > 0),
            "mean_span_length": statistics.mean(spans) if spans else 0.0,
            "spans": len(spans),
        }
    with open(os.path.join(args.data_path, "stats.json"), "w") as f:
        json.dump(stats, f, indent = 2)
    print(json.dumps(stats, indent = 2))


if __name__ == "__main__":
    main()
