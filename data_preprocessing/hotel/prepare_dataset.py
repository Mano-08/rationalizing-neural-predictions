"""Hotel Reviews (one aspect) in the format of the USR files, one JSON list per line.

Source: the archive of Bao et al. (EMNLP 2018, "Deriving Machine Attention from Human
Rationales"), people.csail.mit.edu/yujia/files/r2a/data.zip, with TripAdvisor reviews
(Wang et al., KDD 2010) and binary labels for the aspects Location, Service and
Cleanliness. It holds the split that FR, MGR, DR and VER use (VER, TACL 2025, Table 4):
  - training:   data/oracle/hotel_<Aspect>.train
  - validation: data/oracle/hotel_<Aspect>.dev
  - test:       data/target/hotel_<Aspect>.pred_att.gold_att.train, the 200 reviews
                with human rationales (a 0/1 mark per token)
The texts are lowercased and tokenized; tokens are split at spaces.

--split standard keeps that split. In it, almost every test review is also a training
review (199, 200 and 199 of the 200 for Location, Service and Cleanliness), and some are
validation reviews. --split clean (the default) removes every copy of the test reviews
from training and validation, and drops the validation reviews that also occur in
training, so that validation measures unseen reviews. Duplicates within training (the
oversampled class) are kept, as in the standard split. In both splits, test reviews
whose rationale does not have one mark per token (one in Cleanliness) are dropped.

train.jsonl and valid.jsonl hold [tokens, label], test.jsonl [tokens, label, ranges]
with the rationale's word ranges [start, end), as for USR; stats.json reports sizes,
removals and rationale statistics.

    python prepare_dataset.py --zip_path data.zip --aspect Location --data_path OUT [--split clean]
"""
import json
import os
import statistics
import zipfile
from argparse import ArgumentParser

ASPECTS = ["Location", "Service", "Cleanliness"]


def read_rows(archive, name):
    # tab-separated with a header line; the zip member is read in memory, not extracted
    lines = archive.read(name).decode("utf-8").splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:] if line.strip()]


def normalize(text):
    return " ".join(text.split())


def to_ranges(marks):
    ranges, start = [], None
    for i, marked in enumerate(marks + [0]):
        if marked and start is None:
            start = i
        elif not marked and start is not None:
            ranges.append([start, i])
            start = None
    return ranges


def main():
    parser = ArgumentParser()
    parser.add_argument("--zip_path", required = True)
    parser.add_argument("--aspect", required = True, choices = ASPECTS)
    parser.add_argument("--data_path", required = True)
    parser.add_argument("--split", choices = ["clean", "standard"], default = "clean")
    args = parser.parse_args()

    with zipfile.ZipFile(args.zip_path) as archive:
        train = read_rows(archive, f"data/oracle/hotel_{args.aspect}.train")
        valid = read_rows(archive, f"data/oracle/hotel_{args.aspect}.dev")
        annotated = read_rows(archive, f"data/target/hotel_{args.aspect}.pred_att.gold_att.train")

    test, misaligned = [], 0
    for row in annotated:
        # the text column is "raw" in the Location file and "text" in the other two
        tokens, marks = row.get("raw", row.get("text")).split(), [int(mark) for mark in row["rationale"].split()]
        if len(tokens) != len(marks):
            misaligned += 1
            continue
        test.append((tokens, int(row["label"]), marks))
    test_texts = {" ".join(tokens) for tokens, _, _ in test}
    # labels of the standard training reviews, to compare with the test labels
    standard_labels = {}
    for row in train:
        standard_labels.setdefault(normalize(row["text"]), set()).add(int(row["label"]))
    valid_texts_before = {normalize(row["text"]) for row in valid}
    stats = {"aspect": args.aspect, "split": args.split, "test_misaligned_dropped": misaligned,
             "test_reviews_in_standard_training": sum(1 for text in test_texts if text in standard_labels),
             "test_reviews_in_standard_validation": sum(1 for text in test_texts if text in valid_texts_before)}

    if args.split == "clean":
        kept_train = [row for row in train if normalize(row["text"]) not in test_texts]
        train_texts = {normalize(row["text"]) for row in kept_train}
        no_test = [row for row in valid if normalize(row["text"]) not in test_texts]
        kept_valid = [row for row in no_test if normalize(row["text"]) not in train_texts]
        stats["removed_from_training_test_copies"] = len(train) - len(kept_train)
        stats["removed_from_validation_test_copies"] = len(valid) - len(no_test)
        stats["removed_from_validation_also_in_training"] = len(no_test) - len(kept_valid)
        train, valid = kept_train, kept_valid

    os.makedirs(args.data_path, exist_ok = True)
    for target, rows in (("train", train), ("valid", valid)):
        with open(os.path.join(args.data_path, f"{target}.jsonl"), "w") as f:
            for row in rows:
                f.write(json.dumps([row["text"].split(), int(row["label"])]) + "\n")
        stats[target] = {"reviews": len(rows), "positive_share": sum(int(row["label"]) for row in rows) / len(rows),
                         "mean_tokens": statistics.mean(len(row["text"].split()) for row in rows),
                         "max_tokens": max(len(row["text"].split()) for row in rows),
                         "distinct_texts": len({normalize(row["text"]) for row in rows})}
    with open(os.path.join(args.data_path, "test.jsonl"), "w") as f:
        for tokens, label, marks in test:
            f.write(json.dumps([tokens, label, to_ranges(marks)]) + "\n")
    spans = [high - low for _, _, marks in test for low, high in to_ranges(marks)]
    stats["test"] = {"reviews": len(test), "positive_share": sum(label for _, label, _ in test) / len(test),
                     "mean_tokens": statistics.mean(len(tokens) for tokens, _, _ in test),
                     "rationale_rate": sum(sum(marks) for _, _, marks in test) / sum(len(marks) for _, _, marks in test),
                     "spans_per_review": len(spans) / len(test), "mean_span_length": statistics.mean(spans),
                     "reviews_without_rationale": sum(1 for _, _, marks in test if sum(marks) == 0),
                     "label_differs_from_standard_training": sum(1 for tokens, label, _ in test
                                                                 if label not in standard_labels.get(" ".join(tokens), {label}))}
    with open(os.path.join(args.data_path, "stats.json"), "w") as f:
        json.dump(stats, f, indent = 2)
    print(json.dumps(stats, indent = 2))


if __name__ == "__main__":
    main()
