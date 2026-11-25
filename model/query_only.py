"""Query-only classifier for MultiRC and FEVER: the bias expert of the product of experts.

Predicts the label from the query alone (question || answer, or claim). Two BERT
classifiers are cross-fitted on the training set: each is trained on one half
(split by a hash of the query text, so duplicates stay in one half) and predicts the
other half, so that every training query gets a probability from a model that has not
seen it. Validation and test queries get the mean of both models. Each model is
trained with cross-entropy (on logits) for at most --num_epochs epochs and keeps the
epoch with the lowest validation loss (patience --patience).

Outputs in --save_path: query_probs_{train,valid,test}.json (query text -> [p0, p1]),
results.json (accuracy of the held-out predictions), per_example.json (test labels
and probabilities) and config.json.

    python query_only.py --data_path DATA/fever --save_path OUT/fever_query_only
"""
import hashlib
import json
import os
import random
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", required = True)
    parser.add_argument("--save_path", required = True)
    parser.add_argument("--model", default = "bert-base-uncased")
    parser.add_argument("--max_length", type = int, default = 128)
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--lr", type = float, default = 2e-5)
    parser.add_argument("--num_epochs", type = int, default = 3)
    parser.add_argument("--patience", type = int, default = 1)
    parser.add_argument("--seed", type = int, default = 1)
    parser.add_argument("--device", default = "cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--subset", type = int, default = None)
    return parser.parse_args()


def query_key(query):
    # the key run_ce.py looks the probabilities up with
    return " ".join(query)


def fold_of(key):
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 2


def load_split(data_path, split, subset):
    examples = []
    with open(os.path.join(data_path, f"{split}.jsonl")) as f:
        for line in f:
            _, label, _, query = json.loads(line)
            examples.append((query_key(query), int(label)))
            if subset is not None and len(examples) >= subset:
                break
    return examples


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batches(examples, batch_size, shuffle, rng = None):
    order = list(range(len(examples)))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [examples[i] for i in order[start:start + batch_size]]


def encode(tokenizer, batch, max_length, device):
    keys, labels = zip(*batch)
    encoded = tokenizer(list(keys), truncation = True, max_length = max_length, padding = True, return_tensors = "pt").to(device)
    return encoded, torch.tensor(labels, device = device)


@torch.no_grad()
def predict(model, tokenizer, examples, args):
    model.eval()
    probs, loss_sum = [], 0.0
    for batch in batches(examples, 2 * args.batch_size, shuffle = False):
        encoded, labels = encode(tokenizer, batch, args.max_length, args.device)
        logits = model(**encoded).logits.float()
        loss_sum += F.cross_entropy(logits, labels, reduction = "sum").item()
        probs.append(torch.softmax(logits, -1).cpu())
    model.train()
    return torch.cat(probs).numpy(), loss_sum / max(1, len(examples))


def train_fold(fold, train_examples, valid_examples, tokenizer, args):
    set_seed(args.seed + fold)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels = 2).to(args.device)
    optimizer = optim.Adam(model.parameters(), args.lr)
    rng = random.Random(args.seed + fold)
    best_loss, best_state, patience_left, history = float("inf"), None, args.patience, []
    model.train()
    for epoch in range(args.num_epochs):
        train_loss, steps = 0.0, 0
        for batch in batches(train_examples, args.batch_size, shuffle = True, rng = rng):
            encoded, labels = encode(tokenizer, batch, args.max_length, args.device)
            loss = F.cross_entropy(model(**encoded).logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss, steps = train_loss + loss.item(), steps + 1
        _, valid_loss = predict(model, tokenizer, valid_examples, args)
        history.append({"epoch": epoch + 1, "train_loss": train_loss / max(1, steps), "valid_loss": valid_loss})
        print(f"fold {fold} epoch {epoch + 1}: train loss {train_loss / max(1, steps):.4f}, valid loss {valid_loss:.4f}", flush = True)
        if valid_loss < best_loss:
            best_loss, patience_left = valid_loss, args.patience
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left == 0:
                break
    model.load_state_dict(best_state)
    return model, history


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok = True)
    with open(os.path.join(args.save_path, "config.json"), "w") as f:
        json.dump(vars(args), f, indent = 2)
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast = True)
    splits = {split: load_split(args.data_path, split, args.subset) for split in ("train", "valid", "test")}

    train_probs = {}
    eval_probs = {split: [] for split in ("valid", "test")}
    histories = {}
    for fold in (0, 1):
        own = [example for example in splits["train"] if fold_of(example[0]) == fold]
        held_out = [example for example in splits["train"] if fold_of(example[0]) != fold]
        model, histories[fold] = train_fold(fold, own, splits["valid"], tokenizer, args)
        probs, _ = predict(model, tokenizer, held_out, args)
        for (key, _), p in zip(held_out, probs):
            train_probs[key] = p.tolist()
        for split in ("valid", "test"):
            eval_probs[split].append(predict(model, tokenizer, splits[split], args)[0])
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    outputs = {"train": train_probs}
    for split in ("valid", "test"):
        mean = np.mean(eval_probs[split], axis = 0)
        outputs[split] = {key: p.tolist() for (key, _), p in zip(splits[split], mean)}
    for split, probs in outputs.items():
        with open(os.path.join(args.save_path, f"query_probs_{split}.json"), "w") as f:
            json.dump(probs, f)

    def accuracy(split):
        return float(np.mean([int(np.argmax(outputs[split][key])) == label for key, label in splits[split]]))

    labels_test = [label for _, label in splits["test"]]
    results = {
        "accuracy": accuracy("test"),
        "valid_accuracy": accuracy("valid"),
        "train_heldout_accuracy": accuracy("train"),
        "test_majority_class_share": max(np.mean(labels_test), 1 - np.mean(labels_test)),
        "num_examples": {split: len(examples) for split, examples in splits.items()},
        "history": histories,
        "hours": (time.time() - start) / 3600,
    }
    with open(os.path.join(args.save_path, "results.json"), "w") as f:
        json.dump(results, f, indent = 2)
    with open(os.path.join(args.save_path, "per_example.json"), "w") as f:
        json.dump({"label": labels_test, "p1": [outputs["test"][key][1] for key, _ in splits["test"]]}, f)
    print(f"query-only accuracy: test {results['accuracy']:.4f}, valid {results['valid_accuracy']:.4f}, "
          f"train (held out) {results['train_heldout_accuracy']:.4f}; majority class {results['test_majority_class_share']:.4f}", flush = True)


if __name__ == "__main__":
    main()
