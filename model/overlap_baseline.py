"""Untrained sentence selector for MultiRC and FEVER.

Ranks the sentences of each test document by BM25 against the query (the
question with its candidate answer, or the claim) and selects as many
sentences, among the same candidates, as the trained models do: the
TopkSentenceSelector of run_ce.py picks the top 20% of the sentences that have
a token inside the 512-wordpiece joint input. No training and no labels are
used; document frequencies come from the sentences of the evaluated split.
Plausibility is computed as in run_ce.py (sentence-level micro and macro
precision, recall and F1), and per_example.json holds the per-document counts
for the paired bootstrap.

    python overlap_baseline.py --dataset multirc --data_path DATA/multirc --save_path OUT/multirc_bm25
"""
import json
import math
import os
import string
from argparse import ArgumentParser
from collections import Counter

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from datasets_ce import DataLoaderFactory
from metrics import STOPWORDS
from models_ce import TopkSentenceSelector


def terms(words):
    # lower-cased content words: no stopwords, no punctuation-only tokens
    return [word.lower() for word in words if word.lower() not in STOPWORDS and not all(c in string.punctuation for c in word)]


def sentences_of(evidence, sentence_mask):
    sentences = {}
    for word, sentence in zip(evidence, sentence_mask):
        sentences.setdefault(sentence, []).append(word)
    return sentences


class BM25:
    def __init__(self, documents, k1 = 1.2, b = 0.75):
        self.k1, self.b = k1, b
        self.num_documents = len(documents)
        self.average_length = sum(len(document) for document in documents) / max(1, len(documents))
        self.document_frequency = Counter(term for document in documents for term in set(document))

    def idf(self, term):
        df = self.document_frequency.get(term, 0)
        return math.log(1 + (self.num_documents - df + 0.5) / (df + 0.5))

    def score(self, query, document):
        counts, length = Counter(document), len(document)
        score = 0.0
        for term in set(query):
            tf = counts.get(term, 0)
            if tf:
                score += self.idf(term) * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.average_length))
        return score


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", choices = ["multirc", "fever"], required = True)
    parser.add_argument("--data_path", required = True)
    parser.add_argument("--save_path", required = True)
    parser.add_argument("--split", default = "test")
    parser.add_argument("--model", default = "bert-base-uncased")
    parser.add_argument("--max_length", type = int, default = 512)
    parser.add_argument("--sparsity", type = float, default = 0.2)
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--subset", type = int, default = None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast = True)
    loader = lambda: DataLoaderFactory(data_path = args.data_path, noise_p = 0.0, batch_size = args.batch_size, tokenizer = tokenizer,
                                       max_length = args.max_length, shuffle = False, subset = args.subset).create_dataloader(args.split, False)
    selector = TopkSentenceSelector(args.sparsity, args.max_length, tokenizer.pad_token_id, "cpu")

    # document frequencies over every sentence of the split
    corpus = []
    for batch in tqdm(loader(), desc = "collecting sentences"):
        for evidence, sentence_mask in zip(batch.examples.evidences, batch.sentence_mask):
            corpus += [terms(words) for words in sentences_of(evidence, sentence_mask).values()]
    bm25 = BM25(corpus)

    tp = fp = fn = 0
    precision_sum = recall_sum = f1_sum = 0.0
    per_example = {"label": [], "tp": [], "fp": [], "fn": [], "num_selected": [], "num_gold": []}
    for batch in tqdm(loader(), desc = "selecting"):
        input_ids = batch.tokenized_examples.input_ids
        token_att = torch.zeros(input_ids.shape[0], input_ids.shape[1] - 1, 1)
        for i in range(input_ids.shape[0]):
            ids = selector.map_to_tokens_from_batch(batch, i)
            query = terms(batch.examples.queries[i])
            sentences = sentences_of(batch.examples.evidences[i], batch.sentence_mask[i])
            scores = {sentence: bm25.score(query, terms(words)) for sentence, words in sentences.items()}
            values = torch.tensor([scores.get(int(sentence), 0.0) if sentence >= 0 else 0.0 for sentence in ids])
            offset = batch.evidences_start_no_cls(i)
            token_att[i, offset:offset + len(ids), 0] = values
        _, selected = selector.select(batch, token_att)
        for label, chosen, gold in zip(batch.labels, selected, batch.rationale_ranges):
            chosen, gold = set(chosen), set(gold)
            rtp, rfp, rfn = len(chosen & gold), len(chosen - gold), len(gold - chosen)
            tp, fp, fn = tp + rtp, fp + rfp, fn + rfn
            precision_sum += rtp / (rtp + rfp + 1e-6)
            recall_sum += rtp / (rtp + rfn + 1e-6)
            f1_sum += rtp / (rtp + (rfp + rfn) / 2 + 1e-6)
            for key, value in zip(per_example, (int(label), rtp, rfp, rfn, len(chosen), len(gold))):
                per_example[key].append(value)

    total = len(per_example["tp"])
    results = {
        "rationales": {
            "micro": {"prec": tp / (tp + fp), "rec": tp / (tp + fn), "F1": tp / (tp + (fp + fn) / 2)},
            "macro": {"prec": precision_sum / total, "rec": recall_sum / total, "F1": f1_sum / total},
        },
        "selection_method": "bm25",
        "sentence_selector_sparsity": args.sparsity,
        "mean_sentences_selected": sum(per_example["num_selected"]) / total,
        "num_documents": total,
    }
    os.makedirs(args.save_path, exist_ok = True)
    with open(os.path.join(args.save_path, "results.json"), "w") as f:
        json.dump(results, f)
    with open(os.path.join(args.save_path, "per_example.json"), "w") as f:
        json.dump(per_example, f)
    with open(os.path.join(args.save_path, "config.json"), "w") as f:
        json.dump(vars(args), f, indent = 2)
    micro = results["rationales"]["micro"]
    print(f"{args.dataset} {args.split}: {total} documents, {results['mean_sentences_selected']:.2f} sentences selected, "
          f"P {100 * micro['prec']:.2f} R {100 * micro['rec']:.2f} F1 {100 * micro['F1']:.2f}")


if __name__ == "__main__":
    main()
