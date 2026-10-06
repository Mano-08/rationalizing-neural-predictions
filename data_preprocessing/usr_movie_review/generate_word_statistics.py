import json
import math
import numpy as np
import pickle
import os
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type = str, default = "original")
    return parser.parse_args()


def get_word_statistics_path(data_path):
    return os.path.join(data_path, "word_statistics")


def normalize(arr):
    arr = arr.max() - arr
    return arr / arr.sum()


def get_tokens(line):
    return json.loads(line)[0]


def get_token_counts(document):
    token_counts = defaultdict(int)
    for token in document:
        token_counts[token] += 1
    return token_counts


@dataclass
class WordStatisticsGenerator:
    data_path: str

    def __post_init__(self):
        print("Extracting documents...")
        self.documents = self.get_documents()
        print("Computing IDF...")
        self.idf = self.get_idf()

    def get_documents(self):
        with open(os.path.join(self.data_path, "train.jsonl"), "r") as f:
            return [get_tokens(line) for line in f.read().splitlines()]

    def get_token_doc_counts(self):
        token_doc_counts = defaultdict(int)
        for document in tqdm(self.documents):
            for token in set(document):
                token_doc_counts[token] += 1
        return token_doc_counts

    def get_idf(self):
        return {token: math.log(len(self.documents) / count) for token, count in self.get_token_doc_counts().items()}

    def get_doc_atf(self):
        doc_atf = defaultdict(int)
        for document in tqdm(self.documents):
            for token in document:
                doc_atf[token] += 1 / len(document)
        return doc_atf

    def get_scored_vocab(self):
        doc_atf = self.get_doc_atf()
        vocab, scores = zip(*[(token, doc_atf[token] * self.idf[token]) for token in doc_atf])
        norm_scores = normalize(np.array(scores))
        return vocab, norm_scores

    def save_scored_vocab(self, scored_vocab):
        with open(os.path.join(self.save_path, "scored_vocab.pkl"), "wb") as f:
            pickle.dump(scored_vocab, f)

    def generate_scored_vocab(self):
        vocab, norm_scores = self.get_scored_vocab()
        self.save_scored_vocab([vocab, norm_scores])

    def get_document_replacement_probs(self, document):
        token_counts = get_token_counts(document)
        replacement_prob = np.array([token_counts[token] / len(document) * self.idf[token] for token in document])
        replacement_prob = normalize(replacement_prob) * replacement_prob.shape[0]
        return replacement_prob

    def save_replacement_probs(self, replacement_probs):
        with open(os.path.join(self.save_path, "train_replacement_probs.pkl"), "wb") as f:
            pickle.dump(replacement_probs, f)

    def tget_replacement_probs(self, documents):
        return [self.get_document_replacement_probs(document) for document in tqdm(documents)]

    def generate_replacement_probs(self):
        replacement_probs = self.tget_replacement_probs(self.documents)
        self.save_replacement_probs(replacement_probs)

    def generate_word_statistics(self):
        self.save_path = get_word_statistics_path(self.data_path)
        os.makedirs(self.save_path, exist_ok = True)
        print("Generating scored vocab...")
        self.generate_scored_vocab()
        print("Generating replacement probabilities...")
        self.generate_replacement_probs()


def main(args):
    WordStatisticsGenerator(args.data_path).generate_word_statistics()


if __name__ == "__main__":
    main(parse_args())