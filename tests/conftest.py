import importlib.util
import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))

POSITIVE = ["good", "great", "wonderful"]
NEGATIVE = ["bad", "awful", "boring"]
FILLER = ["the", "movie", "plot", "actor", "scene", "was", "and", "a", "of", "it", ",", ".", "director", "story", "not", "but"]
MAX_LENGTH = 32


def make_document(rng, label, length):
    tokens = rng.choice(FILLER, size = length).tolist()
    # two short spans of evidence, the second one may be cut off by truncation
    cues = POSITIVE if label else NEGATIVE
    ranges = []
    for start in sorted(rng.choice(np.arange(0, length - 2, 3), size = 2, replace = False).tolist()):
        tokens[start:start + 2] = rng.choice(cues, size = 2).tolist()
        ranges.append([start, start + 2])
    # "unbelievably" is not in the vocabulary and is split into wordpieces
    tokens[rng.integers(length)] = "unbelievably"
    return tokens, ranges


def write_split(path, rng, num_documents, annotated):
    with open(path, "w") as f:
        for i in range(num_documents):
            label = i % 2
            # some documents are longer than MAX_LENGTH, some much shorter
            tokens, ranges = make_document(rng, label, int(rng.integers(8, 60)))
            f.write(json.dumps([tokens, label, ranges] if annotated else [tokens, label]) + "\n")


def load_script(name, *path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, *path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope = "session")
def model_path(tmp_path_factory):
    # tiny randomly initialized BERT with a masked language modeling head
    from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast
    path = tmp_path_factory.mktemp("tiny_bert")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[unused0]"] + POSITIVE + NEGATIVE + FILLER
    vocab += ["un", "##bel", "##iev", "##ably", "film", "very", "really", "quite"]
    with open(path / "vocab.txt", "w") as f:
        f.write("\n".join(vocab))
    BertTokenizerFast(vocab_file = str(path / "vocab.txt"), do_lower_case = True).save_pretrained(path)
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size = len(vocab),
        hidden_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        intermediate_size = 64,
        max_position_embeddings = 64
    )
    BertForMaskedLM(config).save_pretrained(path)
    return str(path)


@pytest.fixture(scope = "session")
def tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, use_fast = True)


@pytest.fixture(scope = "session")
def data_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    write_split(path / "train.jsonl", rng, 48, annotated = False)
    write_split(path / "valid.jsonl", rng, 16, annotated = False)
    write_split(path / "test.jsonl", rng, 24, annotated = True)
    load_script("generate_word_statistics", "data_preprocessing", "usr_movie_review", "generate_word_statistics.py").WordStatisticsGenerator(str(path)).generate_word_statistics()
    return str(path)
