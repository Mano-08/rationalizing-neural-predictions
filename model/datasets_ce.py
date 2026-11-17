import json
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


class CEDataset(Dataset):
    def __init__(self, data_path, split, subset = None):
        with open(os.path.join(data_path, f"{split}.jsonl"), "r") as f:
            self.data = [json.loads(line) for line in f.read().splitlines()]
        if subset is not None:
            self.data = self.data[:subset]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

 
class CEDatasetWithReplacementProbs(CEDataset):
    def __init__(self, data_path, split, noise_p, subset = None):
        super().__init__(data_path = data_path, split = split, subset = subset)
        with open(os.path.join(data_path, "word_statistics", f"{split}_replacement_probs.pkl"), "rb") as f:
            self.replacement_probs = pickle.load(f)[:len(self.data)]
        # Finish precomputing replacement probabilities
        for i, document_replacement_probs in enumerate(self.replacement_probs):
            for j, sentence_replacement_probs in enumerate(document_replacement_probs):
                sentence_replacement_probs *= noise_p
                sentence_replacement_probs[sentence_replacement_probs > 1] = 1
                self.replacement_probs[i][j] = sentence_replacement_probs

    def __getitem__(self, idx):
        return self.data[idx] + [self.replacement_probs[idx]]


@dataclass
class CEDatasetFactory:
    data_path: str
    split: str
    noise_p: Optional[int] = None
    subset: Optional[int] = None

    def create_dataset(self, inject_noise):
        if inject_noise:
            return CEDatasetWithReplacementProbs(self.data_path, self.split, self.noise_p, self.subset)
        return CEDataset(self.data_path, self.split, self.subset)


def get_evidences_flat(evidences):
    return [[token for sentence in evidence for token in sentence] for evidence in evidences]


def get_replacement_probs_flat(replacement_probs):
    return [np.concatenate(repl_probs) for repl_probs in replacement_probs]


def one_past_sep_token(input_ids, sep_token_id):
    return input_ids.tolist().index(sep_token_id) + 1


def get_evidences_start(tokenized_examples, sep_token_id):
    return [one_past_sep_token(input_ids, sep_token_id) for input_ids in tokenized_examples.input_ids]


def get_examples(queries, flat_evidences, sep_token):
    return [query + [sep_token] + evidence for query, evidence in zip(queries, flat_evidences)]


def get_sentence_mask(reviews):
    return [[i for i, sentence in enumerate(review) for _ in sentence] for review in reviews]


@dataclass
class ExampleCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    padding: Optional[bool] = True
    truncation: Optional[bool] = True

    def __call__(self, data):
        evidences, labels, _, queries = zip(*data)
        evidences_flat = get_evidences_flat(evidences)
        tokenized_examples = self.collate_examples(queries, evidences_flat)
        return ExampleBatch(
            examples = RawExampleBatch(queries, evidences_flat),
            tokenized_examples = tokenized_examples,
            evidences_start = get_evidences_start(tokenized_examples, self.tokenizer.sep_token_id),
            sentence_mask = get_sentence_mask(evidences),
            labels_bb = torch.tensor(labels, dtype=torch.long),
            labels_rp = torch.tensor(labels, dtype=torch.long)
        )

    def collate_examples(self, queries, evidences_flat):
        return self.tokenizer(
            text = get_examples(queries, evidences_flat, self.tokenizer.sep_token),
            is_split_into_words = True,
            padding = True,
            truncation = True,
            max_length = self.max_length,
            return_tensors = 'pt'
        )


@dataclass
class ExampleCollatorWithReplacementProbs(ExampleCollator):
    def __call__(self, data):
        evidences, labels, _, queries, replacement_probs = zip(*data)
        evidences_flat = get_evidences_flat(evidences)
        tokenized_examples = self.collate_examples(queries, evidences_flat)
        return ExampleBatch(
            examples = RawExampleBatch(queries, evidences_flat),
            tokenized_examples = tokenized_examples,
            evidences_start = get_evidences_start(tokenized_examples, self.tokenizer.sep_token_id),
            sentence_mask = get_sentence_mask(evidences),
            labels_bb = torch.tensor(labels, dtype=torch.long),
            labels_rp = torch.tensor(labels, dtype=torch.long),
            replacement_probs = get_replacement_probs_flat(replacement_probs)
        )


@dataclass
class AnnotatedExampleCollator(ExampleCollator):
    def __call__(self, data):
        evidences, labels, rationale_ranges, queries = zip(*data)
        evidences_flat = get_evidences_flat(evidences)
        tokenized_examples = self.collate_examples(queries, evidences_flat)
        return AnnotatedExampleBatch(
            examples = RawExampleBatch(queries, evidences_flat),
            tokenized_examples = tokenized_examples,
            evidences_start = get_evidences_start(tokenized_examples, self.tokenizer.sep_token_id),
            sentence_mask = get_sentence_mask(evidences),
            labels = labels,
            rationale_ranges = rationale_ranges
        )


@dataclass
class AnnotatedExampleCollatorWithReplacementProbs(ExampleCollator):
    def __call__(self, data):
        evidences, labels, rationale_ranges, queries, replacement_probs = zip(*data)
        evidences_flat = get_evidences_flat(evidences)
        tokenized_examples = self.collate_examples(queries, evidences_flat)
        return AnnotatedExampleBatch(
            examples = RawExampleBatch(queries, evidences_flat),
            tokenized_examples = tokenized_examples,
            evidences_start = get_evidences_start(tokenized_examples, self.tokenizer.sep_token_id),
            sentence_mask = get_sentence_mask(evidences),
            labels = labels,
            rationale_ranges = rationale_ranges,
            replacement_probs = get_replacement_probs_flat(replacement_probs)
        )


@dataclass
class CollatorFactory:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    padding: Optional[bool] = True
    truncation: Optional[bool] = True

    def create_collator(self, split, inject_noise):
        if split == "test":
            if inject_noise:
                return AnnotatedExampleCollatorWithReplacementProbs(
                    tokenizer = self.tokenizer,
                    max_length = self.max_length,
                    padding = self.padding,
                    truncation = self.truncation
                )
            return AnnotatedExampleCollator(
                tokenizer = self.tokenizer,
                max_length = self.max_length,
                padding = self.padding,
                truncation = self.truncation
            )
        if inject_noise:
            return ExampleCollatorWithReplacementProbs(
                tokenizer = self.tokenizer,
                max_length = self.max_length,
                padding = self.padding,
                truncation = self.truncation
            )
        return ExampleCollator(
            tokenizer = self.tokenizer,
            max_length = self.max_length,
            padding = self.padding,
            truncation = self.truncation
        )


@dataclass
class RawExampleBatch:
    queries: list
    evidences: list


@dataclass
class BaseExampleBatch:
    examples: RawExampleBatch
    tokenized_examples: torch.Tensor
    evidences_start: list
    sentence_mask: list

    def evidences_start_cls(self, i):
        return self.evidences_start[i]

    def evidences_start_no_cls(self, i):
        return self.evidences_start[i] - 1


@dataclass
class ExampleBatch(BaseExampleBatch):
    labels_bb: torch.Tensor
    labels_rp: torch.Tensor
    replacement_probs: Optional[np.array] = None


@dataclass
class AnnotatedExampleBatch(BaseExampleBatch):
    labels: torch.Tensor
    rationale_ranges: tuple
    replacement_probs: Optional[np.array] = None


@dataclass
class DataLoaderFactory:
    data_path: str
    noise_p: int
    batch_size: int
    tokenizer: PreTrainedTokenizerBase
    max_length: int
    shuffle: Optional[bool] = True
    subset: Optional[int] = None

    def create_dataloader(self, split, inject_noise):
        dataset = CEDatasetFactory(self.data_path, split, self.noise_p, self.subset).create_dataset(inject_noise)
        collate_fn = CollatorFactory(self.tokenizer, self.max_length).create_collator(split, inject_noise)

        return DataLoader(
            dataset = dataset,
            batch_size = self.batch_size,
            collate_fn = collate_fn,
            shuffle = self.shuffle,
            num_workers = 1
        )