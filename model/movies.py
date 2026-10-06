import json
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


REPLACEMENT_PROBS = {
    "tfidf": "replacement_probs",
    "saliency": "replacement_probs_saliency"
}


class MovieDataset(Dataset):
    def __init__(self, data_path, split, subset = None):
        with open(os.path.join(data_path, f"{split}.jsonl"), "r") as f:
            self.data = [json.loads(line) for line in f.read().splitlines()]
        # Keep the first documents only (fast debugging loop)
        if subset is not None:
            self.data = self.data[:subset]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class MovieDatasetWithReplacementProbs(MovieDataset):
    def __init__(self, data_path, split, replacement_probs = "tfidf", subset = None):
        super().__init__(data_path = data_path, split = split, subset = subset)
        with open(os.path.join(data_path, "word_statistics", f"{split}_{REPLACEMENT_PROBS[replacement_probs]}.pkl"), "rb") as f:
            self.replacement_probs = pickle.load(f)
        # Replacement probabilities are kept unscaled (mean 1 over a document),
        # the noise level p is applied by the rationale extractor on every step
        if len(self.replacement_probs) < len(self.data):
            raise ValueError(f"Replacement probabilities do not cover {split}.jsonl, regenerate the word statistics")
        for document, replacement_probs in zip(self.data, self.replacement_probs):
            if len(document[0]) != len(replacement_probs):
                raise ValueError(f"Replacement probabilities do not match {split}.jsonl, regenerate the word statistics")

    def __getitem__(self, idx):
        return self.data[idx] + [self.replacement_probs[idx], idx]

@dataclass
class MovieDatasetFactory:
    data_path: str
    split: str
    replacement_probs: str = "tfidf"
    subset: Optional[int] = None

    def create_dataset(self, inject_noise):
        if inject_noise:
            return MovieDatasetWithReplacementProbs(self.data_path, self.split, self.replacement_probs, self.subset)
        return MovieDataset(self.data_path, self.split, self.subset)

@dataclass
class ReviewCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    padding: Optional[bool] = True
    truncation: Optional[bool] = True

    def __call__(self, data):
        reviews, labels = zip(*data)
        labels_bb = torch.tensor(labels, dtype=torch.long)
        labels_rp = torch.tensor(labels, dtype=torch.long)
        reviews_tokenized = self.collate_reviews(reviews)
        return ReviewBatch(
            reviews_tokenized = reviews_tokenized,
            labels_bb = labels_bb,
            labels_rp = labels_rp
        )

    def collate_reviews(self, reviews):
        return self.tokenizer(
            text = list(reviews),
            is_split_into_words = True,
            padding = True,
            truncation = True,
            max_length = self.max_length,
            return_tensors = 'pt'
        )

@dataclass
class ReviewCollatorWithReplacementProbs(ReviewCollator):
    def __call__(self, data):
        reviews, labels, replacement_probs, doc_ids = zip(*data)
        labels_bb = torch.tensor(labels, dtype=torch.long)
        labels_rp = torch.tensor(labels, dtype=torch.long)
        reviews_tokenized = self.collate_reviews(reviews)
        return ReviewBatch(
            reviews_tokenized = reviews_tokenized,
            labels_bb = labels_bb,
            labels_rp = labels_rp,
            reviews = reviews,
            replacement_probs = replacement_probs,
            doc_ids = doc_ids
        )

@dataclass
class AnnotatedReviewCollator(ReviewCollator):
    def __call__(self, data):
        reviews, labels, rationale_ranges = zip(*data)
        reviews_tokenized = self.collate_reviews(reviews)
        return AnnotatedReviewBatch(
            reviews = reviews,
            reviews_tokenized = reviews_tokenized,
            labels = labels,
            rationale_ranges = rationale_ranges
        )

@dataclass
class AnnotatedReviewCollatorWithReplacementProbs(ReviewCollator):
    def __call__(self, data):
        reviews, labels, rationale_ranges, replacement_probs, doc_ids = zip(*data)
        reviews_tokenized = self.collate_reviews(reviews)
        return AnnotatedReviewBatch(
            reviews = reviews,
            reviews_tokenized = reviews_tokenized,
            labels = labels,
            rationale_ranges = rationale_ranges,
            replacement_probs = replacement_probs,
            doc_ids = doc_ids
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
                return AnnotatedReviewCollatorWithReplacementProbs(
                    tokenizer = self.tokenizer,
                    max_length = self.max_length,
                    padding = self.padding,
                    truncation = self.truncation
                )
            return AnnotatedReviewCollator(
                tokenizer = self.tokenizer,
                max_length = self.max_length,
                padding = self.padding,
                truncation = self.truncation
            )
        if inject_noise:
            return ReviewCollatorWithReplacementProbs(
                tokenizer = self.tokenizer,
                max_length = self.max_length,
                padding = self.padding,
                truncation = self.truncation
            )
        return ReviewCollator(
            tokenizer = self.tokenizer,
            max_length = self.max_length,
            padding = self.padding,
            truncation = self.truncation
        )

@dataclass
class ReviewBatch:
    reviews_tokenized: torch.Tensor
    labels_bb: torch.Tensor
    labels_rp: torch.Tensor
    reviews: Optional[tuple] = None
    replacement_probs: Optional[np.array] = None
    doc_ids: Optional[tuple] = None

@dataclass
class AnnotatedReviewBatch:
    reviews: tuple
    reviews_tokenized: torch.Tensor
    labels: torch.Tensor
    rationale_ranges: tuple
    replacement_probs: Optional[np.array] = None
    doc_ids: Optional[tuple] = None

@dataclass
class DataLoaderFactory:
    data_path: str
    batch_size: int
    tokenizer: PreTrainedTokenizerBase
    max_length: int
    shuffle: Optional[bool] = True
    replacement_probs: str = "tfidf"
    subset: Optional[int] = None
    seed: Optional[int] = None

    def create_dataloader(self, split, inject_noise):
        dataset = MovieDatasetFactory(self.data_path, split, self.replacement_probs, self.subset).create_dataset(inject_noise)
        collate_fn = CollatorFactory(self.tokenizer, self.max_length).create_collator(split, inject_noise)
        # Seeded shuffling, so that a seed fixes the order of training batches
        generator = None
        if self.shuffle and self.seed is not None:
            generator = torch.Generator().manual_seed(self.seed)

        return DataLoader(
            dataset = dataset,
            batch_size = self.batch_size,
            collate_fn = collate_fn,
            shuffle = self.shuffle,
            generator = generator,
            num_workers = 1
        )