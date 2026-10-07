import json
import math
import os
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging

logging.set_verbosity_error()

# Precomputes, for every word of every training document, a pool of in-context
# substitutes sampled from a masked language model. Substitutes that are
# associated with a label in the training data are filtered out, so that the
# noise is fluent but carries no evidence. Run once before training with
# run_words.py --inject_noise --noise_source=mlm
#
# Every document is encoded num_folds times, each time with a different
# 1/num_folds of its words masked, so the cost is num_folds forward passes per
# document instead of one per word.

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type = str, default = os.path.join("..", "..", "rnp_movie_review", "original"))
    parser.add_argument('--model', type = str, default = 'bert-base-uncased')
    parser.add_argument('--max_length', type = int, default = 512)
    parser.add_argument('--truncation_side', choices = ['right', 'left'], default = 'right')
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--device', type = str, default = 'cuda')
    parser.add_argument('--seed', type = int, default = 0)
    # Share of words masked at once = 1/num_folds
    parser.add_argument('--num_folds', type = int, default = 7)
    # Size of the pool kept per word
    parser.add_argument('--num_candidates', type = int, default = 10)
    # Sampling: only the top_k most likely tokens with probability of at
    # least min_prob are eligible
    parser.add_argument('--top_k', type = int, default = 50)
    parser.add_argument('--min_prob', type = float, default = 1e-3)
    parser.add_argument('--temperature', type = float, default = 1.0)
    # Label filter: a token is label-informative if the log-odds of it
    # appearing in a positive vs. a negative document exceed max_log_odds in
    # absolute value with a z-score of at least min_z
    parser.add_argument('--max_log_odds', type = float, default = 0.4)
    parser.add_argument('--min_z', type = float, default = 2.0)
    parser.add_argument('--subset', type = int, default = None)
    return parser.parse_args()


def get_label_informative_tokens(data, max_log_odds, min_z, prior = 0.5):
    # document frequency of every token per label
    doc_counts = defaultdict(lambda: [0, 0])
    num_docs = [0, 0]
    for tokens, label in data:
        num_docs[label] += 1
        for token in set(tokens):
            doc_counts[token][label] += 1
    informative = set()
    for token, (negative, positive) in doc_counts.items():
        # smoothed log-odds ratio and its standard error
        cells = [positive + prior, num_docs[1] - positive + prior, negative + prior, num_docs[0] - negative + prior]
        log_odds = math.log(cells[0]/cells[1]) - math.log(cells[2]/cells[3])
        z = log_odds/math.sqrt(sum(1/cell for cell in cells))
        if abs(log_odds) > max_log_odds and abs(z) >= min_z:
            informative.add(token)
    return informative


def get_banned_ids(tokenizer, informative):
    # special tokens, wordpiece continuations and label-informative tokens
    # are never used as substitutes
    banned = torch.zeros(len(tokenizer), dtype = torch.bool)
    special = set(tokenizer.all_special_tokens)
    for token, token_id in tokenizer.get_vocab().items():
        if token in special or token.startswith("##") or token.startswith("[unused") or token in informative:
            banned[token_id] = True
    return banned


def get_mlm_logits(model, reviews_tokenized, targets):
    # only run the (large) output layer on positions we need
    if hasattr(model, "cls") and hasattr(model, "base_model"):
        hidden_states = model.base_model(**reviews_tokenized)[0]
        return model.cls(hidden_states[targets])
    return model(**reviews_tokenized).logits[targets]


def sample_candidates(logits, original_ids, banned, args, generator):
    logits = logits.float()/args.temperature
    logits[:, banned] = float("-inf")
    # never propose the token that is already there
    logits.scatter_(1, original_ids.unsqueeze(-1), float("-inf"))
    probs = torch.softmax(logits, -1)
    top_probs, top_ids = probs.topk(k = args.top_k, dim = -1)
    eligible = top_probs >= args.min_prob
    # Gumbel top-k = sampling without replacement proportionally to probability
    gumbel = -torch.log(-torch.log(torch.rand(top_probs.shape, generator = generator).clamp_min(1e-20))).to(top_probs.device)
    scores = (top_probs.clamp_min(1e-20).log() + gumbel).masked_fill(~eligible, float("-inf"))
    picked_scores, picked = scores.topk(k = args.num_candidates, dim = -1)
    candidates = top_ids.gather(1, picked)
    # 0 (= PAD) marks an empty slot
    return candidates.masked_fill(picked_scores == float("-inf"), 0)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast = True)
    tokenizer.truncation_side = args.truncation_side
    if len(tokenizer) >= 2**16 or tokenizer.pad_token_id != 0:
        raise ValueError("Candidates are stored as uint16 with 0 = PAD as the empty slot")
    if args.top_k < args.num_candidates:
        raise ValueError("--top_k must be at least --num_candidates")
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(args.device)
    model.eval()
    generator = torch.Generator().manual_seed(args.seed)

    with open(os.path.join(args.data_path, "train.jsonl"), "r") as f:
        data = [json.loads(line) for line in f.read().splitlines()]
    # label statistics always come from the whole training set
    informative = get_label_informative_tokens(data, args.max_log_odds, args.min_z)
    print(f"Label-informative tokens excluded from substitutes: {len(informative)}")
    banned = get_banned_ids(tokenizer, informative).to(args.device)
    if args.subset is not None:
        data = data[:args.subset]

    save_path = os.path.join(args.data_path, "word_statistics")
    os.makedirs(save_path, exist_ok = True)
    offsets = np.concatenate([[0], np.cumsum([len(tokens) for tokens, _ in data])]).astype(np.int64)
    np.save(os.path.join(save_path, "train_mlm_offsets.npy"), offsets)
    candidates = np.lib.format.open_memmap(
        os.path.join(save_path, "train_mlm_candidates.npy"),
        mode = "w+",
        dtype = np.uint16,
        shape = (int(offsets[-1]), args.num_candidates)
    )
    candidates[:] = 0

    with torch.no_grad():
        for start in tqdm(range(0, len(data), args.batch_size)):
            reviews = [tokens for tokens, _ in data[start:start + args.batch_size]]
            reviews_tokenized = tokenizer(
                text = reviews,
                is_split_into_words = True,
                padding = True,
                truncation = True,
                max_length = args.max_length,
                return_tensors = 'pt'
            )
            # word index of every wordpiece, -1 for CLS/SEP/PAD
            word_ids = torch.tensor([[-1 if word_id is None else word_id for word_id in reviews_tokenized.word_ids(i)] for i in range(len(reviews))])
            is_word = word_ids >= 0
            is_first = is_word.clone()
            is_first[:, 1:] &= word_ids[:, 1:] != word_ids[:, :-1]
            # row of every wordpiece's word in the candidate table
            rows = word_ids + torch.from_numpy(offsets[start:start + len(reviews)]).unsqueeze(-1)
            reviews_tokenized = reviews_tokenized.to(args.device)
            input_ids = reviews_tokenized.input_ids.clone()
            for fold in range(args.num_folds):
                # whole-word masking of every num_folds-th word
                masked = is_word & (word_ids % args.num_folds == fold)
                targets = masked & is_first
                if not targets.any():
                    continue
                reviews_tokenized["input_ids"] = input_ids.masked_fill(masked.to(args.device), tokenizer.mask_token_id)
                logits = get_mlm_logits(model, reviews_tokenized, targets.to(args.device))
                picked = sample_candidates(logits, input_ids[targets.to(args.device)], banned, args, generator)
                candidates[rows[targets].numpy()] = picked.cpu().numpy().astype(np.uint16)

    candidates.flush()
    covered = (np.asarray(candidates[:, 0]) != 0).mean()
    print(f"Words with at least one substitute: {covered:.4f}")
    print(f"Candidates saved to ==> {save_path}")


if __name__ == "__main__":
    main(parse_args())
