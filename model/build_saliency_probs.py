import json
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, logging

from models import FullTextClassifier
from movies import REPLACEMENT_PROBS
from run_full_text import get_checkpoint_path
from run_words import model_load

logging.set_verbosity_error()

# Precomputes replacement probabilities from the contextual saliency of every
# token (gradient x input of a full-text classifier) instead of its TF*IDF.
# Tokens the classifier relies on in THIS context (negation, contrast markers)
# are protected even if they are frequent. Run once before training with
# run_words.py --inject_noise --replacement_probs=saliency

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type = str, default = os.path.join("..", "..", "rnp_movie_review", "original"))
    # Directory of a classifier trained by run_full_text.py
    parser.add_argument('--classifier_path', type = str, default = os.path.join("trained", "full_text"))
    parser.add_argument('--model', type = str, default = 'bert-base-uncased')
    parser.add_argument('--max_length', type = int, default = 512)
    parser.add_argument('--truncation_side', choices = ['right', 'left'], default = 'right')
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--device', type = str, default = 'cuda')
    # rank: probability falls linearly with the saliency rank of a token
    # raw:  probability falls linearly with its saliency, as with TF*IDF
    parser.add_argument('--transform', choices = ['rank', 'raw'], default = 'rank')
    parser.add_argument('--subset', type = int, default = None)
    return parser.parse_args()


def to_replacement_probs(saliency, transform):
    # Mirrors generate_word_statistics.py: probabilities average to 1 over a
    # document and the most salient token is never replaced
    if transform == 'rank':
        saliency = saliency.argsort().argsort().astype(float)
    inverse = saliency.max() - saliency
    if inverse.sum() == 0:
        return np.ones_like(inverse, dtype = float)
    return inverse/inverse.sum() * inverse.shape[0]


def get_wordpiece_saliency(model, reviews_tokenized, labels):
    embeddings = model.encoder.get_input_embeddings()(reviews_tokenized.input_ids).detach().requires_grad_(True)
    logits = model(
        inputs_embeds = embeddings,
        token_type_ids = reviews_tokenized.token_type_ids,
        attention_mask = reviews_tokenized.attention_mask
    )
    gradients = torch.autograd.grad(logits.gather(1, labels.unsqueeze(-1)).sum(), embeddings)[0]
    return (gradients * embeddings).sum(-1).abs().detach()


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast = True)
    tokenizer.truncation_side = args.truncation_side
    model = FullTextClassifier(num_labels = 2, model = args.model).to(args.device)
    model_load(model, get_checkpoint_path(args.classifier_path))
    model.eval()

    with open(os.path.join(args.data_path, "train.jsonl"), "r") as f:
        data = [json.loads(line) for line in f.read().splitlines()]
    if args.subset is not None:
        data = data[:args.subset]

    replacement_probs = []
    for start in tqdm(range(0, len(data), args.batch_size)):
        reviews, labels = zip(*data[start:start + args.batch_size])
        reviews_tokenized = tokenizer(
            text = list(reviews),
            is_split_into_words = True,
            padding = True,
            truncation = True,
            max_length = args.max_length,
            return_tensors = 'pt'
        ).to(args.device)
        saliency = get_wordpiece_saliency(model, reviews_tokenized, torch.tensor(labels, device = args.device)).cpu().numpy()
        for i, review in enumerate(reviews):
            # saliency of a word = maximum over its wordpieces
            word_saliency = np.zeros(len(review))
            in_window = np.zeros(len(review), dtype = bool)
            for word_id, value in zip(reviews_tokenized.word_ids(i), saliency[i]):
                if word_id is not None:
                    word_saliency[word_id] = max(word_saliency[word_id], value)
                    in_window[word_id] = True
            # words cut off by max_length are never selected, keep them neutral
            probs = np.ones(len(review))
            probs[in_window] = to_replacement_probs(word_saliency[in_window], args.transform)
            replacement_probs.append(probs)

    save_path = os.path.join(args.data_path, "word_statistics", f"train_{REPLACEMENT_PROBS['saliency']}.pkl")
    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    with open(save_path, "wb") as f:
        pickle.dump(replacement_probs, f)
    print(f"Replacement probabilities saved to ==> {save_path}")


if __name__ == "__main__":
    main(parse_args())
