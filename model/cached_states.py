"""Diagnostic experiments on cached BERT states (paper: Section 4.3 and App. E).

The scripts that produced the first version of these numbers were lost, so this
file re-implements the experiments from the paper's description. BERT encodes a
fixed sample of 2,000 training and 1,000 annotated test reviews once; the
hidden states are kept on disk. The generator's two linear heads (the scorer
and the classifier of the attention-based predictor) are then trained on these
states, without a rationale-based predictor. Heads, loss and metrics follow the
full pipeline (models.py, metrics.py, run_words.py). Attention and selection
cover the real tokens of a review only ([CLS], [SEP] and padding excluded).

    python cached_states.py cache      --data_path DIR --cache_dir DIR
    python cached_states.py run        --data_path DIR --cache_dir DIR --out FILE
    python cached_states.py classifier --data_path DIR --out FILE

DIR of --data_path holds train.jsonl, test.jsonl and word_statistics/ (made by
data_preprocessing/usr_movie_review/generate_word_statistics.py).
"""
import json
import os
import pickle
import time
from argparse import ArgumentParser
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from metrics import is_trivial, plausibility, selection_counts, selection_report, to_gold_words
from noise import PIController

ENCODERS = {"pretrained": "bert-base-uncased", "imdb": "textattack/bert-base-uncased-imdb"}
# what the experiments below need: (encoder, window, layers)
CACHE_CONFIGS = [("pretrained", "head", (0, 4, 8, 12)), ("pretrained", "tail", (12,)), ("imdb", "head", (0, 12))]
NUM_TRAIN, NUM_TEST, SAMPLE_SEED = 2000, 1000, 0
SEEDS = (1, 2, 3)
SPARSITY = 0.2


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_split(data_path, split):
    with open(os.path.join(data_path, f"{split}.jsonl")) as f:
        return [json.loads(line) for line in f]


def sample_indices(num_docs, num_sample):
    rng = np.random.default_rng(SAMPLE_SEED)
    return sorted(rng.choice(num_docs, num_sample, replace = False).tolist())


def get_tokenizer(window):
    tokenizer = AutoTokenizer.from_pretrained(ENCODERS["pretrained"])
    # HEAD keeps the beginning of a long review, TAIL its end
    tokenizer.truncation_side = "right" if window == "head" else "left"
    return tokenizer


# ---------------------------------------------------------------------------
# cache: encode the sampled reviews once
# ---------------------------------------------------------------------------

def cache(args):
    os.makedirs(args.cache_dir, exist_ok = True)
    device = get_device()
    with open(os.path.join(args.data_path, "word_statistics", "train_replacement_probs.pkl"), "rb") as f:
        train_probs = pickle.load(f)
    splits = {}
    for split, num in (("train", NUM_TRAIN), ("test", NUM_TEST)):
        data = load_split(args.data_path, split)
        indices = sample_indices(len(data), num)
        splits[split] = (indices, [data[i] for i in indices])
    for encoder, window, layers in CACHE_CONFIGS:
        tokenizer = get_tokenizer(window)
        model = AutoModel.from_pretrained(ENCODERS[encoder]).to(device).eval()
        for split, (indices, docs) in splits.items():
            paths = {layer: os.path.join(args.cache_dir, f"{encoder}_{window}_L{layer}_{split}.npy") for layer in layers}
            if all(os.path.exists(path) for path in paths.values()):
                continue
            start = time.time()
            states = {layer: [] for layer in layers}
            meta = {"doc_ids": indices, "word_ids": [], "lengths": [], "labels": [doc[1] for doc in docs]}
            for b in range(0, len(docs), args.batch_size):
                batch = docs[b:b + args.batch_size]
                encoded = tokenizer([doc[0] for doc in batch], is_split_into_words = True, truncation = True,
                                    max_length = 512, padding = True, return_tensors = "pt")
                with torch.no_grad():
                    outputs = model(**{key: value.to(device) for key, value in encoded.items()}, output_hidden_states = True)
                for i in range(len(batch)):
                    word_ids = encoded.word_ids(i)
                    # real tokens only: no CLS, SEP or padding
                    positions = [position for position, word_id in enumerate(word_ids) if word_id is not None]
                    meta["word_ids"].append(np.array([word_ids[position] for position in positions], dtype = np.int32))
                    meta["lengths"].append(len(positions))
                    for layer in layers:
                        states[layer].append(outputs.hidden_states[layer][i, positions].to(torch.float16).cpu().numpy())
            for layer, path in paths.items():
                np.save(path, np.concatenate(states[layer]))
            meta_path = os.path.join(args.cache_dir, f"meta_{window}_{split}.pkl")
            if not os.path.exists(meta_path):
                if split == "train":
                    # replacement weight (at p = 1) of the word of every token
                    meta["r"] = [train_probs[doc_id][word_ids].astype(np.float32) for doc_id, word_ids in zip(indices, meta["word_ids"])]
                else:
                    meta["gold_words"] = [to_gold_words(doc[2]) for doc in docs]
                meta["words"] = [doc[0] for doc in docs]
                with open(meta_path, "wb") as f:
                    pickle.dump(meta, f)
            print(f"cached {encoder} {window} {split} layers {layers} in {time.time() - start:.0f} s", flush = True)
        del model
        if device == "mps":
            torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# run: train the generator's heads on cached states
# ---------------------------------------------------------------------------

@dataclass(frozen = True)
class RunConfig:
    experiment: str
    encoder: str = "pretrained"
    window: str = "head"
    scorer_layer: int = 12
    evidence_layer: int = 12
    restrict: str = "all"         # attention allowed on: all, trivial (stopwords and punctuation), content
    noise_p: float = 0.0          # evidence noise level (fixed)
    pi_target: float = 0.0        # > 0: exposure-controlled noise level instead
    lr: float = 2e-3
    batch_size: int = 32
    epochs: int = 20
    updates: int = 0              # > 0: stop after this many updates
    seed: int = 1

    def key(self):
        # the experiment name is a label only: identical settings are one run
        return json.dumps({k: v for k, v in asdict(self).items() if k != "experiment"}, sort_keys = True)


def run_configs():
    configs = []
    add = lambda **kw: configs.extend(RunConfig(seed = seed, **kw) for seed in SEEDS)
    # Table 9: attention restricted to stopwords and punctuation, or to content words
    for encoder in ENCODERS:
        for restrict in ("all", "trivial", "content"):
            add(experiment = "restricted", encoder = encoder, restrict = restrict)
    # layers of the pretrained encoder
    for layer in (0, 4, 8):
        add(experiment = "layers", scorer_layer = layer, evidence_layer = layer)
    # fine-tuned scorer, context-free (layer-0) evidence
    add(experiment = "layer0_evidence", encoder = "imdb", evidence_layer = 0)
    # the same, attention limited to stopwords and punctuation: their layer-0
    # embeddings carry no sentiment, so accuracy above chance must come from
    # which of them (and where) the scorer attends to
    add(experiment = "layer0_evidence", encoder = "imdb", evidence_layer = 0, restrict = "trivial")
    # control: a context-free scorer (layer 0) cannot know the label when it
    # chooses among stopwords, so this measures what their identity alone carries
    add(experiment = "layer0_evidence", encoder = "imdb", scorer_layer = 0, evidence_layer = 0, restrict = "trivial")
    # Table 10: evidence noise on frozen pretrained states, both windows
    for window in ("head", "tail"):
        for p in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
            add(experiment = "evidence_noise", window = window, noise_p = p)
        for target in (0.9, 0.8, 0.7):
            add(experiment = "evidence_noise", window = window, pi_target = target)
    # evidence noise on fine-tuned states
    for p in (0.3, 1.0):
        add(experiment = "evidence_noise_imdb", encoder = "imdb", noise_p = p)
    # the setup with the full pipeline's learning rate, batch size and number of updates
    add(experiment = "pipeline", window = "tail", lr = 2e-5, batch_size = 16, epochs = 10**6, updates = 12500)
    # one run per distinct setting
    unique = {}
    for config in configs:
        unique.setdefault(config.key(), config)
    order = lambda c: (c.encoder, c.window, c.scorer_layer, c.evidence_layer, c.experiment, c.seed)
    return sorted(unique.values(), key = order)


class Split:
    """Cached states of one split, padded, on the device."""
    def __init__(self, cache_dir, encoder, window, split, layers, device):
        with open(os.path.join(cache_dir, f"meta_{window}_{split}.pkl"), "rb") as f:
            self.meta = meta = pickle.load(f)
        lengths = np.array(meta["lengths"])
        self.lengths = lengths
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        num_docs, max_len = len(lengths), int(lengths.max())
        self.states = {}
        for layer in layers:
            flat = np.load(os.path.join(cache_dir, f"{encoder}_{window}_L{layer}_{split}.npy"), mmap_mode = "r")
            padded = torch.zeros(num_docs, max_len, flat.shape[1], dtype = torch.float16)
            for i in range(num_docs):
                padded[i, :lengths[i]] = torch.from_numpy(np.array(flat[offsets[i]:offsets[i + 1]]))
            self.states[layer] = padded.to(device)
        positions = torch.arange(max_len)
        self.mask = (positions[None, :] < torch.as_tensor(lengths)[:, None]).to(device)
        trivial = torch.zeros(num_docs, max_len, dtype = torch.bool)
        for i, (words, word_ids) in enumerate(zip(meta["words"], meta["word_ids"])):
            trivial[i, :lengths[i]] = torch.tensor([is_trivial(words[w]) for w in word_ids])
        self.trivial = trivial.to(device)
        self.labels = torch.tensor(meta["labels"], dtype = torch.long, device = device)
        if "r" in meta:
            r = torch.zeros(num_docs, max_len)
            for i, values in enumerate(meta["r"]):
                r[i, :lengths[i]] = torch.from_numpy(values)
            self.r = r.to(device)

    def allowed(self, restrict):
        if restrict == "all":
            return self.mask
        allowed = self.mask & (self.trivial if restrict == "trivial" else ~self.trivial)
        # a review without such tokens keeps its full attention
        return torch.where(allowed.any(1, keepdim = True), allowed, self.mask)


class Heads(nn.Module):
    """The generator's scorer and the attention-based predictor's classifier (models.BlackBoxPredictor)."""
    def __init__(self, hidden_size = 768, dropout = 0.1):
        super().__init__()
        self.token_predictor = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))
        self.predictor = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 2))

    def forward(self, scorer_states, evidence_states, allowed, noise_probs = None):
        token_logits = self.token_predictor(scorer_states.float()).squeeze(-1).masked_fill(~allowed, float("-inf"))
        token_att = F.softmax(token_logits, -1)
        evidence = evidence_states.float()
        if noise_probs is not None:
            # evidence noise: a pooled state is replaced, with its probability,
            # by the state of a random real token of the batch
            is_word = noise_probs >= 0
            replace = (torch.rand(noise_probs.shape, device = noise_probs.device) < noise_probs) & is_word
            if replace.any():
                word_states = evidence[is_word].detach()
                evidence = evidence.clone()
                evidence[replace] = word_states[torch.randint(len(word_states), (int(replace.sum()),), device = word_states.device)]
        ctx_vec = torch.bmm(evidence.transpose(1, 2), token_att.unsqueeze(-1)).squeeze(-1)
        return F.softmax(self.predictor(ctx_vec), -1), token_att


def train_heads(config, train, device):
    torch.manual_seed(config.seed)
    heads = Heads().to(device)
    optimizer = torch.optim.Adam(heads.parameters(), lr = config.lr)
    generator = torch.Generator().manual_seed(config.seed)
    schedule = None
    if config.pi_target > 0:
        # the settings of the full-pipeline run with an exposure target (frozen_closed)
        schedule = PIController(signal = "exposure", target = config.pi_target, kp = 1.0, ki = 0.001,
                                ema = 0.99, p_min = 0.0, p_max = 1.0, p_init = 0.2)
    allowed = train.allowed(config.restrict)
    scorer_states, evidence_states = train.states[config.scorer_layer], train.states[config.evidence_layer]
    num_docs, step = len(train.lengths), 0
    epoch_exposure, epoch_p = [], []
    heads.train()
    for _ in range(config.epochs):
        permutation = torch.randperm(num_docs, generator = generator).to(device)
        exposures, levels = [], []
        for b in range(0, num_docs, config.batch_size):
            idx = permutation[b:b + config.batch_size]
            p = schedule.value(step) if schedule is not None else config.noise_p
            noise_probs = None
            if schedule is not None or p > 0:
                noise_probs = torch.where(train.mask[idx], (train.r[idx] * p).clamp(max = 1), torch.full_like(train.r[idx], -1.0))
            att_pred, token_att = heads(scorer_states[idx], evidence_states[idx], allowed[idx], noise_probs)
            # BlackBoxPredictor.get_loss without the closeness term: there is no rationale-based predictor
            loss = F.cross_entropy(att_pred, train.labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # attention exposure: attention-weighted replacement weight, 1 for uniform attention
            exposure = (token_att.detach() * train.r[idx]).sum(-1).mean().item()
            if schedule is not None:
                schedule.update(exposure)
            exposures.append(exposure)
            levels.append(p)
            step += 1
            if config.updates and step >= config.updates:
                break
        epoch_exposure.append(float(np.mean(exposures)))
        epoch_p.append(float(np.mean(levels)))
        if config.updates and step >= config.updates:
            break
    return heads, {"exposure_per_epoch": epoch_exposure, "p_per_epoch": epoch_p, "updates": step}


def window_scores(att, window):
    if window == 1:
        return att
    return F.avg_pool1d(att[None, None, :], kernel_size = window, stride = 1, padding = window // 2, count_include_pad = False)[0, 0]


def to_spans(mask):
    spans, start = [], None
    for i, value in enumerate(list(mask) + [False]):
        if value and start is None:
            start = i
        if not value and start is not None:
            spans.append(set(range(start, i)))
            start = None
    return spans


def evaluate(heads, test, config, windows = (1, 5), batch_size = 64):
    heads.eval()
    allowed = test.allowed(config.restrict)
    scorer_states, evidence_states = test.states[config.scorer_layer], test.states[config.evidence_layer]
    correct = 0
    counts = {w: [] for w in windows}
    iou = {w: {"hit": 0.0, "gen": 0, "gold": 0} for w in windows}
    with torch.no_grad():
        for b in range(0, len(test.lengths), batch_size):
            idx = torch.arange(b, min(b + batch_size, len(test.lengths)), device = scorer_states.device)
            att_pred, token_att = heads(scorer_states[idx], evidence_states[idx], allowed[idx])
            correct += (att_pred.argmax(-1) == test.labels[idx]).sum().item()
            token_att = token_att.float().cpu()
            for j, i in enumerate(idx.tolist()):
                n = int(test.lengths[i])
                word_ids = test.meta["word_ids"][i].tolist()
                gold_words = test.meta["gold_words"][i]
                gold_mask = [word_id in gold_words for word_id in word_ids]
                # as TopkSelector.get_k: round half to even, at least one token
                k = max(int(torch.round(torch.tensor(n * SPARSITY)).item()), 1)
                for w in windows:
                    selected = torch.zeros(n, dtype = torch.bool)
                    selected[window_scores(token_att[j, :n], w).topk(k).indices] = True
                    selected = selected.tolist()
                    counts[w].append(selection_counts(word_ids, selected, gold_words))
                    # IOU as in run_words.evaluate (micro, partial-match threshold 0.1)
                    gen_spans, gold_spans = to_spans(selected), to_spans(gold_mask)
                    for span in gen_spans:
                        best = max([len(span & g)/(len(span | g) + 1e-6) for g in gold_spans] + [0.0])
                        iou[w]["hit"] += len(span) if best >= 0.1 else 0
                    iou[w]["gen"] += sum(len(s) for s in gen_spans)
                    iou[w]["gold"] += sum(len(s) for s in gold_spans)
    report = {"accuracy": correct / len(test.lengths)}
    for w in windows:
        micro = plausibility(counts[w])["wordpiece"]["micro"]
        selection = selection_report(counts[w], test.meta["words"])["selection"]
        prec, rec = iou[w]["hit"] / max(iou[w]["gen"], 1), iou[w]["hit"] / max(iou[w]["gold"], 1)
        report[f"w{w}"] = {
            "P": micro["prec"], "R": micro["rec"], "F1": micro["F1"],
            "IOU": 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0,
            "trivial_share": selection["trivial_word_share"],
            "mean_span_length": selection["mean_span_length"],
            "realized_rate": selection["realized_rate_wordpiece"],
        }
    return report


def run(args):
    device = get_device()
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.dumps(json.loads(line)["config_key"]) for line in f}
    loaded = {}
    for config in run_configs():
        if json.dumps(config.key()) in done:
            continue
        layers = tuple(sorted({config.scorer_layer, config.evidence_layer}))
        name = (config.encoder, config.window, layers)
        if name not in loaded:
            loaded.clear()
            if device == "mps":
                torch.mps.empty_cache()
            loaded[name] = {split: Split(args.cache_dir, config.encoder, config.window, split, layers, device) for split in ("train", "test")}
        start = time.time()
        heads, training = train_heads(config, loaded[name]["train"], device)
        result = {"config": asdict(config), "config_key": config.key(), "training": training,
                  "test": evaluate(heads, loaded[name]["test"], config), "seconds": time.time() - start}
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")
        test = result["test"]
        print(f"{config.experiment:20s} {config.encoder:10s} {config.window} L{config.scorer_layer}/L{config.evidence_layer} "
              f"{config.restrict:7s} p={config.noise_p} pi={config.pi_target} seed {config.seed}: "
              f"acc {100*test['accuracy']:.1f} F1 {100*test['w1']['F1']:.1f} (w5 {100*test['w5']['F1']:.1f}) "
              f"E {training['exposure_per_epoch'][-1]:.3f} [{result['seconds']:.0f} s]", flush = True)


# ---------------------------------------------------------------------------
# classifier: how much the human rationale alone can carry
# ---------------------------------------------------------------------------

def classifier(args):
    device = get_device()
    model = AutoModelForSequenceClassification.from_pretrained(ENCODERS["imdb"]).to(device).eval()
    tokenizer = get_tokenizer("head")
    data = load_split(args.data_path, "test")
    docs = [data[i] for i in sample_indices(len(data), NUM_TEST)]

    def accuracy(texts, labels):
        correct, tokens = 0, []
        for b in range(0, len(texts), args.batch_size):
            encoded = tokenizer(texts[b:b + args.batch_size], is_split_into_words = True, truncation = True,
                                max_length = 512, padding = True, return_tensors = "pt")
            tokens.extend((encoded["attention_mask"].sum(1) - 2).tolist())
            with torch.no_grad():
                logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits
            correct += (logits.argmax(-1).cpu() == torch.tensor(labels[b:b + args.batch_size])).sum().item()
        return correct / len(texts), float(np.mean(tokens))

    labels = [doc[1] for doc in docs]
    # words of the review that survive HEAD truncation
    window_words = []
    for doc in docs:
        encoded = tokenizer(doc[0], is_split_into_words = True, truncation = True, max_length = 512)
        window_words.append({word_id for word_id in encoded.word_ids() if word_id is not None})
    gold = [sorted(to_gold_words(doc[2])) for doc in docs]
    variants = {
        "full_review_head": [doc[0] for doc in docs],
        "rationale_all": [[doc[0][w] for w in words] for doc, words in zip(docs, gold)],
        "rationale_in_head_window": [[doc[0][w] for w in words if w in window] for doc, words, window in zip(docs, gold, window_words)],
    }
    report = {}
    for name, texts in variants.items():
        # a review without rationale words inside the window gets an empty input
        acc, tokens = accuracy(texts, labels)
        report[name] = {"accuracy": acc, "mean_tokens": tokens}
        print(f"{name:26s} accuracy {100*acc:.1f} mean tokens {tokens:.1f}", flush = True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent = 2)


def main():
    parser = ArgumentParser()
    parser.add_argument("stage", choices = ["cache", "run", "classifier"])
    parser.add_argument("--data_path", required = True)
    parser.add_argument("--cache_dir")
    parser.add_argument("--out")
    parser.add_argument("--batch_size", type = int, default = 16)
    args = parser.parse_args()
    {"cache": cache, "run": run, "classifier": classifier}[args.stage](args)


if __name__ == "__main__":
    main()
