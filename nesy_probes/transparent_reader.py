"""Transparent reader and code gap (offline, uses saved test-set outputs; no GPU, no human rationales needed).

A Naive Bayes log-count-ratio model (Wang & Manning, 2012) over binary unigrams is trained on the 40,000 USR training
reviews (last 400 words, approximating the TAIL window). Its decision is a readable sum of word weights. It then reads
only the words of each model's rationale (TAIL window, single tokens = the saved selection; phrases = w=5 window
re-computed from the saved attention). The code gap of a run is the accuracy of its own BERT rationale predictor minus
the accuracy of this fixed transparent reader on the same rationales.

Usage (from the repository root): python3 -B nesy_probes/transparent_reader.py
"""
import json, os
from collections import Counter
import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + "/"
train = [json.loads(l) for l in open(REPO + "usr_movie_review/train.jsonl")]
test = [json.loads(l) for l in open(REPO + "usr_movie_review/test.jsonl")]
WORDS = [r[0] for r in test]; Y = np.array([r[1] for r in test])

# ---------------------------------------------------------------- the transparent reader
df = Counter()
for words, _ in train:
    df.update(set(w.lower() for w in words[-400:]))
vocab = {w: i for i, w in enumerate(w for w, c in df.items() if c >= 5)}
p = np.ones(len(vocab)); q = np.ones(len(vocab)); npos = nneg = 0
for words, y in train:
    idx = [vocab[w] for w in set(x.lower() for x in words[-400:]) if w in vocab]
    if y == 1: p[idx] += 1; npos += 1
    else: q[idx] += 1; nneg += 1
weights = np.log(p / p.sum()) - np.log(q / q.sum()); bias = np.log(npos / nneg)
inv = {i: w for w, i in vocab.items()}; order = np.argsort(weights)
print(f"vocabulary {len(vocab)} words; training reviews pos/neg {npos}/{nneg}")
print("largest positive weights:", ", ".join(f"{inv[i]} {weights[i]:.2f}" for i in order[::-1][:12]))
print("largest negative weights:", ", ".join(f"{inv[i]} {weights[i]:.2f}" for i in order[:12]))

def predict(i, wids):
    ws = {WORDS[i][int(x)].lower() for x in wids if x >= 0}
    return (sum(weights[vocab[w]] for w in ws if w in vocab) + bias) > 0

def accuracy(docs, masks):
    pred = [predict(i, d["wid"] if m is None else d["wid"][m]) for i, (d, m) in enumerate(zip(docs, masks))]
    return 100 * np.mean(np.array(pred, int) == Y)

# ---------------------------------------------------------------- saved selections and attention
def split_docs(path):
    a = np.load(REPO + path + "/attention.npz")
    lengths = a["lengths"]; att = a["attention"]; wid = a["word_ids"]; sel = a["selected"]
    off = np.concatenate([[0], np.cumsum(lengths)])
    return [dict(att=att[off[i]:off[i+1]], wid=wid[off[i]:off[i+1]], sel=sel[off[i]:off[i+1]]) for i in range(len(lengths))]

def phrases(att, real, w=5):
    k = max(1, int(np.round(0.2 * real.sum()))); s = np.where(real, att, 0.0); h = w // 2; n = len(s)
    c = np.concatenate([[0.0], np.cumsum(s)]); lo = np.maximum(np.arange(n) - h, 0); hi = np.minimum(np.arange(n) + h + 1, n)
    sc = np.where(real, (c[hi] - c[lo]) / (hi - lo), -1.0); m = np.zeros(n, bool); m[np.argsort(-sc, kind="stable")[:k]] = True
    return m

RUNS = {"A2R": {1: "results_v11/probe/a2r_tail_seed1", 2: "results_v10/probe/a2r_tail_seed2", 3: "results_v9/probe/a2r_tail_seed3"},
        "NI as described": {1: "results_v11/probe/ni_tail_seed1", 2: "results_v10/probe/ni_tail_seed2", 3: "results_v8/probe/ni_tail_seed3"},
        "NI as released": {1: "results_v13/probe/ni_released_tail_seed1", 2: "results_v12/probe/ni_released_tail_seed2", 3: "results_v14/probe/ni_released_tail_seed3"},
        "FG (frozen)": {1: "results_v3/probe/frozen_tail_w5_seed1", 2: "results_v5/probe/frozen_tail_w5_seed2", 3: "results_v7/probe/frozen_tail_w5_seed3"}}

base = split_docs("results/" + RUNS["FG (frozen)"][1] + "__tail")
rng = np.random.default_rng(0); rand = []
for d in base:
    real = d["wid"] >= 0; k = max(1, int(np.round(0.2 * real.sum())))
    m = np.zeros(len(real), bool); m[rng.choice(np.where(real)[0], size=k, replace=False)] = True; rand.append(m)
print(f"\nreader accuracy: whole TAIL window {accuracy(base, [None] * 2000):.1f}; random 20% selection {accuracy(base, rand):.1f}")

rows = []
for model, runs in RUNS.items():
    for seed, prefix in runs.items():
        docs = split_docs("results/" + prefix + "__tail")
        res = json.load(open(REPO + "results/" + prefix + "__tail/results.json"))
        f1 = 100 * res["rationales"]["micro"]["F1"]; bert = 100 * res["accuracy"]
        c = np.array(json.load(open(REPO + "results/" + prefix + "__tail_random/per_example.json"))["correct"])
        pred = np.where(c == 1, Y, 1 - Y); one_class = 100 * max(pred.mean(), 1 - pred.mean())
        a1 = accuracy(docs, [d["sel"] for d in docs]); a5 = accuracy(docs, [phrases(d["att"], d["wid"] >= 0) for d in docs])
        rows.append((model, seed, f1, bert, a1, a5, bert - a1, one_class))
print(f"\n{'model':16s} seed  token F1  BERT acc  reader acc (tokens / phrases)  code gap  one-class share on random selections")
for model, seed, f1, bert, a1, a5, gap, one in rows:
    print(f"{model:16s} {seed:4d}  {f1:8.1f}  {bert:8.1f}  {a1:10.1f} / {a5:5.1f}            {gap:8.1f}  {one:6.1f}%")
F1 = np.array([r[2] for r in rows]); READER = np.array([r[4] for r in rows]); GAP = np.array([r[6] for r in rows]); ONE = np.array([r[7] for r in rows])
code = ONE > 90
print(f"\ncode gap with a code (>90% one class on random selections): {GAP[code].min():.1f} to {GAP[code].max():.1f}; without: {GAP[~code].min():.1f} to {GAP[~code].max():.1f}")
print(f"Pearson r: reader accuracy vs token F1 {np.corrcoef(READER, F1)[0, 1]:.2f}; code gap vs token F1 {np.corrcoef(GAP, F1)[0, 1]:.2f}; code gap vs one-class share {np.corrcoef(GAP, ONE)[0, 1]:.2f}")
