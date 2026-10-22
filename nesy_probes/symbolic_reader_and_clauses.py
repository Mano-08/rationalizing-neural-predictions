"""Neurosymbolic probe, part 2 (offline, test set, exploratory).

A. Why the re-implemented selection differs from the saved one in ~7% of reviews ([SEP] selection?).
B. A symbolic reader: a transparent lexicon classifier with negation (Hu & Liu lexicon) that reads only a rationale.
   Its accuracy is a training-free "symbolic sufficiency" of a rationale: high if the rationale carries readable
   sentiment evidence, near chance if it is a code (punctuation, function words).
C. Clause-structured selection: select whole clauses (split at punctuation, optionally at discourse markers) ranked by
   the frozen generator's attention, instead of a fixed 5-token window. Variant chosen on DEV, reported on EVAL.
Usage: python3 -B nesy_probes/symbolic_reader_and_clauses.py <opinion_lexicon.zip>
"""
import sys
import json, zipfile, string
import numpy as np

import os
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + "/"
z = zipfile.ZipFile(sys.argv[1])
def huliu(name):
    return {w.strip() for w in z.read(f"opinion_lexicon/{name}").decode("latin-1").splitlines() if w.strip() and not w.startswith(";")}
POS, NEGW = huliu("positive-words.txt"), huliu("negative-words.txt")
NEGATION = {"not", "n't", "no", "never", "nothing", "none", "nobody", "neither", "nor", "without", "hardly", "barely", "cannot", "nt"}

test = [json.loads(l) for l in open(REPO + "usr_movie_review/test.jsonl")]
WORDS = [r[0] for r in test]; Y = np.array([r[1] for r in test])

def split_docs(path):
    a = np.load(REPO + path + "/attention.npz")
    lengths = a["lengths"]; att = a["attention"]; wid = a["word_ids"]; gold = a["gold"]; sel = a["selected"]
    assert (a["labels"] == Y).all(), "label order differs from test.jsonl"
    off = np.concatenate([[0], np.cumsum(lengths)])
    return [dict(att=att[off[i]:off[i+1]], wid=wid[off[i]:off[i+1]], gold=gold[off[i]:off[i+1]], sel=sel[off[i]:off[i+1]]) for i in range(len(lengths))]

def window(scores, real, w):
    if w == 1: return np.where(real, scores, -1.0)
    s = np.where(real, scores, 0.0); h = w // 2; n = len(s); c = np.concatenate([[0.0], np.cumsum(s)])
    lo = np.maximum(np.arange(n) - h, 0); hi = np.minimum(np.arange(n) + h + 1, n)
    return np.where(real, (c[hi] - c[lo]) / (hi - lo), -1.0)

def select(scores, real, w):
    k = max(1, int(np.round(0.2 * real.sum()))); sc = window(scores, real, w)
    m = np.zeros(len(sc), bool); m[np.argsort(-sc, kind="stable")[:k]] = True
    return m

def ranges(mask):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None: start = i
        if not v and start is not None: out.append(set(range(start, i))); start = None
    if start is not None: out.append(set(range(start, len(mask))))
    return out

def plaus(docs, masks, ids):
    tp = fp = fn = 0; hit = gt = rt = 0
    for i in ids:
        m, g = masks[i], docs[i]["gold"]
        tp += int((m & g).sum()); fp += int((m & ~g).sum()); fn += int((~m & g).sum())
        gs, rs = ranges(m), ranges(g); gt += sum(map(len, gs)); rt += sum(map(len, rs))
        hit += sum(len(s) for s in gs if max([len(s & r) / (len(s | r) + 1e-6) for r in rs] + [0.0]) >= 0.1)
    P, R = tp / (tp + fp), tp / (tp + fn); ip, ir = hit / gt, hit / rt
    return 100 * 2 * P * R / (P + R), 100 * 2 * ip * ir / (ip + ir)

def reader(i, wids):
    """Transparent symbolic classifier over the words a rationale keeps (document order, deduplicated)."""
    words = [WORDS[i][x].lower() for x in sorted(set(int(v) for v in wids if v >= 0))]
    score, neg_left = 0, 0
    for w in words:
        if w in NEGATION:
            neg_left = 3; continue
        pol = (w in POS) - (w in NEGW)
        if pol:
            score += -pol if neg_left > 0 else pol
        neg_left = max(0, neg_left - 1)
    return score

def reader_acc(scores):
    s = np.asarray(scores)
    return 100 * np.mean(np.where(s == 0, 0.5, ((s > 0).astype(int) == Y).astype(float))), 100 * np.mean(s != 0)

RUN = {"A2R": {1: "results_v11/probe/a2r_tail_seed1__tail", 2: "results_v10/probe/a2r_tail_seed2__tail", 3: "results_v9/probe/a2r_tail_seed3__tail"},
       "NI as described": {1: "results_v11/probe/ni_tail_seed1__tail", 2: "results_v10/probe/ni_tail_seed2__tail", 3: "results_v8/probe/ni_tail_seed3__tail"},
       "NI as released": {1: "results_v13/probe/ni_released_tail_seed1__tail", 2: "results_v12/probe/ni_released_tail_seed2__tail", 3: "results_v14/probe/ni_released_tail_seed3__tail"},
       "FG (frozen)": {1: "results_v3/probe/frozen_tail_w5_seed1__tail", 2: "results_v5/probe/frozen_tail_w5_seed2__tail", 3: "results_v7/probe/frozen_tail_w5_seed3__tail"}}
DOCS = {(m, s): split_docs("results/" + p) for m, d in RUN.items() for s, p in d.items()}

# ---------------------------------------------------------------- A. [SEP] selection
d1 = DOCS[("FG (frozen)", 1)]
sep_sel = sum(int((d["sel"] & (d["wid"] < 0)).any()) for d in d1)
mine = [select(d["att"], d["wid"] >= 0, 1) for d in d1]
diff = [i for i, (m, d) in enumerate(zip(mine, d1)) if not (m == d["sel"]).all()]
diff_sep = sum(int((d1[i]["sel"] & (d1[i]["wid"] < 0)).any()) for i in diff)
kdiff = sum(int(d1[i]["sel"].sum() != mine[i].sum()) for i in diff)
print(f"[A] FG seed 1: saved selection contains a special token in {sep_sel} reviews; of the {len(diff)} differing reviews, "
      f"{diff_sep} contain a selected special token and {kdiff} have a different number of selected positions")

# ---------------------------------------------------------------- B. symbolic reader
full = [reader(i, d["wid"]) for i, d in enumerate(d1)]
rng = np.random.default_rng(0)
rand = [reader(i, d["wid"][select(rng.random(len(d["wid"])), d["wid"] >= 0, 1)]) for i, d in enumerate(d1)]
fa, fc = reader_acc(full); ra, rc = reader_acc(rand)
print(f"\n[B] symbolic reader accuracy (ties count 0.5), coverage = share of reviews with a nonzero score")
print(f"    full TAIL window: {fa:.1f} (coverage {fc:.0f}%); random 20% selection: {ra:.1f} (coverage {rc:.0f}%)")
rows = []
for m in RUN:
    for w in (1, 5):
        accs, f1s = [], []
        for s in (1, 2, 3):
            docs = DOCS[(m, s)]
            masks = [d["sel"] if w == 1 else select(d["att"], d["wid"] >= 0, 5) for d in docs]
            a, c = reader_acc([reader(i, d["wid"][mk]) for i, (d, mk) in enumerate(zip(docs, masks))])
            accs.append(a); f1s.append(plaus(docs, masks, range(2000))[0])
        rows.append((m, w, accs, f1s))
        print(f"    {m:16s} w={w}: reader accuracy {accs[0]:.1f} / {accs[1]:.1f} / {accs[2]:.1f} (mean {np.mean(accs):.1f}); token F1 {np.mean(f1s):.1f}")
x = np.array([a for _, w, accs, _ in rows if w == 1 for a in accs]); yv = np.array([f for _, w, _, f1s in rows if w == 1 for f in f1s])
print(f"    Pearson r between reader accuracy and token F1 over the 12 single-token runs: {np.corrcoef(x, yv)[0, 1]:.2f}")

# ---------------------------------------------------------------- C. clause-structured selection (frozen generator)
PUNCT = {".", ",", ";", ":", "!", "?", "--", "(", ")", "-"}
MARKERS = {"but", "however", "although", "though", "yet", "while", "whereas", "and"}
def clause_select(i, d, boundaries, agg):
    real = d["wid"] >= 0; k = max(1, int(np.round(0.2 * real.sum())))
    seg = np.full(len(real), -1); cur = 0
    for p in np.where(real)[0]:
        seg[p] = cur
        if WORDS[i][d["wid"][p]] in boundaries: cur += 1
    ids = [np.where(seg == c)[0] for c in range(cur + 1)]; ids = [q for q in ids if len(q)]
    scores = [(d["att"][q].mean() if agg == "mean" else d["att"][q].max()) for q in ids]
    m = np.zeros(len(real), bool); left = k
    for c in np.argsort(-np.array(scores), kind="stable"):
        q = ids[c]
        if left <= 0: break
        take = q if len(q) <= left else q[np.argsort(-d["att"][q], kind="stable")[:left]]
        m[take] = True; left -= len(take)
    return m
DEV, EVAL = range(0, 2000, 2), range(1, 2000, 2)
variants = {("punct", "mean"): PUNCT, ("punct", "max"): PUNCT, ("punct+markers", "mean"): PUNCT | MARKERS, ("punct+markers", "max"): PUNCT | MARKERS}
devF = {}
for (bname, agg), b in variants.items():
    devF[(bname, agg)] = np.mean([plaus(DOCS[("FG (frozen)", s)], [clause_select(i, d, b, agg) for i, d in enumerate(DOCS[("FG (frozen)", s)])], DEV)[0] for s in (1, 2, 3)])
base_dev = np.mean([plaus(DOCS[("FG (frozen)", s)], [select(d["att"], d["wid"] >= 0, 5) for d in DOCS[("FG (frozen)", s)]], DEV)[0] for s in (1, 2, 3)])
best = max(devF, key=devF.get)
print(f"\n[C] dev: PS (w=5) F1 {base_dev:.2f}; clause variants: " + ", ".join(f"{k[0]}/{k[1]} {v:.2f}" for k, v in devF.items()) + f"; best {best}")
for s in (1, 2, 3):
    docs = DOCS[("FG (frozen)", s)]
    b5 = plaus(docs, [select(d["att"], d["wid"] >= 0, 5) for d in docs], EVAL)
    cl = plaus(docs, [clause_select(i, d, variants[best], best[1]) for i, d in enumerate(docs)], EVAL)
    print(f"    eval seed {s}: PS w=5 F1 {b5[0]:.2f} IOU {b5[1]:.2f} | clauses {best} F1 {cl[0]:.2f} IOU {cl[1]:.2f}")
