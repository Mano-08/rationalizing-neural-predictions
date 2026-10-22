"""Offline neurosymbolic probe: does a symbolic sentiment prior improve the frozen generator's rationales?

Uses the saved attention of the FG+PS models (TAIL window, results/*/probe/frozen_tail_w5_seed*__tail/attention.npz).
Selection replicates TopkSelector (k = round(0.2 * real tokens), window average with count_include_pad=False).
Metrics replicate run_words.evaluate: wordpiece micro P/R/F1 and micro IOU-F1 at threshold 0.1.
The prior strength alpha and the lexicon are chosen on even-indexed reviews (dev) and reported on odd-indexed ones (eval).
Usage: python3 -B nesy_probes/lexicon_prior.py <vader_lexicon.txt> <opinion_lexicon.zip>
"""
import sys
import json, zipfile, string
import numpy as np

VADER, HULIU = sys.argv[1], sys.argv[2]
import os
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + "/"
RUNS = {1: "results/results_v3/probe/frozen_tail_w5_seed1__tail", 2: "results/results_v5/probe/frozen_tail_w5_seed2__tail",
        3: "results/results_v7/probe/frozen_tail_w5_seed3__tail"}

# ---------------------------------------------------------------- lexicons (read as plain data)
z = zipfile.ZipFile(HULIU)
def huliu(name):
    return {w.strip() for w in z.read(f"opinion_lexicon/{name}").decode("latin-1").splitlines() if w.strip() and not w.startswith(";")}
HL = huliu("positive-words.txt") | huliu("negative-words.txt")
VD = set()
for line in open(VADER, encoding="utf-8", errors="replace"):
    parts = line.rstrip("\n").split("\t")
    if len(parts) >= 2:
        try:
            if abs(float(parts[1])) >= 1.0 and any(c.isalpha() for c in parts[0]):
                VD.add(parts[0].lower())
        except ValueError:
            pass
NEG = {"not", "n't", "no", "never", "nothing", "none", "nobody", "neither", "nor", "without", "hardly", "barely", "cannot", "nt"}
LEXICONS = {"HuLiu": HL, "VADER": VD, "HuLiu+VADER": HL | VD, "HuLiu+VADER+neg": HL | VD | NEG}
print("lexicon sizes:", {k: len(v) for k, v in LEXICONS.items()})

STOP = None
sys.path.insert(0, REPO + "model")
from metrics import is_trivial  # pure-python helper, no side effects

test = [json.loads(l) for l in open(REPO + "usr_movie_review/test.jsonl")]
WORDS = [r[0] for r in test]

# ---------------------------------------------------------------- helpers
def split_docs(path):
    a = np.load(REPO + path + "/attention.npz")
    lengths = a["lengths"]; att = a["attention"]; wid = a["word_ids"]; gold = a["gold"]; sel = a["selected"]  # decompress once
    off = np.concatenate([[0], np.cumsum(lengths)])
    return [dict(att=att[off[i]:off[i+1]], wid=wid[off[i]:off[i+1]], gold=gold[off[i]:off[i+1]],
                 sel=sel[off[i]:off[i+1]]) for i in range(len(lengths))]

def tie_at_k(d):
    real = d["wid"] >= 0; k = max(1, int(np.round(0.2 * real.sum())))
    s = np.sort(d["att"][real])[::-1]
    return k < len(s) and s[k - 1] == s[k]

def window(scores, real, w):
    if w == 1:
        return np.where(real, scores, -1.0)
    s = np.where(real, scores, 0.0)
    h = w // 2; n = len(s)
    c = np.concatenate([[0.0], np.cumsum(s)])
    lo = np.maximum(np.arange(n) - h, 0); hi = np.minimum(np.arange(n) + h + 1, n)
    avg = (c[hi] - c[lo]) / (hi - lo)
    return np.where(real, avg, -1.0)

def select(scores, real, w):
    k = max(1, int(np.round(0.2 * real.sum())))  # torch.round = round half to even, like np.round
    sc = window(scores, real, w)
    idx = np.argsort(-sc, kind="stable")[:k]
    m = np.zeros(len(sc), bool); m[idx] = True
    return m

def ranges(mask):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None: start = i
        if not v and start is not None: out.append(set(range(start, i))); start = None
    if start is not None: out.append(set(range(start, len(mask))))
    return out

def evaluate(docs, masks, ids):
    tp = fp = fn = 0; iou_hit = gen_tok = rat_tok = 0; triv = nsel = 0
    for i in ids:
        d, m = docs[i], masks[i]
        g = d["gold"]
        tp += int((m & g).sum()); fp += int((m & ~g).sum()); fn += int((~m & g).sum())
        gs, rs = ranges(m), ranges(g)
        gen_tok += sum(len(s) for s in gs); rat_tok += sum(len(s) for s in rs)
        for s in gs:
            best = max([len(s & r) / (len(s | r) + 1e-6) for r in rs] + [0.0])
            iou_hit += len(s) if best >= 0.1 else 0
        sw = set(int(x) for x in d["wid"][m] if x >= 0)
        nsel += len(sw); triv += sum(is_trivial(WORDS[i][x]) for x in sw)
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1); F = 2 * P * R / max(P + R, 1e-12)
    ip = iou_hit / max(gen_tok, 1); ir = iou_hit / max(rat_tok, 1); IF = 2 * ip * ir / max(ip + ir, 1e-12)
    return dict(P=100 * P, R=100 * R, F1=100 * F, IOU=100 * IF, stop=triv / max(nsel, 1), counts=(tp, fp, fn))

def lexmask(i, wid, lex):
    w = WORDS[i]
    return np.array([x >= 0 and w[x].lower() in lex for x in wid], dtype=float)

ALL = range(2000); DEV = range(0, 2000, 2); EVAL = range(1, 2000, 2)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- lexicon statistics in the TAIL window
docs1 = split_docs(RUNS[1])
for name, lex in LEXICONS.items():
    g_in = g_all = o_in = o_all = 0
    for i, d in enumerate(docs1):
        real = d["wid"] >= 0; L = lexmask(i, d["wid"], lex).astype(bool)
        g_in += int((L & d["gold"]).sum()); g_all += int(d["gold"].sum())
        o_in += int((L & real & ~d["gold"]).sum()); o_all += int((real & ~d["gold"]).sum())
    print(f"[stats] {name:16s} share of gold wordpieces in lexicon {100*g_in/g_all:5.1f}% | of non-gold {100*o_in/o_all:5.1f}% | "
          f"lexicon precision for gold {100*g_in/(g_in+o_in):5.1f}% (gold base rate {100*g_all/(g_all+o_all):.1f}%)")

# ---------------------------------------------------------------- reproduce the paper, then the symbolic variants
results = {}
for seed, path in RUNS.items():
    docs = docs1 if seed == 1 else split_docs(path)
    rp = json.load(open(REPO + path + "/results.json")); rp5 = json.load(open(REPO + path.replace("__tail", "__tail_w5") + "/results.json"))
    m1 = [select(d["att"], d["wid"] >= 0, 1) for d in docs]
    same = sum(int((a == d["sel"]).all()) for a, d in zip(m1, docs))
    diff_tied = sum(int(tie_at_k(d)) for a, d in zip(m1, docs) if not (a == d["sel"]).all())
    print(f"[check] seed {seed}: {2000 - same} reviews differ from the saved selection, {diff_tied} of them have a tie at the k-th score")
    e1 = evaluate(docs, m1, ALL)
    m5 = [select(d["att"], d["wid"] >= 0, 5) for d in docs]; e5 = evaluate(docs, m5, ALL)
    print(f"[check] seed {seed}: w1 selection identical in {same}/2000 docs; F1 {e1['F1']:.2f} (paper {100*rp['rationales']['micro']['F1']:.2f}), "
          f"IOU {e1['IOU']:.2f} (paper {100*rp['micro_iou']['0.1']['f1']:.2f}); w5 F1 {e5['F1']:.2f} (paper {100*rp5['rationales']['micro']['F1']:.2f}), "
          f"IOU {e5['IOU']:.2f} (paper {100*rp5['micro_iou']['0.1']['f1']:.2f})")
    results[seed] = dict(docs=docs, FG={1: m1, 5: m5})

# lexicon-only symbolic rationale (random tie-break), seed-independent scores
for name, lex in LEXICONS.items():
    for w in (1, 5):
        masks = []
        for i, d in enumerate(docs1):
            real = d["wid"] >= 0
            masks.append(select(lexmask(i, d["wid"], lex) + 1e-3 * rng.random(len(real)), real, w))
        e = evaluate(docs1, masks, ALL)
        print(f"[lexicon only] {name:16s} w={w}: P {e['P']:.1f} R {e['R']:.1f} F1 {e['F1']:.2f} IOU {e['IOU']:.2f} stop {e['stop']:.2f}")

# FG x lexicon prior: score = att * (1 + alpha * lex); choose lexicon and alpha on DEV (mean over seeds), report on EVAL
ALPHAS = (0.5, 1, 2, 4, 8)
LEXMASKS = {name: [lexmask(i, d["wid"], lex) for i, d in enumerate(docs1)] for name, lex in LEXICONS.items()}
def combo(seed, name, alpha, w):
    docs = results[seed]["docs"]
    return [select(d["att"] * (1 + alpha * LEXMASKS[name][i]), d["wid"] >= 0, w) for i, d in enumerate(docs)]
for w in (1, 5):
    base_dev = np.mean([evaluate(results[s]["docs"], results[s]["FG"][w], DEV)["F1"] for s in RUNS])
    grid = {}
    for name in LEXICONS:
        for a in ALPHAS:
            grid[(name, a)] = np.mean([evaluate(results[s]["docs"], combo(s, name, a, w), DEV)["F1"] for s in RUNS])
    best = max(grid, key=grid.get)
    print(f"\n[dev, w={w}] FG baseline F1 {base_dev:.2f}; best prior {best} F1 {grid[best]:.2f}; all: " +
          ", ".join(f"{k[0]}/{k[1]}:{v:.2f}" for k, v in sorted(grid.items(), key=lambda kv: -kv[1])[:8]))
    # report on EVAL, per seed, with a paired bootstrap over eval reviews
    for s in RUNS:
        docs = results[s]["docs"]; mb = results[s]["FG"][w]; mc = combo(s, best[0], best[1], w)
        eb, ec = evaluate(docs, mb, EVAL), evaluate(docs, mc, EVAL)
        # paired bootstrap of micro F1 difference
        ev = list(EVAL)
        cb = np.array([[int((mb[i] & docs[i]["gold"]).sum()), int((mb[i] & ~docs[i]["gold"]).sum()), int((~mb[i] & docs[i]["gold"]).sum())] for i in ev], float)
        cc = np.array([[int((mc[i] & docs[i]["gold"]).sum()), int((mc[i] & ~docs[i]["gold"]).sum()), int((~mc[i] & docs[i]["gold"]).sum())] for i in ev], float)
        bs = np.random.default_rng(0).integers(0, len(ev), size=(5000, len(ev)))
        def f1(c, idx):
            t = c[idx, 0].sum(1); fp_ = c[idx, 1].sum(1); fn_ = c[idx, 2].sum(1); p = t / (t + fp_); r = t / (t + fn_); return 2 * p * r / (p + r)
        d = (f1(cc, bs) - f1(cb, bs)) * 100; lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"[eval, w={w}] seed {s}: FG F1 {eb['F1']:.2f} IOU {eb['IOU']:.2f} stop {eb['stop']:.2f} -> FG x {best[0]} (alpha {best[1]}) "
              f"F1 {ec['F1']:.2f} IOU {ec['IOU']:.2f} stop {ec['stop']:.2f}; dF1 {ec['F1']-eb['F1']:+.2f} (95% CI {lo:+.2f}, {hi:+.2f})")
