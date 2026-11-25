# Runs the experiments on Kaggle. Python version of cmd.sh.
#
# Notebook settings: Accelerator = GPU, Internet = On.
# Paste this whole file into ONE code cell, set STAGES below and run the cell.
# It clones the repository if needed and runs the stages in order.
#
# After the cell has run once, more stages can be started from any other cell:
#     run("ni")
#     SEED = 2; run("ni", "exp")
#
# Kaggle's "GPU T4 x2" has two GPUs. Two training stages can run at the same
# time, one per GPU, which takes about as long as one of them alone:
#     run_parallel("ni", "released")
# or, in STAGES, as a list inside the list:
#     STAGES = ["setup", ["ni", "released"], "pack"]
# Their output goes to log files in runs/logs; the notebook shows a status
# line per run every few minutes and the end of each log when they finish.
#
# Stages, in the order they are meant to be run:
#   setup     install dependencies, generate word statistics (CPU, ~1 min)
#   smoke     train and evaluate on a small subset (a few minutes)
#   ni        NI with the alignment fix (1 full run)
#   released  NI exactly as released (1 full run)
#   baseline  both of the above; too long for one Kaggle session
#   control   random-mask control on the fixed NI checkpoint (evaluation only)
#   judge     full-text classifier: accuracy ceiling and faithfulness judge (1 full run)
#   rescore   re-evaluate finished runs with the judge (evaluation only)
#   prep      cache MLM substitutes and saliency probabilities (needs judge)
#   a2r       no noise
#   exp       exponential decay schedule
#   cos       cosine decay schedule
#   closed    closed-loop controller
#   full      closed loop + MLM substitutes + saliency (needs prep)
#   frozen    no noise, generator encoder frozen (the paper's "fixed gen. weights")
#   frozen_ni fixed noise, generator encoder frozen
#   frozen_en frozen generator, noise in the generator's own evidence (level EVIDENCE_P)
#   frozen_closed  the same with the noise level held by a feedback
#             controller on the exposure of the attention (target EXPOSURE)
#   probe     re-score finished runs under other settings, without training:
#             beginning or end of long reviews, single tokens or phrases,
#             random control. Finds runs of this session and of earlier
#             sessions attached under /kaggle/input (Add Input)
#   table     mean (std) over seeds and significance tests
#   pack      collect result files into results.tar.gz
#
# A finished run (results.json exists) is skipped, so a stage can be re-run
# after a session was cut off.
#
# TRUNCATION and WINDOW below change what every training stage does and are
# part of the run name (ni_tail_w5_seed1), so runs with different settings
# do not overwrite each other.
#
# DATASET = "multirc" or "fever" runs the ERASER datasets with sentence-level
# rationales through model/run_ce.py instead (see TO_RUN.md). "setup"
# downloads and preprocesses the dataset. Available stages: setup, smoke, a2r,
# ni, released, frozen, probe (random-selection control), pack. Run names
# start with the dataset (multirc_ni_seed1); TRUNCATION and WINDOW do not
# apply. The noise level is that of Storek et al. (2023) for the dataset;
# epochs, patience and validation as in the USR runs.
#
# Round 3 (TO_RUN.md):
#   MultiRC/FEVER: a2r_dm, ni_dm, released_dm, frozen_dm  the generator attends to
#             document tokens only (--document_attention)
#   query_only  query-only BERT (cross-fitted, seed QUERY_ONLY_SEED), needed by:
#   a2r_dm_poe, frozen_dm_poe, frozen_x_poe  generator trained as a product of
#             experts with the query-only model (--poe_probs)
#   smoke_r3  each round-3 option on small subsets (minutes)
# DATASET = "hatexplain" runs HateXplain (GenSPP split) through run_words.py
# with the USR stages (a2r, ni, frozen, probe, pack); "setup" downloads and
# prepares it. Selection rate HX_SPARSITY, special tokens never selected,
# at most HX_EPOCHS epochs.
# A stage name may end in @wN to use WINDOW = N for that stage only, e.g.
#     STAGES = ["setup", ["released", "frozen@w5"], "probe", "pack"]

import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time

STAGES = ["setup", "smoke"]

SEED = 1      # random seed, also part of the run name
BATCH = 8     # the paper uses 16, which needs more than the 16 GB of a Kaggle GPU
P = 0.2       # final / fixed noise level
P0 = 0.5      # initial noise level of decaying schedules
GAMMA = 1.0   # exponential decay rate per epoch
EVIDENCE_P = 0.3       # noise level of frozen_en
EXPOSURE = 0.9         # exposure the controller of frozen_closed holds
TRUNCATION = "right"   # reviews longer than 512 tokens: "right" keeps the beginning, "left" keeps the end
WINDOW = 1             # rank tokens by attention averaged over this many tokens (odd); 5 selects phrases
DATASET = "usr"        # "usr" (USR Movie Review), "multirc", "fever" or "hatexplain"
TIME_LIMIT_HOURS = 9.5 # stop training after this long and evaluate the best checkpoint, so that a run
                       # fits into a 12-hour Kaggle session (USR runs since round 2, MultiRC/FEVER always)

# Round 2 (2026-11-23), fixed before any run: DR sets the predictor's learning
# rate to the selection rate times the generator's (Liu et al., 2023): 0.2 x 2e-5;
# vicinal swaps 20% of the selected tokens; coupling passes the predictor's loss
# to the generator at full weight
DR_RP_LR = 4e-6
PERTURBATION = 0.2
COUPLING = 1.0
# USR probes run by the probe stage (None = all of PROBES; round 2 ran only
# "tail" and "tail_random", round 3 needs all of them for seeds 4 and 5)
USR_PROBES = None

# Round 3 (2026-11-25), fixed before any run.
# HateXplain as released by Mathew et al. (2021), pinned to the commits that last
# changed the two files; GenSPP's split is rebuilt from them (prepare_dataset.py).
HX_FILES = {"dataset.json": "2d804e29058726c76e835e043d9ec79169879fa6",
            "post_id_divisions.json": "b94c830456f2088714a5635a2c79edac652a5c7b"}
HX_URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/{commit}/Data/{name}"
# share of highlighted words in the training posts (9.96%); phrase selection
# uses w = 3, the odd window closest to the mean training span (2.2 words): frozen@w3
HX_SPARSITY = 0.1
# at most this many epochs (patience 2 as everywhere). Changed from 5 after
# results_v43-v45: with 674 updates per epoch (USR: 2,500), the frozen generator's
# validation loss was still falling after 5 epochs in 5 of 6 runs; the fine-tuned runs
# stopped by patience within 5 epochs. 30 epochs = 20,220 updates. Runs with a cap
# other than 5 are named hatexplain_<name>_e<cap>_seed<N>
HX_EPOCHS = 30
HX_PROBES = {"random": ["--selection_method", "random", "--mask_special_tokens"]}
# seed of the query-only model of every PoE run, so that all seeds share it
QUERY_ONLY_SEED = 1

# MultiRC/FEVER: noise level of Storek et al. (2023) for the dataset; epochs,
# patience and one validation per epoch as in our USR runs (run_words.py)
CE = {
    "multirc": {"p": 0.3, "epochs": 5, "patience": 2},
    "fever": {"p": 0.05, "epochs": 5, "patience": 2},
}
ERASER_URL = "http://www.eraserbenchmark.com/zipped/{}.tar.gz"
DATA_DIR = "/kaggle/working/data"

REPO_URL = "https://github.com/Mano-08/rationalizing-neural-predictions.git"
BRANCH = "ani"
REPO = "/kaggle/working/noise_injection"
OUT = "/kaggle/working/runs"

# Parallel runs: devices to use (None = every GPU found) and seconds between
# status lines
DEVICES = None
STATUS_EVERY = 300

# stages that are one independent training run and may run in parallel
PARALLEL_STAGES = ["ni", "released", "a2r", "exp", "cos", "closed", "full", "judge", "frozen", "frozen_ni", "frozen_en", "frozen_closed",
                   "dr", "vicinal", "ni_coupled", "a2r_coupled", "frozen_x",
                   "a2r_dm", "ni_dm", "released_dm", "frozen_dm", "a2r_dm_poe", "frozen_dm_poe", "frozen_x_poe"]

# probe stage: name -> options of run_words.py --evaluate
PROBES = {
    "head": ["--save_attention"],
    "head_w5": ["--selection_window", 5],
    "tail": ["--truncation_side", "left", "--save_attention"],
    "tail_w5": ["--truncation_side", "left", "--selection_window", 5],
    "tail_w9": ["--truncation_side", "left", "--selection_window", 9],
    "tail_random": ["--truncation_side", "left", "--selection_method", "random", "--mask_special_tokens"],
}
INPUTS = "/kaggle/input"

# device and log file of the parallel run a thread belongs to
_parallel = threading.local()


def is_ce():
    if DATASET not in ["usr", "hatexplain"] + list(CE):
        raise ValueError(f"Unknown DATASET {DATASET!r}: use 'usr', 'multirc', 'fever' or 'hatexplain'")
    return DATASET in CE


def is_hx():
    return not is_ce() and DATASET == "hatexplain"


def data_path():
    if is_ce() or is_hx():
        return os.path.join(DATA_DIR, DATASET)
    return os.path.join(REPO, "usr_movie_review")


def stats_path(name):
    return os.path.join(data_path(), "word_statistics", name)


def judge_path():
    return os.path.join(OUT, "full_text", "checkpoints", "full_text.pt")


def device_args():
    device = getattr(_parallel, "device", None)
    return ["--device", device] if device else []


def common():
    if is_hx():
        # short posts: no truncation; special tokens are never attended to or selected
        return ["--data_path", data_path(), "--batch_size", BATCH, "--seed", SEED, "--sparsity", HX_SPARSITY,
                "--mask_special_tokens", "--num_epochs", HX_EPOCHS] + device_args()
    truncation = ["--truncation_side", "left"] if TRUNCATION == "left" else []
    return ["--data_path", data_path(), "--batch_size", BATCH, "--seed", SEED] + truncation + device_args()


def window():
    # WINDOW, or the window of a stage given as "name@wN"
    override = getattr(_parallel, "window", None)
    return WINDOW if override is None else override


def split_stage(stage):
    # "frozen@w5" -> ("frozen", 5); "frozen" -> ("frozen", None)
    if "@w" in stage:
        name, size = stage.split("@w")
        if not size.isdigit() or int(size) % 2 == 0:
            raise ValueError(f"Stage {stage!r}: the window after @w must be an odd number")
        return name, int(size)
    return stage, None


def selection():
    # options of run_words.py only
    return ["--selection_window", window()] if window() > 1 else []


def run_name(name):
    if is_ce():
        return f"{DATASET}_{name}_seed{SEED}"
    if is_hx():
        return f"hatexplain_{name}" + (f"_w{window()}" if window() > 1 else "") + (f"_e{HX_EPOCHS}" if HX_EPOCHS != 5 else "") + f"_seed{SEED}"
    return name + ("_tail" if TRUNCATION == "left" else "") + (f"_w{window()}" if window() > 1 else "") + f"_seed{SEED}"


def sh(*command, cwd = None):
    # runs a command and streams its output into the notebook, or into the
    # log file when it belongs to a parallel run
    output = getattr(_parallel, "log", None) or sys.stdout
    command = [str(part) for part in command]
    print("$ " + " ".join(command), file = output, flush = True)
    # progress bars are refreshed every 30 s to keep the logs short
    env = dict(os.environ, PYTHONUNBUFFERED = "1", TQDM_MININTERVAL = "30")
    process = subprocess.Popen(command, cwd = cwd, env = env, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    for chunk in iter(lambda: process.stdout.read1(4096), b""):
        output.write(chunk.decode(errors = "replace"))
        output.flush()
    if process.wait() != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")


def python(script, *args):
    sh(sys.executable, script, *args, cwd = os.path.join(REPO, "model"))


def need(path, stage):
    if not os.path.exists(path):
        raise RuntimeError(f"Missing {path} - run the '{stage}' stage first")


def clone():
    if not os.path.exists(os.path.join(REPO, "model", "run_words.py")):
        sh("git", "clone", "-b", BRANCH, REPO_URL, REPO)


def ce_common():
    # options of run_ce.py shared by training and evaluation
    return ["--dataset", DATASET, "--data_path", data_path(), "--batch_size", BATCH, "--seed", SEED,
            "--selection_method", "sentences"] + device_args()


def ce_train_run(name, *args):
    # one full training + evaluation on MultiRC or FEVER
    name = run_name(name)
    save_path = os.path.join(OUT, name)
    if os.path.exists(os.path.join(save_path, "results.json")):
        print(f"== {name} is finished, skipping")
        return
    need(os.path.join(data_path(), "word_statistics", "train_replacement_probs.pkl"), "setup")
    if TRUNCATION != "right" or WINDOW != 1:
        print("Note: TRUNCATION and WINDOW do not apply to MultiRC/FEVER and are ignored")
    settings = CE[DATASET]
    print(f"== {name}")
    python("run_ce.py", "--train", "--evaluate", *ce_common(), "--num_epochs", settings["epochs"],
           "--patience", settings["patience"], "--validations_per_epoch", 1, "--time_limit_hours", TIME_LIMIT_HOURS,
           "--save_path", save_path, *args)


def not_for_ce(stage):
    if is_ce():
        raise RuntimeError(f"Stage '{stage}' is not available for {DATASET}; see TO_RUN.md")


def train_run(name, *args):
    # one full training + evaluation
    name = run_name(name)
    save_path = os.path.join(OUT, name)
    if os.path.exists(os.path.join(save_path, "results.json")):
        print(f"== {name} is finished, skipping")
        return
    print(f"== {name}")
    # the faithfulness judge is used as soon as it has been trained
    judge = ["--faithfulness_model", judge_path()] if os.path.exists(judge_path()) else []
    python("run_words.py", "--train", "--evaluate", *common(), *selection(), *judge, "--time_limit_hours", TIME_LIMIT_HOURS,
           "--save_path", save_path, *args)


def ce_setup():
    # download the ERASER dataset, then preprocess it as in the README
    if os.path.exists(os.path.join(data_path(), "word_statistics", "train_replacement_probs.pkl")):
        print(f"{DATASET} is ready in {data_path()}")
        return
    import nltk
    for resource in ["punkt", "punkt_tab"]:
        nltk.download(resource, quiet = True)
    raw = os.path.join(DATA_DIR, "raw")
    os.makedirs(raw, exist_ok = True)
    archive = os.path.join(raw, f"{DATASET}.tar.gz")
    if not os.path.exists(os.path.join(raw, DATASET, "train.jsonl")):
        import urllib.request
        print(f"Downloading {ERASER_URL.format(DATASET)}", flush = True)
        urllib.request.urlretrieve(ERASER_URL.format(DATASET), archive)
        with tarfile.open(archive) as tar:
            tar.extractall(raw, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    scripts = os.path.join(REPO, "data_preprocessing", DATASET)
    sh(sys.executable, "prepare_dataset.py", f"--data_path={os.path.join(raw, DATASET)}", f"--save_path={data_path()}",
       "--lower_case", cwd = scripts)
    sh(sys.executable, "generate_word_statistics.py", f"--data_path={data_path()}", cwd = scripts)
    sh("wc", "-l", *[os.path.join(data_path(), f"{split}.jsonl") for split in ["train", "valid", "test"]])


def stage_setup():
    sh(sys.executable, "-m", "pip", "install", "-q", "termcolor")
    if shutil.which("nvidia-smi"):
        sh("nvidia-smi")
    else:
        print("No GPU found - switch the accelerator on")
    if is_ce():
        ce_setup()
        return
    if is_hx():
        hx_setup()
    # word statistics of the training split (USR's script works for any dataset in its format)
    if not os.path.exists(stats_path("train_replacement_probs.pkl")):
        sh(sys.executable, "generate_word_statistics.py", f"--data_path={data_path()}",
           cwd = os.path.join(REPO, "data_preprocessing", "usr_movie_review"))


def hx_setup():
    # HateXplain: download the pinned release and rebuild GenSPP's split
    if os.path.exists(os.path.join(data_path(), "test.jsonl")):
        print(f"hatexplain is ready in {data_path()}")
        return
    import urllib.request
    raw = os.path.join(DATA_DIR, "raw", "hatexplain")
    os.makedirs(raw, exist_ok = True)
    for name, commit in HX_FILES.items():
        target = os.path.join(raw, name)
        if not os.path.exists(target):
            url = HX_URL.format(commit = commit, name = name)
            print(f"Downloading {url}", flush = True)
            urllib.request.urlretrieve(url, target)
    sh(sys.executable, "prepare_dataset.py", "--raw_path", raw, "--data_path", data_path(),
       cwd = os.path.join(REPO, "data_preprocessing", "hatexplain"))


def stage_smoke():
    if is_ce():
        need(os.path.join(data_path(), "word_statistics", "train_replacement_probs.pkl"), "setup")
        python("run_ce.py", "--train", "--evaluate", "--inject_noise", "--noise_p", CE[DATASET]["p"], *ce_common(),
               "--train_subset", 800, "--valid_subset", 200, "--test_subset", 200, "--num_epochs", 1,
               "--save_path", os.path.join(OUT, f"{DATASET}_smoke"))
        return
    need(stats_path("train_replacement_probs.pkl"), "setup")
    python("run_words.py", "--train", "--evaluate", "--inject_noise", "--noise_p", P, *common(), *selection(),
           "--train_subset", 800, "--valid_subset", 200, "--test_subset", 200, "--num_epochs", 1,
           "--save_path", os.path.join(OUT, "smoke"))


def stage_ni():
    if is_ce():
        # NI with the alignment fixed
        return ce_train_run("ni", "--inject_noise", "--noise_p", CE[DATASET]["p"])
    need(stats_path("train_replacement_probs.pkl"), "setup")
    train_run("ni", "--inject_noise", "--noise_p", P)


def stage_released():
    if is_ce():
        # NI with the alignment defect of the released code (training only)
        return ce_train_run("released", "--inject_noise", "--noise_p", CE[DATASET]["p"], "--legacy_alignment")
    need(stats_path("train_replacement_probs.pkl"), "setup")
    # exactly as released, to compare with the published "tuned gen. weights + NI" row
    train_run("ni_released", "--inject_noise", "--noise_p", P, "--legacy_alignment")


def stage_baseline():
    not_for_ce("baseline")
    stage_ni()
    stage_released()


def stage_control():
    not_for_ce("control")
    source = os.path.join(OUT, run_name("ni"))
    need(os.path.join(source, "checkpoints", "bb_model.pt"), "ni")
    # evaluation overwrites results.json, so it runs on a copy
    target = os.path.join(OUT, run_name("control_random"))
    shutil.rmtree(target, ignore_errors = True)
    shutil.copytree(source, target)
    python("run_words.py", "--evaluate", "--selection_method", "random", "--mask_special_tokens", *common(), *selection(),
           "--save_path", target)


def stage_judge():
    not_for_ce("judge")
    save_path = os.path.join(OUT, "full_text")
    if os.path.exists(os.path.join(save_path, "results.json")):
        print("== full_text is finished, skipping")
        return
    python("run_full_text.py", "--train", "--evaluate", *common(), "--save_path", save_path)


def stage_rescore():
    not_for_ce("rescore")
    need(judge_path(), "judge")
    # only valid for runs trained with the default selection options, as below
    for run_path in sorted(glob.glob(os.path.join(OUT, "*_seed*"))):
        if "control_random" in run_path or not os.path.exists(os.path.join(run_path, "checkpoints", "bb_model.pt")):
            continue
        print(f"== rescoring {os.path.basename(run_path)}")
        python("run_words.py", "--evaluate", *common(), *selection(), *training_options(run_path), "--faithfulness_model", judge_path(), "--save_path", run_path)


def stage_prep():
    not_for_ce("prep")
    need(judge_path(), "judge")
    if not os.path.exists(stats_path("train_mlm_candidates.npy")):
        python("build_mlm_replacements.py", "--data_path", data_path())
    if not os.path.exists(stats_path("train_replacement_probs_saliency.pkl")):
        python("build_saliency_probs.py", "--data_path", data_path(), "--classifier_path", os.path.join(OUT, "full_text"))


def stage_a2r():
    if is_ce():
        return ce_train_run("a2r")
    train_run("a2r")


def stage_exp():
    not_for_ce("exp")
    train_run("ani_exp", "--inject_noise", "--noise_schedule", "exponential", "--noise_p0", P0, "--noise_p", P, "--noise_gamma", GAMMA)


def stage_cos():
    not_for_ce("cos")
    train_run("ani_cos", "--inject_noise", "--noise_schedule", "cosine", "--noise_p0", P0, "--noise_p", P)


def stage_closed():
    not_for_ce("closed")
    train_run("ani_closed", "--inject_noise", "--noise_schedule", "closed_loop", "--ctrl_signal", "jsd", "--noise_p", P)


def stage_full():
    not_for_ce("full")
    need(stats_path("train_mlm_candidates.npy"), "prep")
    need(stats_path("train_replacement_probs_saliency.pkl"), "prep")
    train_run("full", "--inject_noise", "--noise_schedule", "closed_loop", "--ctrl_signal", "jsd", "--noise_p", P,
              "--noise_source", "mlm", "--replacement_probs", "saliency")


def stage_frozen():
    # the generator's encoder keeps its pretrained weights, only its two
    # linear layers are trained
    if is_ce():
        return ce_train_run("frozen", "--freeze_encoder_bb")
    train_run("frozen", "--freeze_encoder_bb")


def stage_frozen_ni():
    not_for_ce("frozen_ni")
    need(stats_path("train_replacement_probs.pkl"), "setup")
    train_run("frozen_ni", "--freeze_encoder_bb", "--inject_noise", "--noise_p", P)


def stage_frozen_en():
    not_for_ce("frozen_en")
    need(stats_path("train_replacement_probs.pkl"), "setup")
    train_run("frozen_en", "--freeze_encoder_bb", "--inject_noise", "--noise_target", "evidence", "--noise_p", EVIDENCE_P)


def stage_frozen_closed():
    not_for_ce("frozen_closed")
    need(stats_path("train_replacement_probs.pkl"), "setup")
    train_run("frozen_closed", "--freeze_encoder_bb", "--inject_noise", "--noise_target", "evidence",
              "--noise_schedule", "closed_loop", "--ctrl_signal", "exposure", "--ctrl_target", EXPOSURE,
              "--noise_p", P, "--noise_p_min", 0.0, "--noise_p_max", 1.0)


def stage_dr():
    # DR (Liu et al., 2023): the predictor learns more slowly than the generator
    not_for_ce("dr")
    train_run("dr", "--rp_lr", DR_RP_LR)


def stage_vicinal():
    # a random vicinity of the selection: PERTURBATION of the selected tokens
    # are swapped for other tokens of the review during training
    not_for_ce("vicinal")
    train_run("vicinal", "--selection_perturbation", PERTURBATION)


def stage_ni_coupled():
    # NI with a path from the predictor's loss to the generator (straight-through)
    not_for_ce("ni_coupled")
    need(stats_path("train_replacement_probs.pkl"), "setup")
    train_run("ni_coupled", "--inject_noise", "--noise_p", P, "--coupling_weight", COUPLING)


def stage_a2r_coupled():
    # the same path without noise
    not_for_ce("a2r_coupled")
    need(stats_path("train_replacement_probs.pkl"), "setup")
    train_run("a2r_coupled", "--inject_noise", "--noise_p", 0, "--coupling_weight", COUPLING)


def stage_frozen_x():
    # frozen generator that encodes the query with each sentence on its own
    if not is_ce():
        raise RuntimeError("Stage 'frozen_x' is for MultiRC/FEVER only")
    ce_train_run("frozen_x", "--generator", "pairwise", "--freeze_encoder_bb")


# Round 3, experiment 1: the joint generator attends to document tokens only
def ce_only(stage):
    if not is_ce():
        raise RuntimeError(f"Stage '{stage}' is for MultiRC/FEVER only")


def stage_a2r_dm():
    ce_only("a2r_dm")
    ce_train_run("a2r_dm", "--document_attention")


def stage_ni_dm():
    ce_only("ni_dm")
    ce_train_run("ni_dm", "--inject_noise", "--noise_p", CE[DATASET]["p"], "--document_attention")


def stage_released_dm():
    ce_only("released_dm")
    ce_train_run("released_dm", "--inject_noise", "--noise_p", CE[DATASET]["p"], "--legacy_alignment", "--document_attention")


def stage_frozen_dm():
    ce_only("frozen_dm")
    ce_train_run("frozen_dm", "--freeze_encoder_bb", "--document_attention")


# Round 3, experiment 2: product of experts with a query-only model
def query_only_path():
    return os.path.join(OUT, f"{DATASET}_query_only")


def stage_query_only():
    ce_only("query_only")
    if os.path.exists(os.path.join(query_only_path(), "query_probs_train.json")):
        print("== query_only is finished, skipping")
        return
    need(os.path.join(data_path(), "train.jsonl"), "setup")
    python("query_only.py", "--data_path", data_path(), "--save_path", query_only_path(), "--seed", QUERY_ONLY_SEED, *device_args())


def poe_options():
    need(os.path.join(query_only_path(), "query_probs_train.json"), "query_only")
    return ["--poe_probs", query_only_path()]


def stage_a2r_dm_poe():
    ce_only("a2r_dm_poe")
    ce_train_run("a2r_dm_poe", "--document_attention", *poe_options())


def stage_frozen_dm_poe():
    ce_only("frozen_dm_poe")
    ce_train_run("frozen_dm_poe", "--freeze_encoder_bb", "--document_attention", *poe_options())


def stage_frozen_x_poe():
    ce_only("frozen_x_poe")
    ce_train_run("frozen_x_poe", "--generator", "pairwise", "--freeze_encoder_bb", *poe_options())


def stage_smoke_r3():
    # each round-3 option on small subsets for one epoch (minutes); "smoke"
    # runs are ignored by probe and pack
    subsets = ["--train_subset", 400, "--valid_subset", 100, "--test_subset", 100, "--num_epochs", 1]
    if is_ce():
        need(os.path.join(data_path(), "word_statistics", "train_replacement_probs.pkl"), "setup")
        query_only = os.path.join(OUT, f"smoke_{DATASET}_query_only")
        python("query_only.py", "--data_path", data_path(), "--save_path", query_only, "--subset", 400, "--num_epochs", 1)
        for name, options in [("a2r_dm", ["--document_attention"]),
                              ("ni_dm", ["--inject_noise", "--noise_p", CE[DATASET]["p"], "--document_attention"]),
                              ("frozen_dm_poe", ["--freeze_encoder_bb", "--document_attention", "--poe_probs", query_only]),
                              ("frozen_x_poe", ["--generator", "pairwise", "--freeze_encoder_bb", "--poe_probs", query_only])]:
            print(f"== smoke {name}", flush = True)
            python("run_ce.py", "--train", "--evaluate", *ce_common(), *subsets, "--patience", 2, "--validations_per_epoch", 1,
                   *options, "--save_path", os.path.join(OUT, f"smoke_{DATASET}_{name}"))
        return
    need(stats_path("train_replacement_probs.pkl"), "setup")
    for name, options in [("a2r", []), ("ni", ["--inject_noise", "--noise_p", P]),
                          ("frozen_w3", ["--freeze_encoder_bb", "--selection_window", 3])]:
        print(f"== smoke {name}", flush = True)
        python("run_words.py", "--train", "--evaluate", *common(), *subsets, *options,
               "--save_path", os.path.join(OUT, f"smoke_{DATASET}_{name}"))


def stage_smoke_new():
    # each round-2 option on small subsets for one epoch (minutes), to catch
    # errors before the long runs; "smoke" runs are ignored by probe and pack
    subsets = ["--train_subset", 400, "--valid_subset", 100, "--test_subset", 100, "--num_epochs", 1]
    if is_ce():
        need(os.path.join(data_path(), "word_statistics", "train_replacement_probs.pkl"), "setup")
        python("run_ce.py", "--train", "--evaluate", *ce_common(), *subsets, "--patience", 2, "--validations_per_epoch", 1,
               "--generator", "pairwise", "--freeze_encoder_bb", "--save_path", os.path.join(OUT, f"smoke_{DATASET}_frozen_x"))
        return
    need(stats_path("train_replacement_probs.pkl"), "setup")
    for name, options in [("dr", ["--rp_lr", DR_RP_LR]), ("vicinal", ["--selection_perturbation", PERTURBATION]),
                          ("ni_coupled", ["--inject_noise", "--noise_p", P, "--coupling_weight", COUPLING]),
                          ("a2r_coupled", ["--inject_noise", "--noise_p", 0, "--coupling_weight", COUPLING])]:
        print(f"== smoke {name}", flush = True)
        python("run_words.py", "--train", "--evaluate", *common(), *selection(), *subsets, *options,
               "--save_path", os.path.join(OUT, f"smoke_{name}"))


def training_options(run_path):
    # options of the training run that evaluation has to repeat
    config_path = os.path.join(run_path, "config.json")
    if not os.path.exists(config_path):
        return []
    with open(config_path) as f:
        config = json.load(f)
    options = []
    for key in ["model", "max_length", "sparsity", "generator_layers"]:
        if config.get(key) is not None:
            options += [f"--{key}", config[key]]
    if config.get("mask_special_tokens"):
        options.append("--mask_special_tokens")
    return options


def find_runs():
    # trained runs of this session and of earlier sessions attached as input
    runs = {}
    here = sorted(glob.glob(os.path.join(OUT, "*", "checkpoints", "bb_model.pt")))
    attached = sorted(glob.glob(os.path.join(INPUTS, "**", "checkpoints", "bb_model.pt"), recursive = True))
    for bb_path in here + attached:
        run_path = os.path.dirname(os.path.dirname(bb_path))
        name = os.path.basename(run_path)
        if os.path.exists(os.path.join(run_path, "checkpoints", "rp_model.pt")) and "control_random" not in name and "smoke" not in name:
            runs.setdefault(name, run_path)
    return runs


def probe_table():
    # one line per run and setting
    rows = []
    for result_path in sorted(glob.glob(os.path.join(OUT, "probe", "*", "results.json"))):
        with open(result_path) as f:
            results = json.load(f)
        micro, selected = results["rationales"]["micro"], results["selection"]
        rows.append((os.path.basename(os.path.dirname(result_path)), 100 * results["accuracy"], 100 * micro["prec"], 100 * micro["rec"],
                     100 * micro["F1"], 100 * results["micro_iou"]["0.1"]["f1"], 100 * results["reference"]["random"]["F1"],
                     selected["trivial_word_share"], selected["mean_span_length"]))
    print(f"{'run__setting':40s} {'Acc':>5s} {'P':>5s} {'R':>5s} {'F1':>5s} {'IOU':>5s} {'floor':>5s} {'stopw':>5s} {'span':>5s}")
    for row in rows:
        print(f"{row[0]:40s} " + " ".join(f"{value:5.1f}" for value in row[1:7]) + f" {row[7]:5.2f} {row[8]:5.1f}")


def ce_generator_options(run_path):
    # the evaluation must build the generator the run was trained with
    config_path = os.path.join(run_path, "config.json")
    if not os.path.exists(config_path):
        return []
    with open(config_path) as f:
        config = json.load(f)
    if config.get("generator", "joint") == "pairwise":
        return ["--generator", "pairwise", "--pair_max_length", config.get("pair_max_length", 128)]
    if config.get("document_attention"):
        return ["--document_attention"]
    return []


def ce_probe():
    # random-selection control for every run of this dataset; needs the
    # checkpoints, so it runs in the session that trained the models
    runs = {name: path for name, path in find_runs().items() if name.startswith(DATASET + "_") and "smoke" not in name}
    if len(runs) == 0:
        raise RuntimeError(f"No trained {DATASET} runs in {OUT}")
    rows = []
    for name, run_path in sorted(runs.items()):
        probes = {"random": ["--random_selection"]}
        for probe, options in probes.items():
            target = os.path.join(OUT, "probe", f"{name}__{probe}")
            if not os.path.exists(os.path.join(target, "results.json")):
                os.makedirs(target, exist_ok = True)
                link = os.path.join(target, "checkpoints")
                if not os.path.lexists(link):
                    os.symlink(os.path.join(run_path, "checkpoints"), link)
                python("run_ce.py", "--evaluate", *ce_common(), *ce_generator_options(run_path), "--save_path", target, *options)
        for result_path in [os.path.join(run_path, "results.json")] + sorted(glob.glob(os.path.join(OUT, "probe", f"{name}__*", "results.json"))):
            if not os.path.exists(result_path):
                continue
            with open(result_path) as f:
                results = json.load(f)
            micro = results["rationales"]["micro"]
            rows.append((os.path.basename(os.path.dirname(result_path)), 100 * results["accuracy"], 100 * micro["prec"],
                         100 * micro["rec"], 100 * micro["F1"], 100 * results.get("majority_class_share", float("nan"))))
    print(f"{'run__setting':45s} {'Acc':>5s} {'P':>5s} {'R':>5s} {'F1':>5s} {'1cls%':>5s}")
    for row in rows:
        print(f"{row[0]:45s} " + " ".join(f"{value:5.1f}" for value in row[1:]))


def stage_probe():
    if is_ce():
        return ce_probe()
    if is_hx():
        runs = {name: path for name, path in find_runs().items() if name.startswith("hatexplain_")}
        probes = HX_PROBES
    else:
        runs = {name: path for name, path in find_runs().items() if not name.startswith(tuple(f"{dataset}_" for dataset in list(CE) + ["hatexplain"]))}
        probes = {probe: options for probe, options in PROBES.items() if USR_PROBES is None or probe in USR_PROBES}
    if len(runs) == 0:
        raise RuntimeError(f"No trained runs in {OUT} or under {INPUTS}. Attach the output of an earlier session with Add Input")
    print(f"Runs: {', '.join(sorted(runs))}")
    jobs = []
    for name, run_path in sorted(runs.items()):
        for probe, options in probes.items():
            target = os.path.join(OUT, "probe", f"{name}__{probe}")
            if os.path.exists(os.path.join(target, "results.json")):
                continue
            jobs.append((name, run_path, probe, options, target))
    os.makedirs(os.path.join(OUT, "logs"), exist_ok = True)
    errors = []
    lock = threading.Lock()

    def work(device):
        _parallel.device = device
        with open(os.path.join(OUT, "logs", f"probe_{device or 'default'}.log".replace(":", "")), "a") as log:
            _parallel.log = log
            while True:
                with lock:
                    if len(jobs) == 0:
                        return
                    name, run_path, probe, options, target = jobs.pop(0)
                try:
                    os.makedirs(target, exist_ok = True)
                    # the models stay where they are
                    link = os.path.join(target, "checkpoints")
                    if not os.path.lexists(link):
                        os.symlink(os.path.join(run_path, "checkpoints"), link)
                    python("run_words.py", "--evaluate", "--data_path", data_path(), "--batch_size", BATCH, "--seed", SEED,
                           *device_args(), *training_options(run_path), "--save_path", target, *options)
                    print(f"== {name}__{probe} done", flush = True)
                except Exception as error:
                    errors.append(f"{name}__{probe}")
                    print(f"== {name}__{probe} FAILED: {error}", flush = True)

    total = len(jobs)
    print(f"{total} evaluations to run")
    threads = [threading.Thread(target = work, args = (device,)) for device in (get_devices() or [None])]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    probe_table()
    if errors:
        raise RuntimeError(f"Failed: {', '.join(errors)}")


def stage_table():
    not_for_ce("table")
    # every group of finished runs that differ in their seed only
    groups = {}
    for result_path in sorted(glob.glob(os.path.join(OUT, "*_seed*", "results.json"))):
        name = os.path.basename(os.path.dirname(result_path))
        groups.setdefault(name[:name.rindex("_seed")], os.path.join(OUT, name[:name.rindex("_seed")] + "_seed*"))
    if len(groups) == 0:
        raise RuntimeError(f"No finished runs in {OUT}")
    # significance tests need the fixed NI runs as the baseline
    baseline = ["--baseline", "ni"] if "ni" in groups else []
    python("aggregate_results.py", *baseline, "--runs", *[f"{name}={pattern}" for name, pattern in groups.items()])


def stage_pack():
    archive_path = os.path.join(os.path.dirname(OUT), "results.tar.gz")
    with tarfile.open(archive_path, "w:gz") as archive:
        for name in ["results.json", "per_example.json", "config.json", "metrics.json", "attention.npz"]:
            for path in sorted(glob.glob(os.path.join(OUT, "**", name), recursive = True)):
                archive.add(path, arcname = os.path.relpath(path, OUT))
    print(f"Results saved to ==> {archive_path}")


def get_devices():
    if DEVICES is not None:
        return list(DEVICES)
    if not shutil.which("nvidia-smi"):
        return []
    listing = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output = True, text = True)
    return [f"cuda:{index}" for index in listing.stdout.split()]


def log_lines(log_path):
    # progress bars rewrite their line with \r, keep every version as a line
    with open(log_path, "r", errors = "replace") as log:
        return [line.strip() for line in log.read().replace("\r", "\n").split("\n") if line.strip()]


def run_parallel(*stages):
    # runs every stage at the same time, each on its own GPU
    not_parallel = [stage for stage in stages if split_stage(stage)[0] not in PARALLEL_STAGES]
    if not_parallel:
        raise ValueError(f"Cannot run in parallel: {', '.join(not_parallel)}. Parallel stages: {', '.join(PARALLEL_STAGES)}")
    if len(set(stages)) != len(stages):
        raise ValueError("A stage can only be given once")
    devices = get_devices()
    if len(stages) > len(devices):
        raise RuntimeError(f"{len(stages)} stages need {len(stages)} GPUs, found {len(devices)}")
    clone()
    os.makedirs(os.path.join(OUT, "logs"), exist_ok = True)
    errors = {}

    def work(stage, device, log_path):
        _parallel.device = device
        name, _parallel.window = split_stage(stage)
        with open(log_path, "a") as log:
            _parallel.log = log
            try:
                globals()[f"stage_{name}"]()
            except Exception as error:
                errors[stage] = error
                print(f"FAILED: {error}", file = log, flush = True)

    jobs = []
    for stage, device in zip(stages, devices):
        log_path = os.path.join(OUT, "logs", (f"{DATASET}_" if is_ce() or is_hx() else "") + f"{stage.replace('@', '_')}_seed{SEED}.log")
        print(f"==== {stage} on {device}, log: {log_path}", flush = True)
        thread = threading.Thread(target = work, args = (stage, device, log_path))
        thread.start()
        jobs.append((stage, thread, log_path))
        # do not download the pretrained model twice at the same moment
        if len(jobs) < len(stages):
            thread.join(timeout = 20)

    start = time.time()
    while any(thread.is_alive() for _, thread, _ in jobs):
        deadline = time.time() + STATUS_EVERY
        while time.time() < deadline and any(thread.is_alive() for _, thread, _ in jobs):
            time.sleep(1)
        for stage, thread, log_path in jobs:
            if thread.is_alive():
                lines = log_lines(log_path)
                print(f"[{(time.time() - start)/60:5.0f} min] {stage}: {lines[-1][-150:] if lines else 'starting'}", flush = True)

    for stage, _, log_path in jobs:
        print(f"==== {stage}: {'FAILED' if stage in errors else 'done'}, end of {log_path}", flush = True)
        # without the progress bars
        lines = [line for line in log_lines(log_path) if "it/s]" not in line and "s/it]" not in line]
        print("\n".join(lines[-80:]), flush = True)
    if errors:
        raise RuntimeError(f"Failed: {', '.join(errors)}")


def run(*stages):
    # a list of stages inside the list is run in parallel
    unknown = [stage for item in stages for stage in (item if isinstance(item, (list, tuple)) else [item])
               if f"stage_{split_stage(stage)[0]}" not in globals()]
    if unknown:
        raise ValueError(f"Unknown stage: {', '.join(unknown)}")
    clone()
    for stage in stages:
        if isinstance(stage, (list, tuple)):
            run_parallel(*stage)
        else:
            print(f"==== {stage}", flush = True)
            name, _parallel.window = split_stage(stage)
            try:
                globals()[f"stage_{name}"]()
            finally:
                _parallel.window = None


if __name__ == "__main__":
    # as a script: python kaggle_run.py setup smoke
    # in a notebook cell the arguments belong to the kernel and are ignored
    arguments = [] if "ipykernel" in sys.modules else sys.argv[1:]
    run(*(arguments or STAGES))
