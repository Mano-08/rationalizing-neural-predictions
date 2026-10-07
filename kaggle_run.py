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

REPO_URL = "https://github.com/Mano-08/rationalizing-neural-predictions.git"
BRANCH = "ani"
REPO = "/kaggle/working/noise_injection"
OUT = "/kaggle/working/runs"

# Parallel runs: devices to use (None = every GPU found) and seconds between
# status lines
DEVICES = None
STATUS_EVERY = 300

# stages that are one independent training run and may run in parallel
PARALLEL_STAGES = ["ni", "released", "a2r", "exp", "cos", "closed", "full", "judge", "frozen", "frozen_ni", "frozen_en", "frozen_closed"]

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


def data_path():
    return os.path.join(REPO, "usr_movie_review")


def stats_path(name):
    return os.path.join(data_path(), "word_statistics", name)


def judge_path():
    return os.path.join(OUT, "full_text", "checkpoints", "full_text.pt")


def device_args():
    device = getattr(_parallel, "device", None)
    return ["--device", device] if device else []


def common():
    truncation = ["--truncation_side", "left"] if TRUNCATION == "left" else []
    return ["--data_path", data_path(), "--batch_size", BATCH, "--seed", SEED] + truncation + device_args()


def selection():
    # options of run_words.py only
    return ["--selection_window", WINDOW] if WINDOW > 1 else []


def run_name(name):
    return name + ("_tail" if TRUNCATION == "left" else "") + (f"_w{WINDOW}" if WINDOW > 1 else "") + f"_seed{SEED}"


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
    python("run_words.py", "--train", "--evaluate", *common(), *selection(), *judge, "--save_path", save_path, *args)


def stage_setup():
    sh(sys.executable, "-m", "pip", "install", "-q", "termcolor")
    if shutil.which("nvidia-smi"):
        sh("nvidia-smi")
    else:
        print("No GPU found - switch the accelerator on")
    if not os.path.exists(stats_path("train_replacement_probs.pkl")):
        sh(sys.executable, "generate_word_statistics.py", f"--data_path={data_path()}",
           cwd = os.path.join(REPO, "data_preprocessing", "usr_movie_review"))


def stage_smoke():
    need(stats_path("train_replacement_probs.pkl"), "setup")
    python("run_words.py", "--train", "--evaluate", "--inject_noise", "--noise_p", P, *common(), *selection(),
           "--train_subset", 800, "--valid_subset", 200, "--test_subset", 200, "--num_epochs", 1,
           "--save_path", os.path.join(OUT, "smoke"))


def stage_ni():
    need(stats_path("train_replacement_probs.pkl"), "setup")
    train_run("ni", "--inject_noise", "--noise_p", P)


def stage_released():
    need(stats_path("train_replacement_probs.pkl"), "setup")
    # exactly as released, to compare with the published "tuned gen. weights + NI" row
    train_run("ni_released", "--inject_noise", "--noise_p", P, "--legacy_alignment")


def stage_baseline():
    stage_ni()
    stage_released()


def stage_control():
    source = os.path.join(OUT, run_name("ni"))
    need(os.path.join(source, "checkpoints", "bb_model.pt"), "ni")
    # evaluation overwrites results.json, so it runs on a copy
    target = os.path.join(OUT, run_name("control_random"))
    shutil.rmtree(target, ignore_errors = True)
    shutil.copytree(source, target)
    python("run_words.py", "--evaluate", "--selection_method", "random", "--mask_special_tokens", *common(), *selection(),
           "--save_path", target)


def stage_judge():
    save_path = os.path.join(OUT, "full_text")
    if os.path.exists(os.path.join(save_path, "results.json")):
        print("== full_text is finished, skipping")
        return
    python("run_full_text.py", "--train", "--evaluate", *common(), "--save_path", save_path)


def stage_rescore():
    need(judge_path(), "judge")
    # only valid for runs trained with the default selection options, as below
    for run_path in sorted(glob.glob(os.path.join(OUT, "*_seed*"))):
        if "control_random" in run_path or not os.path.exists(os.path.join(run_path, "checkpoints", "bb_model.pt")):
            continue
        print(f"== rescoring {os.path.basename(run_path)}")
        python("run_words.py", "--evaluate", *common(), *selection(), *training_options(run_path), "--faithfulness_model", judge_path(), "--save_path", run_path)


def stage_prep():
    need(judge_path(), "judge")
    if not os.path.exists(stats_path("train_mlm_candidates.npy")):
        python("build_mlm_replacements.py", "--data_path", data_path())
    if not os.path.exists(stats_path("train_replacement_probs_saliency.pkl")):
        python("build_saliency_probs.py", "--data_path", data_path(), "--classifier_path", os.path.join(OUT, "full_text"))


def stage_a2r():
    train_run("a2r")


def stage_exp():
    train_run("ani_exp", "--inject_noise", "--noise_schedule", "exponential", "--noise_p0", P0, "--noise_p", P, "--noise_gamma", GAMMA)


def stage_cos():
    train_run("ani_cos", "--inject_noise", "--noise_schedule", "cosine", "--noise_p0", P0, "--noise_p", P)


def stage_closed():
    train_run("ani_closed", "--inject_noise", "--noise_schedule", "closed_loop", "--ctrl_signal", "jsd", "--noise_p", P)


def stage_full():
    need(stats_path("train_mlm_candidates.npy"), "prep")
    need(stats_path("train_replacement_probs_saliency.pkl"), "prep")
    train_run("full", "--inject_noise", "--noise_schedule", "closed_loop", "--ctrl_signal", "jsd", "--noise_p", P,
              "--noise_source", "mlm", "--replacement_probs", "saliency")


def stage_frozen():
    # the generator's encoder keeps its pretrained weights, only its two
    # linear layers are trained
    train_run("frozen", "--freeze_encoder_bb")


def stage_frozen_ni():
    need(stats_path("train_replacement_probs.pkl"), "setup")
    train_run("frozen_ni", "--freeze_encoder_bb", "--inject_noise", "--noise_p", P)


def stage_frozen_en():
    need(stats_path("train_replacement_probs.pkl"), "setup")
    train_run("frozen_en", "--freeze_encoder_bb", "--inject_noise", "--noise_target", "evidence", "--noise_p", EVIDENCE_P)


def stage_frozen_closed():
    need(stats_path("train_replacement_probs.pkl"), "setup")
    train_run("frozen_closed", "--freeze_encoder_bb", "--inject_noise", "--noise_target", "evidence",
              "--noise_schedule", "closed_loop", "--ctrl_signal", "exposure", "--ctrl_target", EXPOSURE,
              "--noise_p", P, "--noise_p_min", 0.0, "--noise_p_max", 1.0)


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
        if os.path.exists(os.path.join(run_path, "checkpoints", "rp_model.pt")) and "control_random" not in name and name != "smoke":
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


def stage_probe():
    runs = find_runs()
    if len(runs) == 0:
        raise RuntimeError(f"No trained runs in {OUT} or under {INPUTS}. Attach the output of an earlier session with Add Input")
    print(f"Runs: {', '.join(sorted(runs))}")
    jobs = []
    for name, run_path in sorted(runs.items()):
        for probe, options in PROBES.items():
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
    not_parallel = [stage for stage in stages if stage not in PARALLEL_STAGES]
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
        with open(log_path, "a") as log:
            _parallel.log = log
            try:
                globals()[f"stage_{stage}"]()
            except Exception as error:
                errors[stage] = error
                print(f"FAILED: {error}", file = log, flush = True)

    jobs = []
    for stage, device in zip(stages, devices):
        log_path = os.path.join(OUT, "logs", f"{stage}_seed{SEED}.log")
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
    unknown = [stage for item in stages for stage in (item if isinstance(item, (list, tuple)) else [item]) if f"stage_{stage}" not in globals()]
    if unknown:
        raise ValueError(f"Unknown stage: {', '.join(unknown)}")
    clone()
    for stage in stages:
        if isinstance(stage, (list, tuple)):
            run_parallel(*stage)
        else:
            print(f"==== {stage}", flush = True)
            globals()[f"stage_{stage}"]()


if __name__ == "__main__":
    # as a script: python kaggle_run.py setup smoke
    # in a notebook cell the arguments belong to the kernel and are ignored
    arguments = [] if "ipykernel" in sys.modules else sys.argv[1:]
    run(*(arguments or STAGES))
