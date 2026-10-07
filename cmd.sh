#!/usr/bin/env bash
# Runs the experiments on Kaggle, one stage at a time.
#
# Notebook settings: Accelerator = GPU, Internet = On. Then, in notebook cells:
#
#   !git clone -b ani https://github.com/Mano-08/rationalizing-neural-predictions.git /kaggle/working/noise_injection
#   !bash /kaggle/working/noise_injection/cmd.sh setup
#   !bash /kaggle/working/noise_injection/cmd.sh smoke
#   !bash /kaggle/working/noise_injection/cmd.sh baseline
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
#   table     mean (std) over seeds and significance tests
#   pack      collect result files into results.tar.gz
#
# Several stages can be given at once: cmd.sh setup smoke
# A finished run (results.json exists) is skipped, so a stage can be re-run
# after a session was cut off.
#
# Settings, e.g.  !SEED=2 bash cmd.sh baseline
#   SEED   random seed, also part of the run name       (default 1)
#   BATCH  batch size; the paper uses 16, which needs
#          more than the 16 GB of a Kaggle GPU           (default 8)
#   P      final / fixed noise level                     (default 0.2)
#   P0     initial noise level of decaying schedules     (default 0.5)
#   GAMMA  exponential decay rate per epoch              (default 1.0)
#   OUT    where runs are saved                          (default /kaggle/working/runs)
#   DATA   preprocessed dataset                          (default usr_movie_review)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
SEED="${SEED:-1}"
BATCH="${BATCH:-8}"
P="${P:-0.2}"
P0="${P0:-0.5}"
GAMMA="${GAMMA:-1.0}"
DATA="${DATA:-$REPO/usr_movie_review}"
if [ -d /kaggle/working ]; then
    OUT="${OUT:-/kaggle/working/runs}"
else
    OUT="${OUT:-$REPO/runs}"
fi
JUDGE="$OUT/full_text/checkpoints/full_text.pt"
COMMON=(--data_path "$DATA" --batch_size "$BATCH" --seed "$SEED")

cd "$REPO/model"

# Adds the faithfulness judge to an evaluation once it has been trained
judge_args() {
    if [ -f "$JUDGE" ]; then
        echo --faithfulness_model "$JUDGE"
    fi
}

# train_run NAME [run_words.py options]: one full training + evaluation
train_run() {
    local name="$1_seed$SEED"
    shift
    if [ -f "$OUT/$name/results.json" ]; then
        echo "== $name is finished, skipping"
        return
    fi
    echo "== $name"
    "$PY" run_words.py --train --evaluate "${COMMON[@]}" $(judge_args) --save_path "$OUT/$name" "$@"
}

need() {
    if [ ! -e "$1" ]; then
        echo "Missing $1 - run the '$2' stage first" >&2
        exit 1
    fi
}

stage_setup() {
    "$PY" -m pip install -q termcolor
    nvidia-smi || echo "No GPU found - switch the accelerator on"
    if [ ! -f "$DATA/word_statistics/train_replacement_probs.pkl" ]; then
        (cd "$REPO/data_preprocessing/usr_movie_review" && "$PY" generate_word_statistics.py --data_path="$DATA")
    fi
}

stage_smoke() {
    need "$DATA/word_statistics/train_replacement_probs.pkl" setup
    "$PY" run_words.py --train --evaluate --inject_noise --noise_p "$P" "${COMMON[@]}" \
        --train_subset 800 --valid_subset 200 --test_subset 200 --num_epochs 1 --save_path "$OUT/smoke"
}

stage_ni() {
    need "$DATA/word_statistics/train_replacement_probs.pkl" setup
    train_run ni --inject_noise --noise_p "$P"
}

stage_released() {
    need "$DATA/word_statistics/train_replacement_probs.pkl" setup
    # exactly as released, to compare with the published "tuned gen. weights + NI" row
    train_run ni_released --inject_noise --noise_p "$P" --legacy_alignment
}

stage_baseline() {
    stage_ni
    stage_released
}

stage_control() {
    need "$OUT/ni_seed$SEED/checkpoints/bb_model.pt" ni
    # evaluation overwrites results.json, so it runs on a copy
    rm -rf "$OUT/control_random_seed$SEED"
    cp -r "$OUT/ni_seed$SEED" "$OUT/control_random_seed$SEED"
    "$PY" run_words.py --evaluate --selection_method random --mask_special_tokens "${COMMON[@]}" \
        --save_path "$OUT/control_random_seed$SEED"
}

stage_judge() {
    if [ -f "$OUT/full_text/results.json" ]; then
        echo "== full_text is finished, skipping"
        return
    fi
    "$PY" run_full_text.py --train --evaluate "${COMMON[@]}" --save_path "$OUT/full_text"
}

stage_rescore() {
    need "$JUDGE" judge
    # only valid for runs trained with the default selection options, as below
    for run in "$OUT"/*_seed*; do
        case "$run" in *control_random*) continue ;; esac
        [ -f "$run/checkpoints/bb_model.pt" ] || continue
        echo "== rescoring $(basename "$run")"
        "$PY" run_words.py --evaluate "${COMMON[@]}" --faithfulness_model "$JUDGE" --save_path "$run"
    done
}

stage_prep() {
    need "$JUDGE" judge
    if [ ! -f "$DATA/word_statistics/train_mlm_candidates.npy" ]; then
        "$PY" build_mlm_replacements.py --data_path "$DATA"
    fi
    if [ ! -f "$DATA/word_statistics/train_replacement_probs_saliency.pkl" ]; then
        "$PY" build_saliency_probs.py --data_path "$DATA" --classifier_path "$OUT/full_text"
    fi
}

stage_a2r() {
    train_run a2r
}

stage_exp() {
    train_run ani_exp --inject_noise --noise_schedule exponential --noise_p0 "$P0" --noise_p "$P" --noise_gamma "$GAMMA"
}

stage_cos() {
    train_run ani_cos --inject_noise --noise_schedule cosine --noise_p0 "$P0" --noise_p "$P"
}

stage_closed() {
    train_run ani_closed --inject_noise --noise_schedule closed_loop --ctrl_signal jsd --noise_p "$P"
}

stage_full() {
    need "$DATA/word_statistics/train_mlm_candidates.npy" prep
    need "$DATA/word_statistics/train_replacement_probs_saliency.pkl" prep
    train_run full --inject_noise --noise_schedule closed_loop --ctrl_signal jsd --noise_p "$P" \
        --noise_source mlm --replacement_probs saliency
}

stage_table() {
    local runs=()
    for name in ni_released ni a2r ani_exp ani_cos ani_closed full control_random; do
        if ls "$OUT"/"$name"_seed*/results.json > /dev/null 2>&1; then
            runs+=("$name=$OUT/${name}_seed*")
        fi
    done
    if [ ${#runs[@]} -eq 0 ]; then
        echo "No finished runs in $OUT" >&2
        exit 1
    fi
    # significance tests need the fixed NI runs as the baseline
    if ls "$OUT"/ni_seed*/results.json > /dev/null 2>&1; then
        "$PY" aggregate_results.py --baseline ni --runs "${runs[@]}"
    else
        "$PY" aggregate_results.py --runs "${runs[@]}"
    fi
}

stage_pack() {
    (cd "$OUT" && find . \( -name results.json -o -name per_example.json -o -name config.json -o -name metrics.json \) -print0 \
        | tar -czf "$(dirname "$OUT")/results.tar.gz" --null -T -)
    echo "Results saved to ==> $(dirname "$OUT")/results.tar.gz"
}

if [ $# -eq 0 ]; then
    # print the header of this file
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$REPO/cmd.sh"
    exit 1
fi

for stage in "$@"; do
    if ! declare -F "stage_$stage" > /dev/null; then
        echo "Unknown stage: $stage" >&2
        exit 1
    fi
    "stage_$stage"
done
