import dataclasses
import json
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report
from termcolor import colored
from tqdm.auto import tqdm
from transformers import AutoTokenizer, logging

from metrics import (normalized_faithfulness, plausibility, reference_points,
                     safe_div, selection_counts, selection_report,
                     to_gold_words)
from models import (BlackBoxPredictor, FullTextClassifier, RationaleExtractor,
                    RationaleExtractorFactory, RationalePredictor,
                    SelectorFactory, get_replacement_probs, get_selectable)
from movies import REPLACEMENT_PROBS, DataLoaderFactory
from noise import (SCHEDULES, SIGNALS, ConstantSchedule, create_noise_schedule,
                   js_div_per_example, normalized_attention_entropy)

logging.set_verbosity_error()

def parse_args():
    parser = ArgumentParser()
    # Whether to train and/or evaluate
    parser.add_argument("--train", action = "store_true")
    parser.add_argument("--evaluate", action = "store_true")
    # Whether to inject noise
    parser.add_argument("--inject_noise", action = "store_true")
    # Magnitude of augmentation hyperparameter (final value for decaying
    # schedules, initial value for the closed-loop controller)
    parser.add_argument('--noise_p', type = float, default = 0.1)
    # How the magnitude changes during training
    parser.add_argument('--noise_schedule', choices = SCHEDULES, default = 'constant')
    # Open-loop schedules: initial magnitude and exponential decay rate per epoch
    parser.add_argument('--noise_p0', type = float, default = None)
    parser.add_argument('--noise_gamma', type = float, default = 1.0)
    # Closed-loop controller: range of the magnitude, degeneracy signal, its
    # target (default 0.05 for jsd, 0.8 for entropy), gains and smoothing
    parser.add_argument('--noise_p_min', type = float, default = 0.05)
    parser.add_argument('--noise_p_max', type = float, default = 0.5)
    parser.add_argument('--ctrl_signal', choices = SIGNALS, default = 'jsd')
    parser.add_argument('--ctrl_target', type = float, default = None)
    parser.add_argument('--ctrl_kp', type = float, default = 1.0)
    parser.add_argument('--ctrl_ki', type = float, default = 0.001)
    parser.add_argument('--ctrl_ema', type = float, default = 0.99)
    parser.add_argument('--ctrl_probe_every', type = int, default = 10)
    # Where the noise goes: into the rationale the predictor reads (as
    # released), into the evidence the generator's own predictor pools, or both
    parser.add_argument('--noise_target', choices = ['rationale', 'evidence', 'both'], default = 'rationale')
    # What to replace tokens with: words sampled from the vocabulary or
    # in-context substitutes precomputed by build_mlm_replacements.py
    parser.add_argument('--noise_source', choices = ['vocab', 'mlm'], default = 'vocab')
    # Which tokens to replace: TF*IDF or build_saliency_probs.py
    parser.add_argument('--replacement_probs', choices = list(REPLACEMENT_PROBS), default = 'tfidf')
    # Reproduce the released noise injection, which trains the predictor on
    # the token before every selected token (off-by-one from the CLS token)
    parser.add_argument('--legacy_alignment', action = "store_true")
    # Let the loss of the rationale predictor reach the generator (straight-
    # through). 0 = as released: the generator never learns from the
    # predictor which of its selected words were useful
    parser.add_argument('--coupling_weight', type = float, default = 0.0)
    # Use only the first layers of the generator's encoder
    parser.add_argument('--generator_layers', type = int, default = None)
    # Rank tokens by attention averaged over this many tokens (odd number)
    parser.add_argument('--selection_window', type = int, default = 1)
    # Which end of a text longer than max_length to cut off: right keeps the
    # beginning, left keeps the end
    parser.add_argument('--truncation_side', choices = ['right', 'left'], default = 'right')
    # Random seed
    parser.add_argument('--seed', type = int, default = None)
    # Device
    parser.add_argument('--device', type = str, default = 'cuda')
    # Optimizer BB: pytorch optim.Adam defaults
    parser.add_argument('--bb_lr', type = float, default = 2e-5)
    # Optimizer RP pytorch optim.Adam defaults
    parser.add_argument('--rp_lr', type = float, default = 2e-5)
    # Freeze BERT weights
    parser.add_argument('--freeze_encoder_bb', action = "store_true")
    parser.add_argument('--freeze_encoder_rp', action = "store_true")
    # Training
    parser.add_argument('--num_epochs', type = int, default = 5)
    # Patience
    parser.add_argument('--patience', type = int, default = 2)
    # Model proximity hyperparameter
    parser.add_argument('--proximity', type = float, default = 0.1)
    # Model
    parser.add_argument('--save_path', type = str, default = os.path.join("trained", "ours"))
    parser.add_argument('--model', type = str, default = 'bert-base-uncased')
    parser.add_argument('--max_length', type = int, default = 512)
    parser.add_argument('--batch_size', type = int, default = 16)
    # Rationale Extraction hyperparameter
    parser.add_argument('--sparsity', type = float, default = 0.2)
    # Dataset
    parser.add_argument('--data_path', type = str, default = os.path.join("..", "..", "rnp_movie_review", "original"))
    # Selection method (random = control selecting tokens uniformly at random)
    parser.add_argument('--selection_method', choices = ['words', 'span', 'random'], default = 'words')
    # Keep SEP and PAD tokens out of the attention and of the rationale
    parser.add_argument('--mask_special_tokens', action = "store_true")
    # Use the first documents of a split only (fast debugging loop)
    parser.add_argument('--train_subset', type = int, default = None)
    parser.add_argument('--valid_subset', type = int, default = None)
    parser.add_argument('--test_subset', type = int, default = None)
    # Eval-related
    # Compare model-generated and hand-labeled rationales
    parser.add_argument('--show_detail', action = "store_true")
    # Plausibility of the top-k tokens at other selection rates
    parser.add_argument('--eval_sparsities', type = float, nargs = '*', default = [0.1, 0.2, 0.3])
    # Checkpoint of an independent full-text classifier (run_full_text.py)
    # used to measure faithfulness, and selection rates of its AOPC curves
    parser.add_argument('--faithfulness_model', type = str, default = None)
    parser.add_argument('--aopc_bins', type = float, nargs = '*', default = [0.01, 0.05, 0.1, 0.2, 0.5])
    # Save the attention of every test token to attention.npz
    parser.add_argument('--save_attention', action = "store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    if not args.train and not args.evaluate:
        print("Must append flag --train or --evaluate")
        return

    checkpoint_dir = os.path.join(args.save_path, "checkpoints")

    if args.seed is not None:
        set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast = True)
    tokenizer.truncation_side = args.truncation_side

    bb_model = BlackBoxPredictor(num_labels = 2, model = args.model, freeze_encoder = args.freeze_encoder_bb, num_layers = args.generator_layers).to(args.device)
    print(f"Black Box Predictor: {get_num_params(bb_model)} parameters")

    rp_model = RationalePredictor(num_labels = 2, model = args.model, freeze_encoder = args.freeze_encoder_rp).to(args.device)
    print(f"Rationale Predictor: {get_num_params(rp_model)} parameters")

    rationale_selector = SelectorFactory(args.sparsity, args.max_length, tokenizer.pad_token_id, args.device, args.seed, args.selection_window).create_selector(args.selection_method)

    if args.mask_special_tokens:
        selectable_fn = lambda reviews_tokenized: get_selectable(reviews_tokenized, tokenizer)
    else:
        selectable_fn = lambda reviews_tokenized: None

    if args.train:
        if args.coupling_weight > 0 and not (args.inject_noise and args.noise_target != 'evidence'):
            raise ValueError('--coupling_weight requires --inject_noise with noise on the rationale (use --noise_p 0 to couple without noise)')
        os.makedirs(checkpoint_dir, exist_ok = True)
        with open(os.path.join(args.save_path, "config.json"), "w") as f:
            json.dump(vars(args), f, indent = 2)

        rationale_extractor = RationaleExtractorFactory(tokenizer, args.device, args.data_path, args.seed, args.legacy_alignment).create_extractor(args.inject_noise and args.noise_target != 'evidence', args.noise_source)

        train_loader = DataLoaderFactory(
            data_path = args.data_path,
            batch_size = args.batch_size,
            tokenizer = tokenizer,
            max_length = args.max_length,
            shuffle = True,
            replacement_probs = args.replacement_probs,
            subset = args.train_subset,
            seed = args.seed
        ).create_dataloader("train", args.inject_noise)
        # validation is always noise-free, for every training configuration
        valid_loader = DataLoaderFactory(
            data_path = args.data_path,
            batch_size = args.batch_size,
            tokenizer = tokenizer,
            max_length = args.max_length,
            shuffle = False,
            subset = args.valid_subset
        ).create_dataloader("valid", False)

        bb_optimizer = optim.Adam(bb_model.parameters(), args.bb_lr)
        rp_optimizer = optim.Adam(rp_model.parameters(), args.rp_lr)

        validation_rationale_extractor = RationaleExtractor(tokenizer, args.device)

        if args.inject_noise:
            if args.ctrl_target is None:
                args.ctrl_target = {'entropy': 0.8, 'exposure': 0.8}.get(args.ctrl_signal, 0.05)
            noise_schedule = create_noise_schedule(
                args = args,
                steps_per_epoch = len(train_loader),
                total_steps = args.num_epochs * len(train_loader)
            )
        else:
            noise_schedule = ConstantSchedule(p = 0.0)

        train(
            bb_model = bb_model,
            bb_optimizer = bb_optimizer,
            rp_model = rp_model,
            rp_optimizer = rp_optimizer,
            train_loader = train_loader,
            valid_loader = valid_loader,
            eval_every = len(train_loader),
            device = args.device,
            proximity = args.proximity,
            num_epochs = args.num_epochs,
            patience = args.patience,
            checkpoint_dir = checkpoint_dir,
            rationale_selector = rationale_selector,
            rationale_extractor = rationale_extractor,
            validation_rationale_extractor = validation_rationale_extractor,
            noise_schedule = noise_schedule,
            selectable_fn = selectable_fn,
            coupling_weight = args.coupling_weight,
            evidence_noise = args.inject_noise and args.noise_target != 'rationale',
        )

    if args.evaluate:
        test_loader = DataLoaderFactory(
            data_path = args.data_path,
            batch_size = args.batch_size,
            tokenizer = tokenizer,
            max_length = args.max_length,
            shuffle = False,
            subset = args.test_subset
        ).create_dataloader("test", False)

        test_rationale_extractor = RationaleExtractor(tokenizer, args.device)

        judge = None
        if args.faithfulness_model is not None:
            judge = FullTextClassifier(num_labels = 2, model = args.model).to(args.device)
            model_load(judge, args.faithfulness_model)

        evaluate(
            bb_model = bb_model,
            rp_model = rp_model,
            tokenizer = tokenizer,
            test_loader = test_loader,
            device = args.device,
            show_detail = args.show_detail,
            rationale_selector = rationale_selector,
            rationale_extractor = test_rationale_extractor,
            checkpoint_dir = checkpoint_dir,
            result_path = args.save_path,
            selectable_fn = selectable_fn,
            eval_sparsities = args.eval_sparsities,
            judge = judge,
            aopc_bins = args.aopc_bins,
            save_attention = args.save_attention
        )

def train(
    bb_model,
    bb_optimizer,
    rp_model,
    rp_optimizer,
    train_loader,
    valid_loader,
    eval_every,
    device,
    proximity,
    num_epochs,
    patience,
    checkpoint_dir,
    rationale_selector,
    rationale_extractor,
    validation_rationale_extractor,
    noise_schedule,
    selectable_fn,
    coupling_weight = 0.0,
    evidence_noise = False,
    ):

    with tqdm(total=num_epochs * len(train_loader)) as pb:

        # Initialize statistics
        bb_running_train_loss = 0.0
        bb_best_train_loss = float("Inf")
        rp_running_train_loss = 0.0
        rp_best_train_loss = float("Inf")
        bb_running_valid_loss = 0.0
        bb_best_valid_loss = float("Inf")
        rp_running_valid_loss = 0.0
        rp_best_valid_loss = float("Inf")
        running_train_replace_ratio = 0.0
        running_valid_replace_ratio = 0.0
        running_noise_p = 0.0
        running_exposure = 0.0
        global_step = 0
        metrics = []
        patience_left = patience

        # training loop
        bb_model.train()
        rp_model.train()
        for epoch in range(num_epochs):
            for batch in train_loader:

                if patience_left == 0:
                    pb.write("Patience is 0, early stopping")
                    break

                # generate prediction and token probs of being in a rationale
                batch.reviews_tokenized = batch.reviews_tokenized.to(device)
                selectable = selectable_fn(batch.reviews_tokenized)

                # set the noise level of this step
                noise_p = noise_schedule.value(global_step)
                rationale_extractor.noise_p = noise_p

                if evidence_noise:
                    replacement_probs = get_replacement_probs(batch, device)
                    att_pred, token_att = bb_model(
                        **batch.reviews_tokenized,
                        selectable = selectable,
                        evidence_noise = torch.where(replacement_probs >= 0, (replacement_probs * noise_p).clamp(max = 1), replacement_probs)
                    )
                else:
                    att_pred, token_att = bb_model(**batch.reviews_tokenized, selectable = selectable)

                hard_mask = rationale_selector(
                    token_att = token_att,
                    input_ids = batch.reviews_tokenized.input_ids,
                    selectable = selectable
                )

                rationale, _, replace_ratio = rationale_extractor(
                    batch = batch,
                    hard_mask = hard_mask
                )

                # predict from rationale
                if coupling_weight > 0:
                    gate = rationale_extractor.get_gate(batch, token_att, rationale, coupling_weight)
                    hard_pred = rp_model(**rationale, gate = gate)
                else:
                    hard_pred = rp_model(**rationale)

                # how much noise injection targets what the generator picks:
                # its attention if the noise is in its evidence, else its rationale
                if evidence_noise:
                    exposure = (token_att.detach().squeeze(-1) * replacement_probs.clamp(min = 0)).sum(-1).mean().item()
                else:
                    exposure = getattr(rationale_extractor, "exposure", 1.0)

                # measure the degeneracy signal of the closed-loop controller,
                # the new noise level applies from the next step
                if noise_schedule.closed_loop:
                    if noise_schedule.signal == 'entropy':
                        noise_schedule.update(normalized_attention_entropy(token_att.detach(), selectable).mean().item())
                    elif noise_schedule.signal == 'exposure':
                        noise_schedule.update(exposure)
                    elif noise_schedule.needs_probe(global_step):
                        # disagreement of the two predictors on the CLEAN
                        # rationale, as disagreement on the noisy one would
                        # grow with the noise level itself
                        rp_model.eval()
                        with torch.no_grad():
                            clean_pred = rp_model(**validation_rationale_extractor.extract_from_mask(batch, hard_mask))
                        rp_model.train()
                        noise_schedule.update(js_div_per_example(att_pred.detach(), clean_pred).mean().item())

                bb_loss = bb_model.get_loss(
                    att_pred = att_pred,
                    hard_pred = hard_pred.detach(),
                    labels = batch.labels_bb.to(device),
                    proximity = proximity
                )

                rp_loss = rp_model.get_loss(
                    att_pred = att_pred.detach(),
                    hard_pred = hard_pred,
                    labels = batch.labels_rp.to(device),
                    proximity = proximity
                )

                bb_optimizer.zero_grad()
                rp_optimizer.zero_grad()

                if coupling_weight > 0:
                    # the loss of the rationale predictor also reaches the
                    # generator, through the gate
                    (bb_loss + rp_loss).backward()
                else:
                    bb_loss.backward()
                    rp_loss.backward()

                bb_optimizer.step()
                rp_optimizer.step()
            
                pb.update(1)

                # update running values
                bb_running_train_loss += bb_loss.item()
                rp_running_train_loss += rp_loss.item()
                running_train_replace_ratio += replace_ratio
                running_noise_p += noise_p
                running_exposure += exposure
                global_step += 1

                # validation step
                if global_step % eval_every == 0:
                    bb_model.eval()
                    rp_model.eval()
                    with torch.no_grad():                    
                        for batch in valid_loader:
                            # generate prediction and token probs of being in a rationale
                            batch.reviews_tokenized = batch.reviews_tokenized.to(device)
                            selectable = selectable_fn(batch.reviews_tokenized)
                            att_pred, token_att = bb_model(**batch.reviews_tokenized, selectable = selectable)

                            hard_mask = rationale_selector(
                                token_att = token_att,
                                input_ids = batch.reviews_tokenized.input_ids,
                                selectable = selectable
                            )

                            rationale, _, replace_ratio = validation_rationale_extractor(
                                batch = batch,
                                hard_mask = hard_mask
                            )

                            # predict from rationale
                            hard_pred = rp_model(**rationale)
            
                            bb_valid = bb_model.get_loss(
                                att_pred = att_pred,
                                hard_pred = hard_pred,
                                labels = batch.labels_bb.to(device),
                                proximity = proximity
                            )

                            rp_valid = rp_model.get_loss(
                                att_pred = att_pred,
                                hard_pred = hard_pred,
                                labels = batch.labels_rp.to(device),
                                proximity = proximity
                            )

                            bb_running_valid_loss += bb_valid.item()
                            rp_running_valid_loss += rp_valid.item()
                            running_valid_replace_ratio += replace_ratio

                    # evaluation
                    bb_average_train_loss = bb_running_train_loss / eval_every
                    rp_average_train_loss = rp_running_train_loss / eval_every
                    average_train_replace_ratio = running_train_replace_ratio / eval_every
                    average_noise_p = running_noise_p / eval_every
                    average_exposure = running_exposure / eval_every

                    bb_average_valid_loss = bb_running_valid_loss / len(valid_loader)
                    rp_average_valid_loss = rp_running_valid_loss / len(valid_loader)
                    average_valid_replace_ratio = running_valid_replace_ratio / len(valid_loader)

                    # bb_improved = bb_best_train_loss > bb_average_train_loss and bb_best_valid_loss > bb_average_valid_loss
                    # rp_improved = rp_best_train_loss > rp_average_train_loss and rp_best_valid_loss > rp_average_valid_loss
                    bb_improved = bb_best_valid_loss > bb_average_valid_loss
                    rp_improved = rp_best_valid_loss > rp_average_valid_loss

                    patience_left = patience if bb_improved and rp_improved else patience_left - 1

                    metrics.append({
                        "bb": {
                            "train_loss": bb_average_train_loss,
                            "valid_loss": bb_average_valid_loss
                        },
                        "rp": {
                            "train_loss": rp_average_train_loss,
                            "valid_loss": rp_average_valid_loss
                        },
                        "replace_ratio": {
                            "replace_train_ratio": average_train_replace_ratio,
                            "replace_valid_ratio": average_valid_replace_ratio
                        },
                        # realized noise level: mean over the steps since the
                        # last validation and the level of the next step
                        "noise": {
                            "p_mean": average_noise_p,
                            "p_next": noise_schedule.value(global_step),
                            # 1 = the rationale holds as many of the words noise
                            # injection targets as a random rationale would
                            "exposure": average_exposure,
                            **noise_schedule.state()
                        },
                        "patience_left": patience_left,
                        "step": global_step,
                    })

                    # update running values
                    bb_best_train_loss = min(bb_best_train_loss, bb_average_train_loss)
                    bb_best_valid_loss = min(bb_best_valid_loss, bb_average_valid_loss)
                    rp_best_train_loss = min(rp_best_train_loss, rp_average_train_loss)
                    rp_best_valid_loss = min(rp_best_valid_loss, rp_average_valid_loss)

                    # resetting running values
                    bb_running_train_loss = 0.0
                    rp_running_train_loss = 0.0
                    running_train_replace_ratio = 0.0
                    running_noise_p = 0.0
                    running_exposure = 0.0
                    bb_running_valid_loss = 0.0
                    rp_running_valid_loss = 0.0
                    running_valid_replace_ratio = 0.0

                    # print progress
                    pb.write(f'Epoch [{epoch+1}/{num_epochs}], Step [{global_step}/{num_epochs*len(train_loader)}]')
                    pb.write(f'Noise p: {average_noise_p:.4f}')
                    pb.write(f'Exposure of the rationale to noise: {average_exposure:.4f}')
                    pb.write(f'Train Probability of Replacement: {average_train_replace_ratio * 100:.4f}')
                    pb.write(f'Valid Probability of Replacement: {average_valid_replace_ratio * 100:.4f}')
                    pb.write(f'BB Train Loss: {bb_average_train_loss:.4f}, BB Valid Loss: {bb_average_valid_loss:.4f}')
                    pb.write(f'RP Train Loss: {rp_average_train_loss:.4f}, RP Valid Loss: {rp_average_valid_loss:.4f}')
                    pb.write(f"Patience: {patience_left}")

                    # checkpoint 
                    if bb_improved and rp_improved:
                        pb.write(f'Model saved to ==> {bb_model_save(bb_model, checkpoint_dir)}')
                        pb.write(f'Model saved to ==> {rp_model_save(rp_model, checkpoint_dir)}')
                    pb.write(f'Metrics saved to ==> {metrics_save(metrics, checkpoint_dir)}')

                    bb_model.train()
                    rp_model.train()

            if patience_left == 0:
                break


def evaluate(
    bb_model,
    rp_model,
    tokenizer,
    test_loader,
    device,
    show_detail,
    rationale_selector,
    rationale_extractor,
    checkpoint_dir,
    result_path,
    selectable_fn = lambda reviews_tokenized: None,
    eval_sparsities = (),
    judge = None,
    aopc_bins = (),
    save_attention = False):

    if checkpoint_dir is not None:
        print('Loading BB model')
        bb_model_load(bb_model, checkpoint_dir)
        print('Loading RP model')
        rp_model_load(rp_model, checkpoint_dir)

    if show_detail:
        detail_path = os.path.join(os.path.dirname(checkpoint_dir), "details")
        os.makedirs(detail_path, exist_ok=True)

    gen_spans = 0
    rat_spans = 0
    gen_rat_span_ratio = 0.0
    gen_rat_span_rtotal = 0
    max_gen_span = torch.zeros(1)
    max_rat_span = torch.zeros(1)

    tp = 0
    fp = 0
    fn = 0

    rratio = 0.0

    rprec = 0
    rrec = 0
    rf1 = 0
    rtotal = 0

    y_pred = []
    y_pred_bb = []
    y_true = []

    comp = []
    suff = []
    comp_prob = []
    suff_prob = []

    ious = []
    num_gen_tokens = []
    num_rat_tokens = []

    # per-document comparison with human rationales, at the trained selection
    # rate and at every other selection rate of interest
    counts = []
    counts_at_k = {sparsity: [] for sparsity in eval_sparsities}
    reviews = []
    attention_dump = {"attention": [], "word_ids": [], "gold": [], "selected": []}

    aopc_bins = aopc_bins if judge is not None else ()
    selectors_at_k = {
        sparsity: dataclasses.replace(rationale_selector, sparsity = sparsity)
        for sparsity in set(eval_sparsities) | set(aopc_bins)
    }

    # probabilities of the class predicted by the independent classifier
    judge_probs = {"full": [], "rationale": [], "remainder": [], "null": []}
    judge_correct = []
    aopc_comp = {sparsity: [] for sparsity in aopc_bins}
    aopc_suff = {sparsity: [] for sparsity in aopc_bins}
    if judge is not None:
        judge.eval()
        # empty input: CLS SEP
        null_ids = torch.tensor([[tokenizer.cls_token_id, tokenizer.sep_token_id]], device = device)
        with torch.no_grad():
            null_probs = F.softmax(judge(
                input_ids = null_ids,
                token_type_ids = torch.zeros_like(null_ids),
                attention_mask = torch.ones_like(null_ids)
            ), -1)[0]

    review_count = 0

    bb_model.eval()
    rp_model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch.reviews_tokenized = batch.reviews_tokenized.to(device)
            selectable = selectable_fn(batch.reviews_tokenized)
            # generate prediction and token probs of being in a rationale
            att_pred, token_att = bb_model(**batch.reviews_tokenized, selectable = selectable)
            # mask based on probs
            hard_mask = rationale_selector(
                token_att = token_att,
                input_ids = batch.reviews_tokenized.input_ids,
                selectable = selectable
            )
            masks_at_k = {
                sparsity: selector(
                    token_att = token_att,
                    input_ids = batch.reviews_tokenized.input_ids,
                    selectable = selectable
                ) for sparsity, selector in selectors_at_k.items()
            }
            # apply mask and recover rationale
            rationale, remainder, replace_ratio = rationale_extractor(batch, hard_mask)
            rratio += replace_ratio
            # predict from rationale
            hard_pred_logits = rp_model(**rationale)
            hard_pred_probs = torch.sigmoid(hard_pred_logits)

            y_pred.extend(torch.argmax(hard_pred_logits, 1).tolist())
            y_pred_bb.extend(torch.argmax(att_pred, 1).tolist())
            y_true.extend(batch.labels)

            label_pred_probs = get_label_pred_probs(hard_pred_probs, batch.labels)

            remainder_hard_pred_logits = rp_model(**remainder)
            remainder_hard_pred_probs = torch.sigmoid(remainder_hard_pred_logits)
            remainder_label_pred_probs = get_label_pred_probs(remainder_hard_pred_probs, batch.labels)

            all_hard_pred_logits = rp_model(**batch.reviews_tokenized)
            all_hard_pred_probs = torch.sigmoid(all_hard_pred_logits)
            all_label_pred_probs = get_label_pred_probs(all_hard_pred_probs, batch.labels)

            comp.extend((all_label_pred_probs - remainder_label_pred_probs).tolist())
            suff.extend((all_label_pred_probs - label_pred_probs).tolist())

            # The predictors already return probabilities, so the sigmoid above
            # squashes them into [0.5, 0.73]. Same scores on actual probabilities:
            all_label_probs = get_label_pred_probs(all_hard_pred_logits, batch.labels)
            comp_prob.extend((all_label_probs - get_label_pred_probs(remainder_hard_pred_logits, batch.labels)).tolist())
            suff_prob.extend((all_label_probs - get_label_pred_probs(hard_pred_logits, batch.labels)).tolist())

            # faithfulness according to an independent full-text classifier
            if judge is not None:
                full_probs = F.softmax(judge(**batch.reviews_tokenized), -1)
                judge_pred = full_probs.argmax(-1)
                judge_prob = lambda inputs: F.softmax(judge(**inputs), -1).gather(1, judge_pred.unsqueeze(-1)).squeeze(-1)
                full_prob = full_probs.gather(1, judge_pred.unsqueeze(-1)).squeeze(-1)
                judge_probs["full"].extend(full_prob.tolist())
                judge_probs["rationale"].extend(judge_prob(rationale).tolist())
                judge_probs["remainder"].extend(judge_prob(remainder).tolist())
                judge_probs["null"].extend(null_probs[judge_pred].tolist())
                judge_correct.extend([pred == label for pred, label in zip(judge_pred.tolist(), batch.labels)])
                for sparsity in aopc_bins:
                    rationale_at_k, remainder_at_k, _ = rationale_extractor(batch, masks_at_k[sparsity])
                    aopc_comp[sparsity].extend((full_prob - judge_prob(remainder_at_k)).tolist())
                    aopc_suff[sparsity].extend((full_prob - judge_prob(rationale_at_k)).tolist())

            for i in range(hard_mask.shape[0]):
                word_ids = batch.reviews_tokenized.word_ids(i)
                gold_words = to_gold_words(batch.rationale_ranges[i])
                gen_mask = torch.tensor([False] + hard_mask[i, :, :].squeeze(-1).tolist())
                rat_mask = torch.tensor([word_id in gold_words for word_id in word_ids])

                counts.append(selection_counts(word_ids, gen_mask.tolist(), gold_words))
                if save_attention:
                    # one entry per token after CLS; -1 = SEP/PAD
                    attention_dump["attention"].append(token_att[i, :, 0].float().cpu().numpy().astype(np.float32))
                    attention_dump["word_ids"].append(np.array([-1 if word_id is None else word_id for word_id in word_ids[1:]], dtype = np.int32))
                    attention_dump["gold"].append(rat_mask[1:].numpy())
                    attention_dump["selected"].append(gen_mask[1:].numpy())
                for sparsity in eval_sparsities:
                    counts_at_k[sparsity].append(selection_counts(word_ids, [False] + masks_at_k[sparsity][i, :, :].squeeze(-1).tolist(), gold_words))
                reviews.append(batch.reviews[i])

                gen_span = torch.logical_and(gen_mask[:-1] == False, gen_mask[1:] == True).sum()
                gen_spans += gen_span
                max_gen_span = torch.max(torch.stack([max_gen_span.squeeze(), gen_span.squeeze()]))
                rat_span = torch.logical_and(rat_mask[:-1] == False, rat_mask[1:] == True).sum()
                max_rat_span = torch.max(torch.stack([max_rat_span.squeeze(), rat_span.squeeze()]))
                rat_spans += rat_span
                if not rat_span == torch.zeros(1): # rat_span is LongTensor, so it works
                    gen_rat_span_ratio += gen_span/(rat_span)
                    gen_rat_span_rtotal += 1

                rtp = torch.sum(gen_mask & rat_mask).item()
                tp += rtp
                rfn = torch.sum(~gen_mask & rat_mask).item()
                fn += rfn
                rfp = torch.sum(gen_mask & ~rat_mask).item()
                fp += rfp

                rtotal += 1
                rprec += rtp/(rtp + rfp + 1e-6)
                rrec += rtp/(rtp + rfn + 1e-6)
                rf1 += rtp/(rtp + ((rfp + rfn)/2) + 1e-6)

                if show_detail:
                    with open(os.path.join(detail_path, f"review_{review_count}.txt"), "w") as f:
                        review_tokens = tokenizer.convert_ids_to_tokens(batch.reviews_tokenized.input_ids[i])
                        review_tokens_colored = [color_token(token, g, h) for token, g, h in zip(review_tokens, gen_mask, rat_mask)]
                        print(" ".join(review_tokens_colored), file = f)
                        print(f"Class: {'POSITIVE' if batch.labels[i] else 'NEGATIVE'}", file = f)
                        print(f"P: {100*rtp/(rtp + rfp + 1e-6):.2f} R: {100*rtp/(rtp + rfn + 1e-6):.2f} F1: {100*rtp/(rtp + ((rfp + rfn)/2) + 1e-6):.2f}", file = f)
                    review_count += 1

                gen_sets = to_sets(to_ranges(gen_mask))
                num_gen_tokens.append(sum([len(s) for s in gen_sets]))
                rat_sets = to_sets(to_ranges(rat_mask))
                num_rat_tokens.append(sum([len(s) for s in rat_sets]))

                rious = [(max([len(gen_set & rat_set)/(len(gen_set | rat_set) + 1e-6) for rat_set in rat_sets] + [0.0]), len(gen_set)) for gen_set in gen_sets]
                ious.append(rious)

    micro_prec = safe_div(tp, tp + fp)
    micro_rec = safe_div(tp, tp + fn)
    micro_f1 = safe_div(tp, tp + ((fp + fn)/2))
    macro_prec = rprec/rtotal
    macro_rec = rrec/rtotal
    macro_f1 = rf1/rtotal

    micro_iou = dict()
    macro_iou = dict()
    
    iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    for threshold in iou_thresholds:
        thresholded_ious = [sum([int(riou >= threshold) * riou_tokens for riou, riou_tokens in rious]) for rious in ious]

        micro_iou[threshold] = dict()
        micro_iou[threshold]["prec"] = safe_div(sum(thresholded_ious), sum(num_gen_tokens))
        micro_iou[threshold]["rec"] = safe_div(sum(thresholded_ious), sum(num_rat_tokens))
        micro_iou[threshold]["f1"] = safe_div(2 * micro_iou[threshold]["prec"] * micro_iou[threshold]["rec"], micro_iou[threshold]["prec"] + micro_iou[threshold]["rec"])

        iou_rprec = [x/(y + 1e-6) for x,y in zip(thresholded_ious, num_gen_tokens)]
        iou_rrec = [x/(y + 1e-6) for x,y in zip(thresholded_ious, num_rat_tokens)]
        macro_iou[threshold] = dict()
        macro_iou[threshold]["prec"] = sum(iou_rprec) / len(iou_rprec)
        macro_iou[threshold]["rec"] = sum(iou_rrec) / len(iou_rrec)
        macro_iou[threshold]["f1"] = safe_div(2 * macro_iou[threshold]["prec"] * macro_iou[threshold]["rec"], macro_iou[threshold]["prec"] + macro_iou[threshold]["rec"])

    diagnostics = selection_report(counts, reviews)

    faithfulness_judge = None
    if judge is not None:
        full, on_rationale, on_remainder, null = (np.array(judge_probs[key]) for key in ("full", "rationale", "remainder", "null"))
        norm_suff, norm_comp, normalizable = normalized_faithfulness(full, on_rationale, on_remainder, null)
        faithfulness_judge = {
            "comprehensiveness": float(np.mean(full - on_remainder)),
            "sufficiency": float(np.mean(full - on_rationale)),
            # Carton et al. (2020), both in [0, 1] and higher is better
            "normalized_comprehensiveness": float(np.mean(norm_comp[normalizable])) if normalizable.any() else None,
            "normalized_sufficiency": float(np.mean(norm_suff[normalizable])) if normalizable.any() else None,
            "normalizable_share": float(np.mean(normalizable)),
            # area over the perturbation curve: mean over selection rates
            "aopc_comprehensiveness": float(np.mean([np.mean(aopc_comp[sparsity]) for sparsity in aopc_bins])) if aopc_bins else None,
            "aopc_sufficiency": float(np.mean([np.mean(aopc_suff[sparsity]) for sparsity in aopc_bins])) if aopc_bins else None,
            "curve": {sparsity: {"comprehensiveness": float(np.mean(aopc_comp[sparsity])), "sufficiency": float(np.mean(aopc_suff[sparsity]))} for sparsity in aopc_bins},
            "accuracy": float(np.mean(judge_correct))
        }

    results = {
        # wordpiece-level, as in Storek et al. (2023): selected SEP/PAD tokens
        # count as false positives
        "rationales": {
            "micro": {"prec": micro_prec, "rec": micro_rec, "F1": micro_f1},
            "macro": {"prec": macro_prec, "rec": macro_rec, "F1": macro_f1},
        },
        # word-level: a word is selected if any of its wordpieces is
        "rationales_word_level": plausibility(counts)["word"],
        # floor (random selection) and ceiling (oracle) at the same selection rate
        "reference": reference_points(counts),
        "plausibility_at_k": {
            sparsity: {**plausibility(counts_at_k[sparsity]), **selection_report(counts_at_k[sparsity], reviews)["selection"]}
            for sparsity in eval_sparsities
        },
        "token_selector_sparsity": rationale_selector.sparsity,
        **diagnostics,
        "replace_ratio": rratio/len(test_loader),
        # as in Storek et al. (2023): rationale predictor, sigmoid of probabilities
        "comp_suff": {"comprehensiveness": sum(comp)/rtotal, "sufficiency": sum(suff)/rtotal},
        # rationale predictor, actual probabilities
        "comp_suff_prob": {"comprehensiveness": sum(comp_prob)/rtotal, "sufficiency": sum(suff_prob)/rtotal},
        "faithfulness_judge": faithfulness_judge,
        "macro_iou": macro_iou,
        "micro_iou": micro_iou,
        "accuracy": classification_report(y_true, y_pred, labels=[1,0], digits=4, output_dict=True, zero_division=0)["accuracy"],
        "accuracy_attention_predictor": float(np.mean([pred == label for pred, label in zip(y_pred_bb, y_true)]))
    }

    if save_attention:
        # documents are padded to different lengths, so arrays are stored flat
        lengths = np.array([len(attention) for attention in attention_dump["attention"]])
        np.savez_compressed(
            os.path.join(result_path, "attention.npz"),
            lengths = lengths,
            labels = np.array(y_true),
            **{key: np.concatenate(values) for key, values in attention_dump.items()}
        )

    if not show_detail:
        save_results(results, result_path)
        # per-document records for significance tests across runs
        save_per_example({
            "tp": [count["tp"] for count in counts],
            "fp": [count["fp"] + count["special"] for count in counts],
            "fn": [count["fn"] for count in counts],
            "tp_word": [count["tp_word"] for count in counts],
            "fp_word": [count["fp_word"] for count in counts],
            "fn_word": [count["fn_word"] for count in counts],
            "correct": [int(pred == label) for pred, label in zip(y_pred, y_true)],
            "comprehensiveness": comp_prob,
            "sufficiency": suff_prob
        }, result_path)

    print("Rationales:")
    print(f"Token-level Micro-Averaged Precision: {micro_prec:.4f} Recall: {micro_rec:.4f} F1: {micro_f1:.4f}")
    print(f"Token-level Macro-Averaged Precision: {macro_prec:.4f} Recall: {macro_rec:.4f} F1: {macro_f1:.4f}")
    word_level = results["rationales_word_level"]["micro"]
    print(f"Word-level Micro-Averaged Precision: {word_level['prec']:.4f} Recall: {word_level['rec']:.4f} F1: {word_level['F1']:.4f}")
    for name, reference in results["reference"].items():
        print(f"Reference ({name} selection at the same rate) Micro-Averaged Precision: {reference['prec']:.4f} Recall: {reference['rec']:.4f} F1: {reference['F1']:.4f}")
    for sparsity in eval_sparsities:
        at_k = results["plausibility_at_k"][sparsity]
        print(f"Top {sparsity:.0%} (realized {at_k['realized_rate_wordpiece']:.4f}) Micro-Averaged Precision: {at_k['wordpiece']['micro']['prec']:.4f} Recall: {at_k['wordpiece']['micro']['rec']:.4f} F1: {at_k['wordpiece']['micro']['F1']:.4f}")
    for t in iou_thresholds:
        print(f"Token-level IOU Micro-Averaged Precision (threshold={t}): {micro_iou[t]['prec']:.4f} Recall: {micro_iou[t]['rec']:.4f} F1: {micro_iou[t]['f1']:.4f}")
        print(f"Token-level IOU Macro-Averaged Precision (threshold={t}): {macro_iou[t]['prec']:.4f} Recall: {macro_iou[t]['rec']:.4f} F1: {macro_iou[t]['f1']:.4f}")
    print(f"Replacement Ratio: {rratio/len(test_loader)}")
    print(f"Average number of generated spans: {gen_spans/rtotal:.4f}, labeled rationale spans: {rat_spans/rtotal:.4f}")
    print(f"Maximum number of generated spans: {max_gen_span:.0f}, labeled rationale spans: {max_rat_span:.0f}")
    print(f"Average ratio of generated spans to labeled rationale spans: {safe_div(gen_rat_span_ratio, gen_rat_span_rtotal):.4f}")
    print("Selection:")
    for key, value in {**diagnostics["selection"], **diagnostics["truncation"]}.items():
        print(f"{key}: {value:.4f}")
    print(f"Comprehensiveness: {sum(comp)/rtotal:.4f}")
    print(f"Sufficiency: {sum(suff)/rtotal:.4f}")
    print(f"Comprehensiveness (probabilities): {sum(comp_prob)/rtotal:.4f}")
    print(f"Sufficiency (probabilities): {sum(suff_prob)/rtotal:.4f}")
    if faithfulness_judge is not None:
        print("Faithfulness (independent full-text classifier):")
        for key, value in faithfulness_judge.items():
            if key != "curve" and value is not None:
                print(f"{key}: {value:.4f}")
        for sparsity, point in faithfulness_judge["curve"].items():
            print(f"Top {sparsity:.0%} Comprehensiveness: {point['comprehensiveness']:.4f} Sufficiency: {point['sufficiency']:.4f}")
    print(f"Attention Predictor Accuracy: {results['accuracy_attention_predictor']:.4f}")
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4, zero_division=0))


def save_results(results, result_path):
    with open(os.path.join(result_path, "results.json"), "w") as f:
        json.dump(results, f)


def save_per_example(per_example, result_path):
    with open(os.path.join(result_path, "per_example.json"), "w") as f:
        json.dump(per_example, f)


def to_ranges(mask):
    t1 = torch.tensor(mask.tolist() + [0], dtype=torch.bool)
    t2 = torch.tensor([0] + mask.tolist(), dtype=torch.bool)
    start = torch.logical_and(t1, ~t2)
    end = torch.logical_and(~t1, t2)
    indices = torch.arange(len(t1))
    return list(zip(indices[start].tolist(), indices[end].tolist()))


def to_sets(ranges):
    return [set(range(low, high)) for low, high in ranges]


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


def get_label_pred_probs(pred_probs, labels):
    return torch.tensor([pred_prob[label] for pred_prob, label in zip(pred_probs, labels)])


def color_token(token, generated, handlabeled):
    if generated and handlabeled:
        return colored(token, "green")
    if not generated and handlabeled:
        return colored(token, "blue")
    if generated and not handlabeled:
        return colored(token, "red")
    return token


def model_save(model, path):
    torch.save(model.state_dict(), path)


def bb_model_save(model, path):
    save_path = os.path.join(path, "bb_model.pt")
    model_save(model, save_path)
    return save_path


def rp_model_save(model, path):
    save_path = os.path.join(path, "rp_model.pt")
    model_save(model, save_path)
    return save_path


def model_load(model, path):
    return model.load_state_dict(torch.load(path, map_location = next(model.parameters()).device))


def bb_model_load(model, path):
    model_load(model, os.path.join(path, "bb_model.pt"))


def rp_model_load(model, path):
    model_load(model, os.path.join(path, "rp_model.pt"))


def metrics_save(metrics, path):
    save_path = os.path.join(path, "metrics.json")
    with open(save_path, "w") as f:
        json.dump(metrics, f)
    return save_path


if __name__ == "__main__":
    main(parse_args())
