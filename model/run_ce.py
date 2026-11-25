import json
import os
import random
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import classification_report
from termcolor import colored
from tqdm.auto import tqdm
from transformers import AutoTokenizer, logging

from models import (BlackBoxPredictor, RationaleExtractor,
                    RationaleExtractorFactory, RationalePredictor)
from models_ce import (PairwiseSentenceGenerator, RationaleExtractor, RationaleExtractorFactory,
                       RationaleSelectorFactory, TopkSentenceSelector)
from datasets_ce import DataLoaderFactory

logging.set_verbosity_error()

MULTIRC_PATH = os.path.join("..", "..", "rnp_multirc", "original")
FEVER_PATH = os.path.join("..", "..", "rnp_fever", "original")
DATAPATH = {"multirc": MULTIRC_PATH, "fever": FEVER_PATH}

def parse_args():
    parser = ArgumentParser()
    # Whether to train and/or evaluate
    parser.add_argument("--train", action = "store_true")
    parser.add_argument("--evaluate", action = "store_true")
    # Whether to inject noise
    parser.add_argument("--inject_noise", action = "store_true")
    # Magnitude of augmentation hyperparameter
    parser.add_argument('--noise_p', type = float, default = 0.1)
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
    parser.add_argument('--patience', type = int, default = -1)
    # Model proximity hyperparameter
    parser.add_argument('--proximity', type = float, default = 0.1)
    # Model
    parser.add_argument('--save_path', type = str, default = os.path.join("trained", "ours"))
    parser.add_argument('--model', type = str, default = 'bert-base-uncased')
    parser.add_argument('--max_length', type = int, default = 512)
    parser.add_argument('--batch_size', type = int, default = 16)
    # Selection method
    parser.add_argument('--selection_method', type = str, default = "words", choices = ["words", "sentences"])
    # Rationale Extraction hyperparameter
    parser.add_argument('--sparsity', type = float, default = 0.2)
    # Dataset
    parser.add_argument('--dataset', type = str, default = "multirc", choices = ["multirc", "fever"])
    parser.add_argument('--data_path', type = str, default = None)
    # Eval-related
    # Compare model-generated and hand-labeled rationales
    parser.add_argument('--show_detail', action = "store_true")
    # Random seed: initialization, batch order and noise
    parser.add_argument('--seed', type = int, default = None)
    # NI as released: during training with noise, the rationale-based
    # predictor receives words shifted away from the selected ones (training only)
    parser.add_argument('--legacy_alignment', action = "store_true")
    # Selection as released, for every model: sentence scores from shifted
    # words, sentences cut off by max_length ranked first, and the query twice
    # in the input used for comprehensiveness (training and evaluation)
    parser.add_argument('--legacy_selection', action = "store_true")
    # Validations per epoch; patience counts validations. The released code
    # validates twice per epoch, our USR runs (run_words.py) once
    parser.add_argument('--validations_per_epoch', type = int, default = 2)
    # Evaluate with k sentences picked uniformly at random (random-selection control)
    parser.add_argument('--random_selection', action = "store_true")
    # Stop training after this many hours and evaluate the best checkpoint so far
    parser.add_argument('--time_limit_hours', type = float, default = None)
    # Generator: "joint" encodes query and document together (BERT-A2R);
    # "pairwise" is a frozen BERT that encodes the query with each sentence
    # on its own and trains only its two linear layers
    parser.add_argument('--generator', type = str, default = "joint", choices = ["joint", "pairwise"])
    parser.add_argument('--pair_max_length', type = int, default = 128)
    # Joint generator: attend to document tokens only. The released generator's
    # attention also covers the query, [SEP] and padding, so its loss can be met
    # by attending to the query alone.
    parser.add_argument('--document_attention', action = "store_true")
    # Product of experts with a frozen query-only model (output of query_only.py):
    # the generator is trained on what the selected evidence adds to the query
    parser.add_argument('--poe_probs', type = str, default = None)
    # Use the first examples only, for a fast debugging loop
    parser.add_argument('--train_subset', type = int, default = None)
    parser.add_argument('--valid_subset', type = int, default = None)
    parser.add_argument('--test_subset', type = int, default = None)
    args = parser.parse_args()
    if args.data_path is None:
        args.data_path = DATAPATH[args.dataset]
    return args


def main(args):
    if not args.train and not args.evaluate:
        print("Must append flag --train or --evaluate")
        return

    checkpoint_dir = os.path.join(args.save_path, "checkpoints")

    if args.seed is not None:
        set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast = True)

    rationale_selector = RationaleSelectorFactory(args.sparsity, args.max_length, tokenizer.pad_token_id, args.device).create_selector(args.selection_method, args.legacy_selection)

    if args.generator == "pairwise":
        if args.selection_method != "sentences":
            raise ValueError("--generator pairwise requires --selection_method sentences")
        if args.document_attention:
            raise ValueError("--document_attention is for the joint generator; the pairwise one attends to sentences only")
        bb_model = PairwiseSentenceGenerator(num_labels = 2, model = args.model, tokenizer = tokenizer, selector = rationale_selector,
                                             max_pair_length = args.pair_max_length).to(args.device)
    else:
        bb_model = BlackBoxPredictor(num_labels = 2, model = args.model, freeze_encoder = args.freeze_encoder_bb).to(args.device)
    print(f"Black Box Predictor: {get_num_params(bb_model)} parameters, {sum(p.numel() for p in bb_model.parameters() if p.requires_grad)} trainable")

    rp_model = RationalePredictor(num_labels = 2, model = args.model, freeze_encoder = args.freeze_encoder_rp).to(args.device)
    print(f"Rationale Predictor: {get_num_params(rp_model)} parameters")

    rationale_extractor = RationaleExtractorFactory(tokenizer, args.device, args.data_path, args.seed, args.legacy_alignment, args.legacy_selection).create_extractor(args.inject_noise)

    if args.train:
        os.makedirs(checkpoint_dir, exist_ok = True)
        with open(os.path.join(args.save_path, "config.json"), "w") as f:
            json.dump(vars(args), f, indent = 2)
        train_loader = DataLoaderFactory(
            data_path = args.data_path,
            noise_p = args.noise_p,
            batch_size = args.batch_size,
            tokenizer = tokenizer,
            max_length = args.max_length,
            shuffle = True,
            subset = args.train_subset
        ).create_dataloader("train", args.inject_noise)
        valid_loader = DataLoaderFactory(
            data_path = args.data_path,
            noise_p = args.noise_p,
            batch_size = args.batch_size,
            tokenizer = tokenizer,
            max_length = args.max_length,
            shuffle = True,
            subset = args.valid_subset
        ).create_dataloader("valid", False)  # validation and test are noise-free; only train has replacement probabilities

        bb_optimizer = optim.Adam(bb_model.parameters(), args.bb_lr)
        rp_optimizer = optim.Adam(rp_model.parameters(), args.rp_lr)

        validation_rationale_extractor = RationaleExtractor(tokenizer, args.device, legacy_selection = args.legacy_selection)

        poe = None
        if args.poe_probs is not None:
            poe = {}
            for split in ("train", "valid"):
                with open(os.path.join(args.poe_probs, f"query_probs_{split}.json")) as f:
                    poe[split] = json.load(f)

        train(
            bb_model = bb_model,
            bb_optimizer = bb_optimizer,
            rp_model = rp_model,
            rp_optimizer = rp_optimizer,
            train_loader = train_loader,
            valid_loader = valid_loader,
            eval_every = len(train_loader) // args.validations_per_epoch,
            device = args.device,
            proximity = args.proximity,
            num_epochs = args.num_epochs,
            patience = args.patience,
            checkpoint_dir = checkpoint_dir,
            rationale_selector = rationale_selector,
            rationale_extractor = rationale_extractor,
            validation_rationale_extractor = validation_rationale_extractor,
            time_limit_hours = args.time_limit_hours,
            document_attention = args.document_attention,
            sep_token_id = tokenizer.sep_token_id,
            poe = poe,
        )

    if args.evaluate:
        test_loader = DataLoaderFactory(
            data_path = args.data_path,
            noise_p = args.noise_p,
            batch_size = args.batch_size,
            tokenizer = tokenizer,
            max_length = args.max_length,
            shuffle = False,
            subset = args.test_subset
        ).create_dataloader("test", False)  # validation and test are noise-free; only train has replacement probabilities

        sentence_selector = TopkSentenceSelector(args.sparsity, args.max_length, tokenizer.pad_token_id, args.device,
                                                 random = args.random_selection, seed = args.seed, legacy = args.legacy_selection)
        test_rationale_extractor = RationaleExtractor(tokenizer, args.device, legacy_selection = args.legacy_selection)

        evaluate(
            bb_model = bb_model,
            rp_model = rp_model,
            test_loader = test_loader,
            device = args.device,
            show_detail = args.show_detail,
            rationale_selector = rationale_selector,
            rationale_extractor = test_rationale_extractor,
            sentence_selector = sentence_selector,
            checkpoint_dir = checkpoint_dir,
            result_path = args.save_path,
            random_selection = args.random_selection,
            document_attention = args.document_attention,
            sep_token_id = tokenizer.sep_token_id
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
    time_limit_hours = None,
    document_attention = False,
    sep_token_id = None,
    poe = None,
    ):

    start_time = time.time()
    out_of_time = False

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

                if time_limit_hours is not None and time.time() - start_time > time_limit_hours * 3600:
                    pb.write(f"Time limit of {time_limit_hours} h reached, stopping training")
                    out_of_time = True
                    metrics.append({"stopped_by_time_limit": True, "step": global_step})
                    if not os.path.exists(os.path.join(checkpoint_dir, "bb_model.pt")):
                        # stopped before the first validation: keep the current models
                        bb_model_save(bb_model, checkpoint_dir)
                        rp_model_save(rp_model, checkpoint_dir)
                    metrics_save(metrics, checkpoint_dir)
                    break

                # generate prediction and token probs of being in a rationale
                batch.tokenized_examples = batch.tokenized_examples.to(device)
                att_pred, token_att = generate(bb_model, batch, document_attention, sep_token_id)

                with torch.no_grad():
                    hard_mask = rationale_selector(batch, token_att)
                    rationale, _, replace_ratio = rationale_extractor(batch, hard_mask)

                # predict from rationale
                hard_pred = rp_model(**rationale)

                bb_loss = bb_model.get_loss(
                    att_pred = att_pred,
                    hard_pred = hard_pred.detach(),
                    labels = batch.labels_bb.to(device),
                    proximity = proximity,
                    query_logprobs = query_logprobs(poe, "train", batch, device)
                )

                rp_loss = rp_model.get_loss(
                    att_pred = att_pred.detach(),
                    hard_pred = hard_pred,
                    labels = batch.labels_rp.to(device),
                    proximity = proximity
                )

                bb_optimizer.zero_grad()
                rp_optimizer.zero_grad()

                bb_loss.backward()
                rp_loss.backward()

                bb_optimizer.step()
                rp_optimizer.step()
            
                pb.update(1)

                # update running values
                bb_running_train_loss += bb_loss.item()
                rp_running_train_loss += rp_loss.item()
                running_train_replace_ratio += replace_ratio
                global_step += 1

                # validation step
                if global_step % eval_every == 0:
                    bb_model.eval()
                    rp_model.eval()
                    with torch.no_grad():                    
                        for batch in valid_loader:
                            # generate prediction and token probs of being in a rationale
                            batch.tokenized_examples = batch.tokenized_examples.to(device)
                            att_pred, token_att = generate(bb_model, batch, document_attention, sep_token_id)

                            with torch.no_grad():
                                hard_mask = rationale_selector(batch, token_att)
                                rationale, _, replace_ratio = validation_rationale_extractor(batch, hard_mask)

                            # predict from rationale
                            hard_pred = rp_model(**rationale)

                            bb_valid = bb_model.get_loss(
                                att_pred = att_pred,
                                hard_pred = hard_pred,
                                labels = batch.labels_bb.to(device),
                                proximity = proximity,
                                query_logprobs = query_logprobs(poe, "valid", batch, device)
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

                    bb_average_valid_loss = bb_running_valid_loss / len(valid_loader)
                    rp_average_valid_loss = rp_running_valid_loss / len(valid_loader)
                    average_valid_replace_ratio = running_valid_replace_ratio / len(valid_loader)

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
                        "patience_left": patience_left,
                        "step": global_step
                    })

                    # update running values
                    if bb_improved and rp_improved:
                        bb_best_train_loss = min(bb_best_train_loss, bb_average_train_loss)
                        bb_best_valid_loss = min(bb_best_valid_loss, bb_average_valid_loss)
                        rp_best_train_loss = min(rp_best_train_loss, rp_average_train_loss)
                        rp_best_valid_loss = min(rp_best_valid_loss, rp_average_valid_loss)

                    # resetting running values
                    bb_running_train_loss = 0.0
                    rp_running_train_loss = 0.0
                    running_train_replace_ratio = 0.0
                    bb_running_valid_loss = 0.0
                    rp_running_valid_loss = 0.0
                    running_valid_replace_ratio = 0.0

                    # print progress
                    pb.write(f'Epoch [{epoch+1}/{num_epochs}], Step [{global_step}/{num_epochs*len(train_loader)}]')
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

            if patience_left == 0 or out_of_time:
                break

# In Progress
def evaluate(
    bb_model,
    rp_model,
    test_loader,
    device,
    show_detail,
    rationale_selector,
    rationale_extractor,
    sentence_selector,
    checkpoint_dir,
    result_path,
    random_selection = False,
    document_attention = False,
    sep_token_id = None
    ):

    if checkpoint_dir is not None:
        print('Loading BB model')
        bb_model_load(bb_model, checkpoint_dir)
        print('Loading RP model')
        rp_model_load(rp_model, checkpoint_dir)

    tp = 0
    fp = 0
    fn = 0

    rratio = 0.0

    rprec = 0
    rrec = 0
    rf1 = 0
    rtotal = 0

    y_pred = []
    y_true = []

    comp = []
    suff = []
    comp_prob = []
    suff_prob = []
    per_example = {"label": [], "pred": [], "tp": [], "fp": [], "fn": [], "num_selected": [], "num_gold": []}

    bb_model.eval()
    rp_model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch.tokenized_examples = batch.tokenized_examples.to(device)
            # generate prediction and token probs of being in a rationale
            _, token_att = generate(bb_model, batch, document_attention, sep_token_id)
            # mask based on probs
            if random_selection:
                hard_mask, selected_sentences = sentence_selector.select(batch, token_att)
            else:
                hard_mask = rationale_selector(batch, token_att)
                selected_sentences = sentence_selector.get_selected_sentences(batch, token_att)
            # apply mask and recover rationale
            rationale, remainder, replace_ratio = rationale_extractor(batch, hard_mask)
            rratio += replace_ratio
            # predict from rationale
            # the predictor outputs probabilities; the released code applies a
            # sigmoid to them for comp_suff, comp_suff_prob uses them as they are
            hard_pred_logits = rp_model(**rationale)
            hard_pred_probs = torch.sigmoid(hard_pred_logits)
            rationale_probs = get_label_pred_probs(hard_pred_logits, batch.labels)

            y_pred.extend(torch.argmax(hard_pred_logits, 1).tolist())
            y_true.extend(batch.labels)

            label_pred_probs = get_label_pred_probs(hard_pred_probs, batch.labels)

            remainder_outputs = rp_model(**remainder)
            remainder_hard_pred_probs = torch.sigmoid(remainder_outputs)
            remainder_label_pred_probs = get_label_pred_probs(remainder_hard_pred_probs, batch.labels)

            all_outputs = rp_model(**batch.tokenized_examples)
            all_hard_pred_probs = torch.sigmoid(all_outputs)
            all_label_pred_probs = get_label_pred_probs(all_hard_pred_probs, batch.labels)

            comp.extend((all_label_pred_probs - remainder_label_pred_probs).tolist())
            suff.extend((all_label_pred_probs - label_pred_probs).tolist())
            all_probs = get_label_pred_probs(all_outputs, batch.labels)
            comp_prob.extend((all_probs - get_label_pred_probs(remainder_outputs, batch.labels)).tolist())
            suff_prob.extend((all_probs - rationale_probs).tolist())

            per_example["label"].extend(int(label) for label in batch.labels)
            per_example["pred"].extend(torch.argmax(hard_pred_logits, 1).tolist())
            for gen_sents, rat_sents in zip(selected_sentences, batch.rationale_ranges):
                if show_detail:
                    raise NotImplementedError("Show detail has not been implemented yet")
                
                gen_sents = set(gen_sents)
                rat_sents = set(rat_sents)

                rtp = len(gen_sents.intersection(rat_sents))
                tp += rtp
                rfn = len(rat_sents - gen_sents)
                fn += rfn
                rfp = len(gen_sents - rat_sents)
                fp += rfp

                per_example["tp"].append(rtp)
                per_example["fp"].append(rfp)
                per_example["fn"].append(rfn)
                per_example["num_selected"].append(len(gen_sents))
                per_example["num_gold"].append(len(rat_sents))

                rtotal += 1
                rprec += rtp/(rtp + rfp + 1e-6)
                rrec += rtp/(rtp + rfn + 1e-6)
                rf1 += rtp/(rtp + ((rfp + rfn)/2) + 1e-6)

    results = {
        "rationales": {
            "micro": {"prec": tp/(tp + fp), "rec": tp/(tp + fn), "F1": tp/(tp + ((fp + fn)/2))},
            "macro": {"prec": rprec/rtotal, "rec": rrec/rtotal, "F1": rf1/rtotal},
        },
        "sentence_selector_sparsity": sentence_selector.sparsity,
        "replace_ratio": rratio/len(test_loader),
        "comp_suff": {"comprehensiveness": sum(comp)/rtotal, "sufficiency": sum(suff)/rtotal},
        "comp_suff_prob": {"comprehensiveness": sum(comp_prob)/rtotal, "sufficiency": sum(suff_prob)/rtotal},
        "selection_method": "random" if random_selection else "attention",
        "majority_class_share": max(np.mean(np.array(per_example["pred"]) == 1), np.mean(np.array(per_example["pred"]) == 0)),
        "mean_sentences_selected": float(np.mean(per_example["num_selected"])),
        "accuracy": classification_report(y_true, y_pred, labels=[1,0], digits=4, output_dict=True)["accuracy"]
    }

    save_results(results, result_path)
    with open(os.path.join(result_path, "per_example.json"), "w") as f:
        json.dump(per_example, f)

    print("Rationales:")
    print(f"Sentence-level Micro-Averaged Precision: {tp/(tp + fp):.4f} Recall: {tp/(tp + fn):.4f} F1: {tp/(tp + ((fp + fn)/2)):.4f}")
    print(f"Sentence-level Macro-Averaged Precision: {rprec/rtotal:.4f} Recall: {rrec/rtotal:.4f} F1: {rf1/rtotal:.4f}")
    print(f"Replacement Ratio: {rratio/len(test_loader)}")
    print(f"Comprehensiveness: {sum(comp)/rtotal:.4f}")
    print(f"Sufficiency: {sum(suff)/rtotal:.4f}")
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))


def save_results(results, result_path):
    with open(os.path.join(result_path, "results.json"), "w") as f:
        json.dump(results, f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate(bb_model, batch, document_attention = False, sep_token_id = None):
    # class probabilities of the attention-based predictor and token attention
    if isinstance(bb_model, PairwiseSentenceGenerator):
        return bb_model(batch)
    if document_attention:
        return bb_model(**batch.tokenized_examples, selectable = document_tokens(batch, sep_token_id))
    return bb_model(**batch.tokenized_examples)


def document_tokens(batch, sep_token_id):
    # [CLS] query [SEP] document [SEP] padding: True for the document's tokens,
    # aligned with the generator's attention, which leaves out [CLS]
    input_ids = batch.tokenized_examples.input_ids
    real = (batch.tokenized_examples.attention_mask == 1) & (input_ids != sep_token_id)
    positions = torch.arange(input_ids.shape[1], device = input_ids.device).unsqueeze(0)
    start = torch.tensor(batch.evidences_start, device = input_ids.device).unsqueeze(1)
    selectable = (real & (positions >= start))[:, 1:]
    # a query that fills the whole window leaves no document token: keep every real token
    empty = ~selectable.any(dim = 1)
    if empty.any():
        selectable[empty] = real[:, 1:][empty]
    return selectable


def query_logprobs(poe, split, batch, device):
    # log-probabilities of the frozen query-only model for the batch's queries
    if poe is None:
        return None
    probs = [poe[split][" ".join(query)] for query in batch.examples.queries]
    return torch.tensor(probs, dtype = torch.float, device = device).clamp_min(1e-6).log()


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
    return model.load_state_dict(torch.load(path, map_location = "cpu"))


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
