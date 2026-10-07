import json
import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
from transformers import AutoTokenizer, logging

from models import FullTextClassifier
from movies import DataLoaderFactory
from run_words import model_load, model_save, set_seed

logging.set_verbosity_error()

# Trains a classifier on the full (unmasked) input. It serves as
#   - the accuracy ceiling of rationalization models,
#   - the independent judge of faithfulness (run_words.py --faithfulness_model),
#   - the source of token saliency (build_saliency_probs.py).

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--train", action = "store_true")
    parser.add_argument("--evaluate", action = "store_true")
    parser.add_argument('--device', type = str, default = 'cuda')
    parser.add_argument('--lr', type = float, default = 2e-5)
    parser.add_argument('--num_epochs', type = int, default = 5)
    parser.add_argument('--patience', type = int, default = 2)
    parser.add_argument('--save_path', type = str, default = os.path.join("trained", "full_text"))
    parser.add_argument('--model', type = str, default = 'bert-base-uncased')
    parser.add_argument('--max_length', type = int, default = 512)
    parser.add_argument('--truncation_side', choices = ['right', 'left'], default = 'right')
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--data_path', type = str, default = os.path.join("..", "..", "rnp_movie_review", "original"))
    parser.add_argument('--seed', type = int, default = None)
    parser.add_argument('--train_subset', type = int, default = None)
    parser.add_argument('--valid_subset', type = int, default = None)
    parser.add_argument('--test_subset', type = int, default = None)
    return parser.parse_args()


def get_checkpoint_path(save_path):
    return os.path.join(save_path, "checkpoints", "full_text.pt")


def run_epoch(model, loader, device, optimizer = None):
    total_loss = 0.0
    num_correct = 0
    num_examples = 0
    for batch in tqdm(loader):
        # annotated (test) batches carry labels as a tuple
        labels = batch.labels_bb if hasattr(batch, "labels_bb") else torch.tensor(batch.labels, dtype = torch.long)
        labels = labels.to(device)
        logits = model(**batch.reviews_tokenized.to(device))
        loss = F.cross_entropy(logits, labels)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.shape[0]
        num_correct += (logits.argmax(-1) == labels).sum().item()
        num_examples += labels.shape[0]
    return total_loss/num_examples, num_correct/num_examples


def main(args):
    if not args.train and not args.evaluate:
        print("Must append flag --train or --evaluate")
        return

    if args.seed is not None:
        set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast = True)
    tokenizer.truncation_side = args.truncation_side
    model = FullTextClassifier(num_labels = 2, model = args.model).to(args.device)
    checkpoint_path = get_checkpoint_path(args.save_path)

    def create_dataloader(split, subset):
        return DataLoaderFactory(
            data_path = args.data_path,
            batch_size = args.batch_size,
            tokenizer = tokenizer,
            max_length = args.max_length,
            shuffle = split == "train",
            subset = subset,
            seed = args.seed
        ).create_dataloader(split, False)

    if args.train:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok = True)
        with open(os.path.join(args.save_path, "config.json"), "w") as f:
            json.dump(vars(args), f, indent = 2)
        train_loader = create_dataloader("train", args.train_subset)
        valid_loader = create_dataloader("valid", args.valid_subset)
        optimizer = optim.Adam(model.parameters(), args.lr)
        best_valid_loss = float("Inf")
        patience_left = args.patience
        for epoch in range(args.num_epochs):
            model.train()
            train_loss, train_accuracy = run_epoch(model, train_loader, args.device, optimizer)
            model.eval()
            with torch.no_grad():
                valid_loss, valid_accuracy = run_epoch(model, valid_loader, args.device)
            print(f'Epoch [{epoch+1}/{args.num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
            print(f'Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}')
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_left = args.patience
                model_save(model, checkpoint_path)
                print(f'Model saved to ==> {checkpoint_path}')
            else:
                patience_left -= 1
                if patience_left == 0:
                    print("Patience is 0, early stopping")
                    break

    if args.evaluate:
        model_load(model, checkpoint_path)
        test_loader = create_dataloader("test", args.test_subset)
        model.eval()
        with torch.no_grad():
            test_loss, test_accuracy = run_epoch(model, test_loader, args.device)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        with open(os.path.join(args.save_path, "results.json"), "w") as f:
            json.dump({"accuracy": test_accuracy, "loss": test_loss}, f)


if __name__ == "__main__":
    main(parse_args())
