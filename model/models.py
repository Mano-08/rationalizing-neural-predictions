import os
import pickle
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedTokenizerBase


def js_div(P, Q):
    M = (P + Q)/2
    kl_div = lambda X, M: F.kl_div(X.log(), M, reduction = 'sum')
    return (kl_div(P, M) + kl_div(Q, M))/2


class BlackBoxPredictor(nn.Module):
    def __init__(self, num_labels:int, model:str, freeze_encoder:bool, num_layers:Optional[int] = None):
        super().__init__()
        self.num_labels = num_labels
        self.encoder = AutoModel.from_pretrained(model)
        if num_layers is not None:
            # Keep the lower layers only. Every layer mixes more of the whole
            # text into each token, until the attention below can sit on any
            # token (a period, "the") and still see the evidence.
            self.encoder.encoder.layer = self.encoder.encoder.layer[:num_layers]
            self.encoder.config.num_hidden_layers = num_layers
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.token_predictor = nn.Sequential(
            nn.Dropout(self.encoder.config.hidden_dropout_prob),
            nn.Linear(self.encoder.config.hidden_size, 1)
        )
        self.predictor = nn.Sequential(
            nn.Dropout(self.encoder.config.hidden_dropout_prob),
            nn.Linear(self.encoder.config.hidden_size, self.num_labels)
        )

    def forward(self, input_ids, token_type_ids, attention_mask, selectable = None, evidence_noise = None):
        # get contextualized embeddings from a transformer-based encoder
        outputs = self.encoder(
            input_ids = input_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask
        )
        # get hidden states without the CLS token
        hidden_states_no_cls = outputs[0][:, 1:, :]
        # use token classification head to generate token logits
        token_logits = self.token_predictor(hidden_states_no_cls).squeeze(-1)
        # keep SEP and PAD tokens from receiving attention
        if selectable is not None:
            token_logits = token_logits.masked_fill(~selectable, float("-inf"))
        # use softmax to get attention over tokens
        token_att = F.softmax(token_logits, -1).unsqueeze(-1)
        # Noise injection in the generator's own evidence: the state pooled at
        # a token is replaced, with the token's replacement probability, by the
        # state of a random token of the batch. Attending to the words noise
        # injection targets then yields unreliable evidence, and unlike noise
        # on the rationale this reaches the generator through its own loss.
        # evidence_noise holds the probabilities, -1 for tokens that are no words.
        pooled_states = hidden_states_no_cls
        if evidence_noise is not None:
            is_word = evidence_noise >= 0
            replace = (torch.rand(evidence_noise.shape, device = evidence_noise.device) < evidence_noise) & is_word
            if replace.any():
                word_states = hidden_states_no_cls[is_word].detach()
                random_states = word_states[torch.randint(len(word_states), (int(replace.sum()),), device = word_states.device)]
                pooled_states = hidden_states_no_cls.clone()
                pooled_states[replace] = random_states
        # generate context vector
        ctx_vec = torch.bmm(pooled_states.transpose(1, 2), token_att).squeeze(-1)
        # return predicted labels, per token probabilities P(z|x)
        return F.softmax(self.predictor(ctx_vec), -1), token_att

    def get_loss(self, att_pred, hard_pred, labels, proximity):
        return F.cross_entropy(att_pred, labels) + proximity * js_div(att_pred, hard_pred)


class RationalePredictor(nn.Module):
    def __init__(self, num_labels:int, model:str, freeze_encoder:bool):
        super().__init__()
        self.num_labels = num_labels
        self.encoder = AutoModel.from_pretrained(model)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.predictor = nn.Sequential(
            nn.Dropout(self.encoder.config.hidden_dropout_prob),
            nn.Linear(self.encoder.config.hidden_size, self.num_labels)
        )

    def forward(self, input_ids, token_type_ids, attention_mask, gate = None):
        if gate is None:
            outputs = self.encoder(
                input_ids = input_ids,
                token_type_ids = token_type_ids,
                attention_mask = attention_mask
            )
        else:
            # scale every token embedding by its gate (see get_gate)
            embeddings = self.encoder.get_input_embeddings()(input_ids) * gate.unsqueeze(-1)
            outputs = self.encoder(
                inputs_embeds = embeddings,
                token_type_ids = token_type_ids,
                attention_mask = attention_mask
            )
        return F.softmax(self.predictor(outputs[1]), -1)

    def get_loss(self, att_pred, hard_pred, labels, proximity):
        return F.cross_entropy(hard_pred, labels) + proximity * js_div(att_pred, hard_pred)


class FullTextClassifier(nn.Module):
    def __init__(self, num_labels:int, model:str):
        super().__init__()
        self.num_labels = num_labels
        self.encoder = AutoModel.from_pretrained(model)
        self.predictor = nn.Sequential(
            nn.Dropout(self.encoder.config.hidden_dropout_prob),
            nn.Linear(self.encoder.config.hidden_size, self.num_labels)
        )

    # returns logits, unlike the two predictors above
    def forward(self, input_ids = None, token_type_ids = None, attention_mask = None, inputs_embeds = None):
        outputs = self.encoder(
            input_ids = input_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask,
            inputs_embeds = inputs_embeds
        )
        return self.predictor(outputs[1])


def get_replacement_probs(batch, device):
    # replacement probability (at p = 1) of the word of every token after
    # CLS, -1 for tokens that are no words
    probs = torch.full(batch.reviews_tokenized.input_ids[:, 1:].shape, -1.0)
    for i, replacement_probs in enumerate(batch.replacement_probs):
        word_ids = batch.reviews_tokenized.word_ids(i)[1:]
        positions = [position for position, word_id in enumerate(word_ids) if word_id is not None]
        probs[i, positions] = torch.as_tensor(replacement_probs[[word_ids[position] for position in positions]], dtype = torch.float)
    return probs.to(device)


def get_selectable(reviews_tokenized, tokenizer):
    # real tokens that may be attended to and selected, without the CLS column
    input_ids = reviews_tokenized.input_ids[:, 1:]
    selectable = reviews_tokenized.attention_mask[:, 1:].bool()
    for special_id in (tokenizer.pad_token_id, tokenizer.sep_token_id, tokenizer.cls_token_id):
        selectable = selectable & (input_ids != special_id)
    return selectable


@dataclass
class RationaleExtractor:
    tokenizer: PreTrainedTokenizerBase
    device: str

    def extract_from_mask(self, batch, hard_mask):
        # Add CLS tokens
        hard_mask = torch.cat((torch.ones_like(hard_mask[:, :1]), hard_mask), 1).squeeze(-1) # (batch_size, seq_len - CLS, 1) -> (batch_size, seq_len)
        # Mask PAD tokens
        hard_mask.masked_fill_(batch.reviews_tokenized.input_ids == self.tokenizer.pad_token_id, False)
        # Unmask SEP tokens - because we are taking rationale batches as is
        hard_mask.masked_fill_(batch.reviews_tokenized.input_ids == self.tokenizer.sep_token_id, True)
        # Get max seq length to pad to
        max_len = hard_mask.sum(dim = 1).max()
        # Extract rationales and pad
        rationale_ids = []
        attention_mask = []
        for ids, mask in zip(batch.reviews_tokenized.input_ids, hard_mask):
            rationale_ids.append(ids.masked_select(mask).unsqueeze(0))
            attention_mask.append(torch.ones_like(rationale_ids[-1], device = self.device))
            # pad if necessary
            if rationale_ids[-1].shape[1] < max_len:
                pad = torch.full(
                    size = (1, max_len - rationale_ids[-1].shape[1]),
                    fill_value = self.tokenizer.pad_token_id,
                    device = self.device
                )
                rationale_ids[-1] = torch.cat((rationale_ids[-1], pad), 1)
                attention_mask[-1] = torch.cat((attention_mask[-1], torch.zeros_like(pad)), 1)

        # to tensors
        rationale_ids = torch.cat(rationale_ids, 0).to(self.device)
        attention_mask = torch.cat(attention_mask, 0).to(self.device)
        token_type_ids = torch.zeros_like(rationale_ids).to(self.device)
        return {'input_ids': rationale_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}


    def __call__(self, batch, hard_mask):
        tokenized_rationales = self.extract_from_mask(batch, hard_mask)
        tokenized_remainders = self.extract_from_mask(batch, ~hard_mask)
        replacement_ratio = 0
        return tokenized_rationales, tokenized_remainders, replacement_ratio


@dataclass
class BaseNoisyRationaleExtractor(RationaleExtractor):
    data_path: str
    seed: Optional[int] = None
    # current noise level, updated by the noise schedule on every step
    noise_p: float = 0.0
    # reproduce the released code, see extract_from_mask_with_replacement
    legacy_alignment: bool = False
    # words of the last batch of rationales and how exposed they are to noise
    selected_words: Optional[list] = None
    exposure: float = 1.0

    def __post_init__(self):
        self.load_scored_vocab()
        self.rng = np.random.default_rng(self.seed)

    def load_scored_vocab(self):
        with open(os.path.join(self.data_path, "word_statistics", "scored_vocab.pkl"), "rb") as f:
            self.vocab, self.scores = pickle.load(f)

    @abstractmethod
    def extract_rationale(self, review, indices, replacement_mask, doc_id):
        pass

    def extract_remainder(self, review, remainder_indices):
        return [review[i] for i in remainder_indices]

    def batch_tokenize(self, text):
        return self.tokenizer(text = text, is_split_into_words = True, padding = True, return_tensors = 'pt').to(self.device)

    def extract_from_mask_with_replacement(self, batch, hard_mask):
        # Mask PAD tokens
        hard_mask = hard_mask.squeeze(-1).clone() # (batch_size, seq_len - CLS, 1) -> (batch_size, seq_len - CLS)
        hard_mask.masked_fill_(batch.reviews_tokenized.input_ids[:,1:] == self.tokenizer.pad_token_id, False) # (batch_size, seq_len - CLS, 1)
        # Mask SEP tokens - because we are retokenizing rationale batches
        hard_mask.masked_fill_(batch.reviews_tokenized.input_ids[:,1:] == self.tokenizer.sep_token_id, False) # (batch_size, seq_len - CLS, 1)
        # Extract rationales and pad
        rationales = []
        replacement_ratio_sum = 0
        exposure_sum = 0
        self.selected_words = []
        doc_ids = batch.doc_ids if batch.doc_ids is not None else [None] * len(batch.reviews)
        for i, (review, replacement_probs, mask, doc_id) in enumerate(zip(batch.reviews, batch.replacement_probs, hard_mask, doc_ids)):
            # Masked select. hard_mask has no CLS token, so word ids must not
            # have one either. The released code kept it, which hands the
            # predictor the token BEFORE every selected token.
            word_ids = batch.reviews_tokenized.word_ids(i)
            if not self.legacy_alignment:
                word_ids = word_ids[1:]
            indices = [x for x, is_masked in zip(word_ids, mask) if x is not None and is_masked]
            # Unique Consecutive
            indices = [x for i, x in enumerate(indices) if i == 0 or x != indices[i - 1]]
            # If no valid indices (selected tokens do not map to words such as
            # selecting only CLS/SEP/PAD tokens), select the first word
            if len(indices) == 0:
                indices = [0]
            # Sample replacement mask
            replacement_mask = self.rng.binomial(n = 1, p = np.minimum(replacement_probs[indices] * self.noise_p, 1))
            # Extract rationale
            rationale, replacement_ratio = self.extract_rationale(
                review = review,
                indices = indices,
                replacement_mask = replacement_mask,
                doc_id = doc_id
            )
            replacement_ratio_sum += replacement_ratio
            rationales.append(rationale)
            self.selected_words.append(indices)
            exposure_sum += float(np.mean(replacement_probs[indices]))

        average_replacement_ratio = replacement_ratio_sum / len(batch.reviews)
        # Mean replacement probability of the selected words at p = 1. It is 1
        # for a random rationale, above 1 if the rationale favors the words
        # noise injection targets and below 1 if it avoids them.
        self.exposure = exposure_sum / len(batch.reviews)

        tokenized_rationales = self.batch_tokenize(rationales)
        return tokenized_rationales, average_replacement_ratio

    def __call__(self, batch, hard_mask):
        tokenized_rationales, replacement_ratio = self.extract_from_mask_with_replacement(batch, hard_mask)
        tokenized_remainder = self.extract_from_mask(batch, ~hard_mask)
        return tokenized_rationales, tokenized_remainder, replacement_ratio

    def get_gate(self, batch, token_att, tokenized_rationales, weight):
        # Straight-through coupling of the generator to the rationale
        # predictor. Every rationale token gets a gate that equals 1, so the
        # predictor sees the same input, but whose gradient is that of
        # weight * log(attention of the word the token stands for). The loss
        # of the predictor on the (noisy) rationale thereby tells the
        # generator which of its selected words helped and which hurt.
        gate = torch.ones(tokenized_rationales.input_ids.shape, dtype = token_att.dtype, device = token_att.device)
        attention = token_att.squeeze(-1)
        for i, words in enumerate(self.selected_words):
            # attention of every word = sum over its wordpieces
            word_ids = batch.reviews_tokenized.word_ids(i)[1:]
            positions = [position for position, word_id in enumerate(word_ids) if word_id is not None]
            word_of_position = torch.tensor([word_ids[position] for position in positions], device = token_att.device)
            word_attention = torch.zeros(len(batch.reviews[i]), dtype = token_att.dtype, device = token_att.device)
            word_attention = word_attention.index_add(0, word_of_position, attention[i, positions])
            log_attention = word_attention.clamp_min(1e-12).log()
            # rationale token -> rationale word -> word of the document
            rationale_positions = [(position, words[word_id]) for position, word_id in enumerate(tokenized_rationales.word_ids(i)) if word_id is not None]
            if len(rationale_positions) == 0:
                continue
            token_positions, document_words = zip(*rationale_positions)
            selected = log_attention[list(document_words)]
            gate[i, list(token_positions)] = 1 + weight * (selected - selected.detach())
        return gate


@dataclass
class RandomNoisyRationaleExtractor(BaseNoisyRationaleExtractor):
    def __post_init__(self):
        super().__post_init__()
        self.vocab = np.array(self.vocab)

    def extract_rationale(self, review, indices, replacement_mask, doc_id = None):
        rationale = np.array(review, dtype = self.vocab.dtype)[indices]
        rationale[replacement_mask == 1] = self.rng.choice(self.vocab, size = replacement_mask.sum(), p = self.scores)
        return rationale.tolist(), replacement_mask.sum()/replacement_mask.shape[0]


@dataclass
class MLMNoisyRationaleExtractor(RandomNoisyRationaleExtractor):
    # Replaces tokens with in-context substitutes sampled offline from a masked
    # language model (see build_mlm_replacements.py) instead of words drawn
    # from the whole vocabulary
    split: str = "train"

    def __post_init__(self):
        super().__post_init__()
        stats_path = os.path.join(self.data_path, "word_statistics")
        self.candidates = np.load(os.path.join(stats_path, f"{self.split}_mlm_candidates.npy"), mmap_mode = "r")
        self.offsets = np.load(os.path.join(stats_path, f"{self.split}_mlm_offsets.npy"))
        self.fallbacks = 0
        self.replacements = 0

    def extract_rationale(self, review, indices, replacement_mask, doc_id = None):
        rationale = [review[i] for i in indices]
        if doc_id + 1 >= len(self.offsets):
            raise ValueError(f"No MLM candidates for document {doc_id}, rebuild them")
        offset = self.offsets[doc_id]
        if self.offsets[doc_id + 1] - offset != len(review):
            raise ValueError(f"MLM candidates of document {doc_id} do not match its length, rebuild them")
        for position in np.flatnonzero(replacement_mask):
            candidates = self.candidates[offset + indices[position]]
            # 0 (= PAD) marks an empty candidate slot
            candidates = candidates[candidates != 0]
            self.replacements += 1
            if len(candidates) == 0:
                # no in-context substitute survived filtering, use vocabulary noise
                self.fallbacks += 1
                rationale[position] = str(self.rng.choice(self.vocab, p = self.scores))
            else:
                rationale[position] = self.tokenizer.convert_ids_to_tokens(int(self.rng.choice(candidates)))
        return rationale, replacement_mask.sum()/replacement_mask.shape[0]



@dataclass
class RationaleExtractorFactory:
    tokenizer: PreTrainedTokenizerBase
    device: str
    data_path: Optional[str] = None
    seed: Optional[int] = None
    legacy_alignment: bool = False

    def create_extractor(self, inject_noise, noise_source = 'vocab'):
        if inject_noise and noise_source == 'mlm':
            return MLMNoisyRationaleExtractor(
                tokenizer = self.tokenizer,
                device = self.device,
                data_path = self.data_path,
                seed = self.seed,
                legacy_alignment = self.legacy_alignment
            )
        if inject_noise:
            return RandomNoisyRationaleExtractor(
                tokenizer = self.tokenizer,
                device = self.device,
                data_path = self.data_path,
                seed = self.seed,
                legacy_alignment = self.legacy_alignment
            )
        return RationaleExtractor(
            tokenizer = self.tokenizer,
            device = self.device
        )


@dataclass
class TopkSelector:
    sparsity: float
    max_length: int
    pad_token_id: int
    device: str
    # tokens are ranked by their attention averaged over a window of this
    # many tokens, which selects phrases instead of scattered tokens
    window: int = 1

    # gets empty mask
    def get_mask(self, token_att):
        return torch.zeros_like(token_att, dtype=torch.bool, requires_grad=False, device=self.device)

    # gets k = how many tokens to select
    def get_k(self, input_ids: torch.Tensor, selectable: Optional[torch.Tensor] = None) -> torch.Tensor:
        if selectable is not None:
            # count real tokens only
            seq_lens = selectable.sum(dim = 1)
        else:
            seq_lens = torch.sum(input_ids != self.pad_token_id, dim = 1)
            seq_lens[seq_lens > self.max_length - 1] = self.max_length - 1 # - CLS token
        k = torch.round(seq_lens * self.sparsity).type(torch.long)
        k[k < 1] = 1 # select at least 1 token
        return k

    # gets scores to rank tokens by, SEP and PAD tokens always rank last
    def get_scores(self, token_att: torch.Tensor, selectable: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = token_att.squeeze(-1).detach()
        if self.window > 1:
            if selectable is not None:
                scores = scores.masked_fill(~selectable, 0.0)
            scores = F.avg_pool1d(scores.unsqueeze(1), kernel_size = self.window, stride = 1, padding = self.window // 2, count_include_pad = False).squeeze(1)
        if selectable is not None:
            scores = scores.masked_fill(~selectable, -1.0)
        return scores

    def __call__(self, token_att: torch.Tensor, input_ids: torch.Tensor, selectable: Optional[torch.Tensor] = None) -> torch.Tensor:
        hard_mask = self.get_mask(token_att)
        k = self.get_k(input_ids, selectable)
        scores = self.get_scores(token_att, selectable)
        # batch-first
        for i in range(token_att.shape[0]):
            _, indices = scores[i].topk(k = k[i], dim = -1)
            hard_mask[i, indices, :] = True
        return hard_mask


@dataclass
class RandomSelector(TopkSelector):
    # Control: selects the same number of tokens as TopkSelector uniformly at
    # random among real tokens. Gives the plausibility floor at matched k.
    seed: Optional[int] = None

    def __post_init__(self):
        self.generator = torch.Generator().manual_seed(0 if self.seed is None else self.seed)

    def get_scores(self, token_att: torch.Tensor, selectable: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.rand(token_att.shape, generator = self.generator).to(token_att.device)
        return super().get_scores(scores, selectable)


@dataclass
class TopkContiguousSpanSelector(TopkSelector):
    def __call__(self, token_att: torch.Tensor, input_ids: torch.Tensor, selectable: Optional[torch.Tensor] = None) -> torch.Tensor:
        hard_mask = self.get_mask(token_att)
        k = self.get_k(input_ids, selectable)
        # batch-first
        for i in range(token_att.shape[0]):
            atts = token_att[i, :, :].unsqueeze(0).swapdims(2, 1)
            filter = torch.ones((1, 1, k[i]), device = self.device)
            start = F.conv1d(atts, filter).argmax()
            hard_mask[i, start:start + k[i], :] = True
        return hard_mask


@dataclass
class TopkSentenceSelector(TopkSelector):
    def get_num_sentences(self, sentence_mask: list):
        # batch-first - select last sentence index (= num_sentences - 1) from each seq in a batch
        return [sent_mask[-1] + 1 for sent_mask in sentence_mask]

    def get_k_sentences(self, sentence_mask: list):
        num_sents = torch.tensor(self.get_num_sentences(sentence_mask))
        k_sentences = torch.round(num_sents * self.sparsity).type(torch.long)
        k_sentences[k_sentences < 1] = 1 # select at least 1 sentence
        return k_sentences

    def map_to_tokens(self, sent_mask: list, word_ids: list):
        return torch.tensor([sent_mask[i] if i is not None else -1 for i in word_ids[1:]], dtype=torch.long) # word_ids[1:] -> - CLS token

    def __call__(self, token_att: torch.Tensor, reviews_tokenized, sentence_mask: list) -> torch.Tensor:
        hard_mask = self.get_mask(token_att)
        k = self.get_k_sentences(sentence_mask)
        num_sents = self.get_num_sentences(sentence_mask)
        # batch-first
        for i in range(token_att.shape[0]):
            atts = token_att[i, :, :].squeeze().detach()
            sent_mask = self.map_to_tokens(sentence_mask[i], reviews_tokenized.word_ids(i))
            sent_atts = torch.tensor([torch.sum(atts[sent_mask==i]) for i in range(num_sents[i])])
            _, sent_indices = sent_atts.topk(k = k[i], dim = -1)
            indices = torch.tensor([i for i in range(len(atts)) if torch.any(sent_indices == sent_mask[i])])
            hard_mask[i, indices, :] = True
        return hard_mask


@dataclass
class SelectorFactory:
    sparsity: float
    max_length: int
    pad_token_id: int
    device: str
    seed: Optional[int] = None
    window: int = 1

    def create_selector(self, selection_method):
        if self.window % 2 == 0:
            raise ValueError('The selection window must be an odd number of tokens')
        if selection_method == 'random':
            return RandomSelector(
                sparsity = self.sparsity,
                max_length = self.max_length,
                pad_token_id = self.pad_token_id,
                device = self.device,
                window = self.window,
                seed = self.seed
            )
        if selection_method == 'words':
            return TopkSelector(
                sparsity = self.sparsity,
                max_length = self.max_length,
                pad_token_id = self.pad_token_id,
                device = self.device,
                window = self.window
            )
        elif selection_method == 'span':
            return TopkContiguousSpanSelector(
                sparsity = self.sparsity,
                max_length = self.max_length,
                pad_token_id = self.pad_token_id,
                device = self.device
            )
        raise ValueError(f'Unknown selection method {selection_method}')
