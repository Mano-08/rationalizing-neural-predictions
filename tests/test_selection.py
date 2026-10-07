import os
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models import (MLMNoisyRationaleExtractor, RandomNoisyRationaleExtractor,
                    RandomSelector, RationaleExtractor, TopkSelector,
                    get_selectable)

REVIEWS = [
    ["the", "movie", "was", "unbelievably", "good", "and", "great", ",", "a", "wonderful", "story", "."],
    ["bad", "plot", "."],
]


def make_batch(tokenizer, reviews = REVIEWS, max_length = 32, **kwargs):
    reviews_tokenized = tokenizer(text = list(reviews), is_split_into_words = True, padding = True, truncation = True, max_length = max_length, return_tensors = 'pt')
    return SimpleNamespace(reviews = reviews, reviews_tokenized = reviews_tokenized, doc_ids = None, **kwargs)


def make_selector(cls = TopkSelector, sparsity = 0.2, **kwargs):
    return cls(sparsity = sparsity, max_length = 32, pad_token_id = 0, device = "cpu", **kwargs)


def test_selectable_covers_real_tokens_only(tokenizer):
    batch = make_batch(tokenizer)
    selectable = get_selectable(batch.reviews_tokenized, tokenizer)
    for i in range(len(REVIEWS)):
        expected = [word_id is not None for word_id in batch.reviews_tokenized.word_ids(i)][1:]
        assert selectable[i].tolist() == expected
    # "unbelievably" is 4 wordpieces: 12 words -> 15 wordpieces
    assert selectable.sum(-1).tolist() == [15, 3]


def test_topk_never_selects_special_tokens_when_masked(tokenizer):
    batch = make_batch(tokenizer)
    selectable = get_selectable(batch.reviews_tokenized, tokenizer)
    # attention that prefers SEP and PAD over every real token
    token_att = torch.where(selectable, 0.01, 1.0).unsqueeze(-1)
    hard_mask = make_selector()(token_att = token_att, input_ids = batch.reviews_tokenized.input_ids, selectable = selectable).squeeze(-1)
    assert not (hard_mask & ~selectable).any()
    # exactly round(20% of the real wordpieces), at least one
    assert hard_mask.sum(-1).tolist() == [3, 1]
    # the original behavior spends budget on SEP/PAD in every document here
    legacy_mask = make_selector()(token_att = token_att, input_ids = batch.reviews_tokenized.input_ids).squeeze(-1)
    assert (legacy_mask & ~selectable).any(-1).all()


def test_topk_without_mask_matches_the_original_selector(tokenizer):
    batch = make_batch(tokenizer)
    input_ids = batch.reviews_tokenized.input_ids
    torch.manual_seed(0)
    token_att = torch.softmax(torch.randn(input_ids.shape[0], input_ids.shape[1] - 1), -1).unsqueeze(-1)
    hard_mask = make_selector(sparsity = 0.3)(token_att = token_att, input_ids = input_ids)
    for i in range(input_ids.shape[0]):
        k = max(int(torch.round((input_ids[i] != 0).sum() * 0.3)), 1)
        expected = torch.zeros(input_ids.shape[1] - 1, dtype = torch.bool)
        expected[token_att[i].squeeze().topk(k).indices] = True
        assert hard_mask[i].squeeze(-1).tolist() == expected.tolist()


def test_random_selector_matches_k_and_is_reproducible(tokenizer):
    batch = make_batch(tokenizer)
    selectable = get_selectable(batch.reviews_tokenized, tokenizer)
    token_att = torch.zeros(selectable.shape).unsqueeze(-1)
    kwargs = dict(token_att = token_att, input_ids = batch.reviews_tokenized.input_ids, selectable = selectable)
    topk_mask = make_selector()(**kwargs)
    random_mask = make_selector(RandomSelector, seed = 1)(**kwargs)
    assert random_mask.sum((1, 2)).tolist() == topk_mask.sum((1, 2)).tolist()
    assert not (random_mask.squeeze(-1) & ~selectable).any()
    assert torch.equal(random_mask, make_selector(RandomSelector, seed = 1)(**kwargs))
    draws = [make_selector(RandomSelector, seed = seed)(**kwargs) for seed in range(8)]
    assert any(not torch.equal(draws[0], draw) for draw in draws[1:])


def test_random_mask_precision_is_the_share_of_annotated_tokens(tokenizer):
    # the control of the whole evaluation: a random rationale must score the
    # base rate of human-annotated tokens, no more and no less
    from metrics import plausibility, reference_points, selection_counts
    rng = np.random.default_rng(0)
    reviews = [rng.choice(["the", "movie", "unbelievably", "good", "plot"], size = int(rng.integers(5, 40))).tolist() for _ in range(300)]
    batch = make_batch(tokenizer, reviews)
    selectable = get_selectable(batch.reviews_tokenized, tokenizer)
    hard_mask = make_selector(RandomSelector, sparsity = 0.3, seed = 0)(
        token_att = torch.zeros(selectable.shape).unsqueeze(-1),
        input_ids = batch.reviews_tokenized.input_ids,
        selectable = selectable
    )
    counts = []
    for i, review in enumerate(reviews):
        gold_words = set(range(2, 2 + len(review)//4))
        counts.append(selection_counts(batch.reviews_tokenized.word_ids(i), [False] + hard_mask[i].squeeze(-1).tolist(), gold_words))
    observed = plausibility(counts)["wordpiece"]["micro"]
    expected = reference_points(counts)["random"]
    assert observed["prec"] == pytest.approx(expected["prec"], abs = 0.02)
    assert observed["rec"] == pytest.approx(0.3, abs = 0.03)


def test_plain_extractor_keeps_selected_tokens_in_order(tokenizer):
    batch = make_batch(tokenizer)
    input_ids = batch.reviews_tokenized.input_ids
    hard_mask = torch.zeros(input_ids.shape[0], input_ids.shape[1] - 1, 1, dtype = torch.bool)
    hard_mask[0, [1, 4, 7]] = True
    hard_mask[1, 0] = True
    before = hard_mask.clone()
    rationale, remainder, _ = RationaleExtractor(tokenizer, "cpu")(batch, hard_mask)
    assert torch.equal(hard_mask, before)
    assert tokenizer.convert_ids_to_tokens(rationale["input_ids"][0]) == ["[CLS]", "movie", "##bel", "good", "[SEP]"]
    assert tokenizer.convert_ids_to_tokens(rationale["input_ids"][1]) == ["[CLS]", "bad", "[SEP]", "[PAD]", "[PAD]"]
    assert rationale["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]
    # rationale and remainder partition the real tokens
    assert remainder["attention_mask"].sum().item() + rationale["attention_mask"].sum().item() == batch.reviews_tokenized.attention_mask.sum().item() + 4


def make_noisy_batch(tokenizer, data_path, num_documents = 16):
    with open(os.path.join(data_path, "word_statistics", "train_replacement_probs.pkl"), "rb") as f:
        replacement_probs = pickle.load(f)[:num_documents]
    import json
    with open(os.path.join(data_path, "train.jsonl")) as f:
        reviews = [json.loads(line)[0] for line in f.read().splitlines()[:num_documents]]
    batch = make_batch(tokenizer, reviews, replacement_probs = replacement_probs)
    batch.doc_ids = tuple(range(num_documents))
    selectable = get_selectable(batch.reviews_tokenized, tokenizer)
    return batch, selectable.unsqueeze(-1).clone()


def test_noise_level_controls_the_share_of_replaced_words(tokenizer, data_path):
    batch, hard_mask = make_noisy_batch(tokenizer, data_path)
    extractor = RandomNoisyRationaleExtractor(tokenizer = tokenizer, device = "cpu", data_path = data_path, seed = 0)
    before = hard_mask.clone()
    extractor.noise_p = 0.0
    clean, ratio = extractor.extract_from_mask_with_replacement(batch, hard_mask)
    assert ratio == 0
    assert torch.equal(hard_mask, before)
    # selecting every token without noise returns every word that survived
    # truncation (whole words, even if truncation cut one in the middle)
    surviving = [[review[word] for word in sorted({word_id for word_id in batch.reviews_tokenized.word_ids(i) if word_id is not None})] for i, review in enumerate(batch.reviews)]
    assert torch.equal(clean.input_ids, tokenizer(surviving, is_split_into_words = True, padding = True, return_tensors = 'pt').input_ids)
    ratios = []
    for noise_p in (0.1, 0.3, 0.6):
        extractor.noise_p = noise_p
        ratios.append(np.mean([extractor.extract_from_mask_with_replacement(batch, hard_mask)[1] for _ in range(30)]))
    # replacement probabilities average to 1 over a document, so the share
    # of replaced words tracks p until probabilities start to clip at 1
    assert ratios[0] == pytest.approx(0.1, abs = 0.03)
    assert ratios[1] == pytest.approx(0.3, abs = 0.05)
    assert ratios[0] < ratios[1] < ratios[2] <= 0.6 + 0.03


def test_predictor_receives_the_selected_tokens(tokenizer, data_path):
    review = ["the", "plot", "was", "bad", "but", "the", "story", "was", "good", "."]
    batch = make_batch(tokenizer, [review, review + ["."]], replacement_probs = [np.ones(10), np.ones(11)])
    tokens = tokenizer.convert_ids_to_tokens(batch.reviews_tokenized.input_ids[0])
    # hard_mask has no CLS token: position j is token j + 1
    hard_mask = torch.zeros(2, len(tokens) - 1, 1, dtype = torch.bool)
    hard_mask[0, [tokens.index("bad") - 1, tokens.index("good") - 1]] = True
    hard_mask[1, 0] = True
    def extract(**kwargs):
        extractor = RandomNoisyRationaleExtractor(tokenizer = tokenizer, device = "cpu", data_path = data_path, seed = 0, **kwargs)
        tokenized, _ = extractor.extract_from_mask_with_replacement(batch, hard_mask)
        return [token for token in tokenizer.convert_ids_to_tokens(tokenized.input_ids[0]) if token not in ("[CLS]", "[SEP]", "[PAD]")]
    plain = RationaleExtractor(tokenizer, "cpu").extract_from_mask(batch, hard_mask)
    assert tokenizer.convert_ids_to_tokens(plain["input_ids"][0]) == ["[CLS]", "bad", "good", "[SEP]"]
    # training with noise injection sees what validation and test see
    assert extract() == ["bad", "good"]
    # the released code shifted the rationale by one token to the left
    assert extract(legacy_alignment = True) == ["was", "was"]


def test_noise_level_set_per_step_equals_noise_scaled_in_advance(tokenizer, data_path):
    # the original code multiplied the probabilities by p once, when loading
    batch, hard_mask = make_noisy_batch(tokenizer, data_path)
    per_step = RandomNoisyRationaleExtractor(tokenizer = tokenizer, device = "cpu", data_path = data_path, seed = 3)
    per_step.noise_p = 0.25
    in_advance = RandomNoisyRationaleExtractor(tokenizer = tokenizer, device = "cpu", data_path = data_path, seed = 3)
    in_advance.noise_p = 1.0
    scaled = make_noisy_batch(tokenizer, data_path)[0]
    scaled.replacement_probs = [np.minimum(probs * 0.25, 1) for probs in scaled.replacement_probs]
    a, ratio_a = per_step.extract_from_mask_with_replacement(batch, hard_mask)
    b, ratio_b = in_advance.extract_from_mask_with_replacement(scaled, hard_mask)
    assert ratio_a == ratio_b > 0
    assert torch.equal(a.input_ids, b.input_ids)


def test_mlm_extractor_uses_cached_substitutes(tokenizer, data_path, tmp_path):
    batch, hard_mask = make_noisy_batch(tokenizer, data_path)
    stats_path = tmp_path / "word_statistics"
    os.makedirs(stats_path)
    with open(os.path.join(data_path, "word_statistics", "scored_vocab.pkl"), "rb") as f:
        scored_vocab = pickle.load(f)
    with open(stats_path / "scored_vocab.pkl", "wb") as f:
        pickle.dump(scored_vocab, f)
    # the most informative word of a document is never replaced under
    # TF*IDF, make every word replaceable here
    batch.replacement_probs = [np.ones(len(review)) for review in batch.reviews]
    lengths = [len(review) for review in batch.reviews]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    film, very = tokenizer.convert_tokens_to_ids(["film", "very"])
    candidates = np.zeros((offsets[-1], 3), dtype = np.uint16)
    # even words have substitutes, odd words have none
    for offset, length in zip(offsets, lengths):
        candidates[offset:offset + length:2] = [film, very, 0]
    np.save(stats_path / "train_mlm_candidates.npy", candidates)
    np.save(stats_path / "train_mlm_offsets.npy", offsets)

    extractor = MLMNoisyRationaleExtractor(tokenizer = tokenizer, device = "cpu", data_path = str(tmp_path), seed = 0)
    extractor.noise_p = 10.0 # replace every word
    review = batch.reviews[0]
    indices = list(range(len(review)))
    rationale, ratio = extractor.extract_rationale(review, indices, np.ones(len(review), dtype = int), doc_id = 0)
    assert ratio == 1
    assert set(rationale[0::2]) == {"film", "very"}
    # words without substitutes fall back to vocabulary noise
    assert extractor.fallbacks == len(review)//2
    assert all(word in set(scored_vocab[0]) for word in rationale[1::2])
    # the whole batch goes through and stays tokenizable
    tokenized, ratio = extractor.extract_from_mask_with_replacement(batch, hard_mask)
    assert ratio == 1 and tokenized.input_ids.shape[0] == len(batch.reviews)
    with pytest.raises(ValueError):
        extractor.extract_rationale(review + ["extra"], indices, np.ones(len(review), dtype = int), doc_id = 0)
    with pytest.raises(ValueError):
        extractor.extract_rationale(review, indices, np.ones(len(review), dtype = int), doc_id = len(lengths))


def test_selection_window_selects_phrases(tokenizer):
    batch = make_batch(tokenizer)
    selectable = get_selectable(batch.reviews_tokenized, tokenizer)
    # two isolated peaks in the first document
    token_att = torch.full(selectable.shape, 0.001)
    token_att[0, 3] = 0.3
    token_att[0, 11] = 0.3
    kwargs = dict(token_att = token_att.unsqueeze(-1), input_ids = batch.reviews_tokenized.input_ids, selectable = selectable)
    scattered = make_selector(sparsity = 0.4)(**kwargs).squeeze(-1)
    phrases = make_selector(sparsity = 0.4, window = 3)(**kwargs).squeeze(-1)
    # same number of tokens, never a special token
    assert scattered.sum(-1).tolist() == phrases.sum(-1).tolist() == [6, 1]
    assert not (phrases & ~selectable).any()
    # the window turns the two peaks into two phrases of three tokens
    assert phrases[0].nonzero().flatten().tolist() == [2, 3, 4, 10, 11, 12]
    assert scattered[0, [3, 11]].all() and not scattered[0, [2, 4, 10, 12]].all()
    with pytest.raises(ValueError):
        from models import SelectorFactory
        SelectorFactory(0.2, 32, 0, "cpu", window = 4).create_selector("words")


def test_truncation_side_keeps_word_indices(tokenizer):
    review = ["good"] * 20 + ["bad"] * 20
    try:
        tokenizer.truncation_side = "left"
        tail = make_batch(tokenizer, [review], max_length = 12)
    finally:
        tokenizer.truncation_side = "right"
    head = make_batch(tokenizer, [review], max_length = 12)
    # word indices still refer to the whole document, whichever end is kept
    assert [word_id for word_id in head.reviews_tokenized.word_ids(0) if word_id is not None] == list(range(10))
    assert [word_id for word_id in tail.reviews_tokenized.word_ids(0) if word_id is not None] == list(range(30, 40))


def test_generator_can_use_lower_layers_only(model_path, tokenizer):
    from models import BlackBoxPredictor
    batch = make_batch(tokenizer)
    torch.manual_seed(0)
    full = BlackBoxPredictor(num_labels = 2, model = model_path, freeze_encoder = False).eval()
    torch.manual_seed(0)
    shallow = BlackBoxPredictor(num_labels = 2, model = model_path, freeze_encoder = False, num_layers = 1).eval()
    assert len(full.encoder.encoder.layer) == 2 and len(shallow.encoder.encoder.layer) == 1
    with torch.no_grad():
        prediction, token_att = shallow(**batch.reviews_tokenized)
    assert prediction.shape == (2, 2) and token_att.shape == (2, batch.reviews_tokenized.input_ids.shape[1] - 1, 1)
    # the checkpoint of a shallow generator loads into a shallow generator only
    shallow.load_state_dict(shallow.state_dict())
    with pytest.raises(RuntimeError):
        shallow.load_state_dict(full.state_dict())


def test_exposure_measures_how_much_noise_targets_the_rationale(tokenizer, data_path):
    batch, hard_mask = make_noisy_batch(tokenizer, data_path)
    extractor = RandomNoisyRationaleExtractor(tokenizer = tokenizer, device = "cpu", data_path = data_path, seed = 0)
    # the whole document: exposure is the mean replacement probability, 1
    extractor.extract_from_mask_with_replacement(batch, hard_mask)
    assert extractor.exposure == pytest.approx(1.0, abs = 0.05)
    # only the word noise injection protects the most in every document
    protected = torch.zeros_like(hard_mask)
    for i, probs in enumerate(batch.replacement_probs):
        word = int(np.argmin(probs))
        positions = [position for position, word_id in enumerate(batch.reviews_tokenized.word_ids(i)[1:]) if word_id == word]
        if positions:
            protected[i, positions] = True
    extractor.extract_from_mask_with_replacement(batch, protected)
    assert extractor.exposure < 0.3


def test_coupling_gate_leaves_the_input_alone_and_carries_the_gradient(model_path, tokenizer, data_path):
    from models import BlackBoxPredictor, RationalePredictor
    batch, _ = make_noisy_batch(tokenizer, data_path, num_documents = 4)
    torch.manual_seed(0)
    bb_model = BlackBoxPredictor(num_labels = 2, model = model_path, freeze_encoder = False).eval()
    rp_model = RationalePredictor(num_labels = 2, model = model_path, freeze_encoder = False).eval()
    selectable = get_selectable(batch.reviews_tokenized, tokenizer)
    _, token_att = bb_model(**batch.reviews_tokenized, selectable = selectable)
    hard_mask = make_selector(sparsity = 0.3)(token_att = token_att, input_ids = batch.reviews_tokenized.input_ids, selectable = selectable)
    extractor = RandomNoisyRationaleExtractor(tokenizer = tokenizer, device = "cpu", data_path = data_path, seed = 0)
    extractor.noise_p = 0.3
    rationale, _, _ = extractor(batch, hard_mask)
    gate = extractor.get_gate(batch, token_att, rationale, weight = 1.0)
    # the predictor sees exactly the same input with and without the gate
    assert torch.equal(gate.detach(), torch.ones_like(gate))
    with torch.no_grad():
        plain = rp_model(**rationale)
    gated = rp_model(**rationale, gate = gate)
    assert torch.allclose(plain, gated, atol = 1e-6)
    # without the gate the predictor's loss cannot reach the generator
    assert not plain.requires_grad
    loss = torch.nn.functional.cross_entropy(gated, torch.tensor([0, 1, 0, 1]))
    gradient = torch.autograd.grad(loss, token_att)[0].squeeze(-1)
    # the gradient lands on the tokens of the selected words (the predictor
    # gets whole words) and nowhere else
    selected = torch.zeros_like(selectable)
    for i, words in enumerate(extractor.selected_words):
        word_ids = batch.reviews_tokenized.word_ids(i)[1:]
        selected[i] = torch.tensor([word_id in set(words) for word_id in word_ids])
    assert (selected >= (hard_mask.squeeze(-1) & selectable)).all()
    assert gradient[selected].abs().sum() > 0
    assert gradient[~selected].abs().sum() == 0
    # twice the weight, twice the gradient
    double = extractor.get_gate(batch, token_att, rationale, weight = 2.0)
    double_loss = torch.nn.functional.cross_entropy(rp_model(**rationale, gate = double), torch.tensor([0, 1, 0, 1]))
    assert torch.allclose(torch.autograd.grad(double_loss, token_att)[0].squeeze(-1), 2 * gradient, rtol = 1e-4, atol = 1e-8)


def test_evidence_noise_replaces_pooled_states_only(model_path, tokenizer, data_path):
    from models import BlackBoxPredictor, get_replacement_probs
    batch, _ = make_noisy_batch(tokenizer, data_path, num_documents = 4)
    torch.manual_seed(0)
    bb_model = BlackBoxPredictor(num_labels = 2, model = model_path, freeze_encoder = True).eval()
    probs = get_replacement_probs(batch, "cpu")
    # one probability per word token, -1 for SEP and PAD
    selectable = get_selectable(batch.reviews_tokenized, tokenizer)
    assert ((probs >= 0) == selectable).all()
    for i, replacement_probs in enumerate(batch.replacement_probs):
        word_ids = batch.reviews_tokenized.word_ids(i)[1:]
        assert probs[i, 0].item() == pytest.approx(replacement_probs[word_ids[0]])
    with torch.no_grad():
        clean_pred, clean_att = bb_model(**batch.reviews_tokenized)
        zero_pred, zero_att = bb_model(**batch.reviews_tokenized, evidence_noise = torch.where(probs >= 0, 0.0, -1.0))
        noisy_pred, noisy_att = bb_model(**batch.reviews_tokenized, evidence_noise = torch.where(probs >= 0, 1.0, -1.0))
    # the attention is computed on the clean input whatever the noise
    assert torch.equal(clean_att, zero_att) and torch.equal(clean_att, noisy_att)
    # no noise: same prediction; every state replaced: a different one
    assert torch.equal(clean_pred, zero_pred)
    assert not torch.allclose(clean_pred, noisy_pred)
    # the gradient of the noisy prediction still reaches the token scorer
    bb_model.train()
    prediction, _ = bb_model(**batch.reviews_tokenized, evidence_noise = torch.where(probs >= 0, 0.5, -1.0))
    prediction[:, 0].sum().backward()
    assert bb_model.token_predictor[1].weight.grad.abs().sum() > 0
    assert bb_model.encoder.embeddings.word_embeddings.weight.grad is None
