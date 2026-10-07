"""Hand-crafted features for the fake review classifier.

* Stylometric cues: length, vocabulary richness, punctuation, pronoun use, superlatives...
* Perplexity: how predictable the wording is. Machine-generated text tends to have
  low perplexity under a language model. Uses GPT-2 (distilgpt2) when
  transformers + torch are installed, otherwise an interpolated bigram LM trained
  on the genuine training reviews (cheaper, weaker, but dependency-free).
* Rating-sentiment mismatch: a 5-star rating on a negative text (or vice versa).
"""
import math
import re
from collections import Counter

import numpy as np

from .sentiment import SentimentScorer
from .text_utils import sentences, tokens_lower, words

FIRST_PERSON = {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'd", "i'll"}
SECOND_PERSON = {"you", "your", "yours", "you'll", "you're"}
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for", "with", "at",
    "by", "from", "is", "are", "was", "were", "be", "been", "it", "this", "that", "these",
    "those", "as", "so", "than", "too", "very", "can", "will", "just", "have", "has", "had",
    "do", "does", "did", "not", "no", "i", "you", "he", "she", "they", "we", "my", "your",
}
SUPERLATIVE_WORDS = {
    "best", "perfect", "amazing", "excellent", "outstanding", "incredible", "fantastic",
    "exceptional", "flawless", "superb", "phenomenal", "must-have", "exceeded", "highly",
    "absolutely", "definitely", "truly", "everyone", "anyone",
}
HEDGES = {"maybe", "probably", "guess", "think", "seems", "kinda", "sorta", "though", "but", "although"}

STYLO_FEATURES = [
    "n_words", "n_sentences", "avg_word_len", "avg_sent_len", "sent_len_std",
    "type_token_ratio", "stopword_ratio", "first_person_ratio", "second_person_ratio",
    "superlative_ratio", "hedge_ratio", "exclamations", "questions", "punct_ratio",
    "upper_ratio", "digit_ratio", "lowercase_sentence_starts", "repeated_chars",
]


def stylometric(text: str) -> dict:
    text = str(text)
    ws = words(text)
    lw = [w.lower() for w in ws]
    n = max(len(ws), 1)
    sents = sentences(text) or [text]
    sent_lens = [len(words(s)) for s in sents]
    letters = sum(c.isalpha() for c in text) or 1
    return {
        "n_words": math.log1p(len(ws)),
        "n_sentences": len(sents),
        "avg_word_len": float(np.mean([len(w) for w in ws])) if ws else 0.0,
        "avg_sent_len": float(np.mean(sent_lens)),
        "sent_len_std": float(np.std(sent_lens)),
        "type_token_ratio": len(set(lw)) / n,
        "stopword_ratio": sum(w in STOPWORDS for w in lw) / n,
        "first_person_ratio": sum(w in FIRST_PERSON for w in lw) / n,
        "second_person_ratio": sum(w in SECOND_PERSON for w in lw) / n,
        "superlative_ratio": sum(w in SUPERLATIVE_WORDS for w in lw) / n,
        "hedge_ratio": sum(w in HEDGES for w in lw) / n,
        "exclamations": text.count("!"),
        "questions": text.count("?"),
        "punct_ratio": sum(c in ",;:-()\"'" for c in text) / n,
        "upper_ratio": sum(c.isupper() for c in text) / letters,
        "digit_ratio": sum(c.isdigit() for c in text) / max(len(text), 1),
        "lowercase_sentence_starts": sum(s[:1].islower() for s in sents) / len(sents),
        "repeated_chars": len(re.findall(r"(.)\1{2,}", text)),
    }


class PerplexityScorer:
    """Returns log-perplexity per text (lower = more predictable wording)."""

    def __init__(self, backend: str = "auto", model_name: str = "distilbert/distilgpt2", max_tokens: int = 256):
        self.requested = backend
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.backend = None
        self._model = self._tok = None
        self.unigrams, self.bigrams, self.total = Counter(), Counter(), 0
        self._init_backend()

    def _init_backend(self):
        if self.requested in ("auto", "gpt2"):
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self._tok = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForCausalLM.from_pretrained(self.model_name).eval()
                self._model.to("cuda" if torch.cuda.is_available() else "cpu")
                self.backend = "gpt2"
                return
            except Exception:
                if self.requested == "gpt2":
                    raise
        self.backend = "bigram"

    # --- bigram fallback -------------------------------------------------
    def fit(self, texts):
        if self.backend != "bigram":
            return self
        for t in texts:
            toks = ["<s>"] + tokens_lower(t) + ["</s>"]
            self.unigrams.update(toks)
            self.bigrams.update(zip(toks, toks[1:]))
        self.total = sum(self.unigrams.values())
        return self

    def cross_fit_score(self, texts, labels, k: int = 5) -> np.ndarray:
        """Training-set scores from bigram LMs that never saw the scored review.

        Scoring a review with an LM trained on it makes genuine training reviews look far
        more predictable than unseen ones, and the classifier learns that artefact.
        """
        texts = list(texts)
        if self.backend != "bigram":
            return self.score(texts)
        labels = np.asarray(labels)
        folds = np.random.default_rng(0).permutation(len(texts)) % k
        out = np.empty(len(texts))
        for f in range(k):
            lm = PerplexityScorer("bigram").fit([t for t, y, g in zip(texts, labels, folds) if y == 0 and g != f])
            idx = np.flatnonzero(folds == f)
            out[idx] = lm.score([texts[i] for i in idx])
        return out

    def _bigram_logppl(self, text: str, lam: float = 0.7) -> float:
        toks = ["<s>"] + tokens_lower(text) + ["</s>"]
        v = len(self.unigrams) + 1
        nll = 0.0
        for prev, cur in zip(toks, toks[1:]):
            p_uni = (self.unigrams[cur] + 1) / (self.total + v)
            c_prev = self.unigrams[prev]
            p_bi = self.bigrams[(prev, cur)] / c_prev if c_prev else 0.0
            nll -= math.log(lam * p_bi + (1 - lam) * p_uni)
        return nll / max(len(toks) - 1, 1)

    # --- GPT-2 -------------------------------------------------------------
    def _gpt2_logppl(self, texts, batch_size: int = 8):
        import torch

        out = []
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        for i in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[i:i + batch_size]]
            enc = self._tok(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=self.max_tokens).to(self._model.device)
            with torch.no_grad():
                logits = self._model(**enc).logits[:, :-1]
            target = enc["input_ids"][:, 1:]
            mask = enc["attention_mask"][:, 1:].float()
            nll = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), target, reduction="none")
            out += ((nll * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().tolist()
        return out

    def score(self, texts) -> np.ndarray:
        texts = list(texts)
        if self.backend == "gpt2":
            return np.array(self._gpt2_logppl(texts))
        return np.array([self._bigram_logppl(t) for t in texts])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = state["_tok"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.backend == "gpt2":
            self.requested = "gpt2"
            self._init_backend()


def rating_mismatch(rating, sentiment: float) -> float:
    """|normalised rating - text sentiment| / 2, in [0, 1]. 0 when the rating is unknown."""
    try:
        r = float(rating)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(r):
        return 0.0
    return abs((r - 3.0) / 2.0 - sentiment) / 2.0


DENSE_FEATURES = STYLO_FEATURES + ["log_perplexity", "sentiment", "rating_extremity", "rating_mismatch"]


def dense_features(texts, ratings, perplexity: PerplexityScorer, sentiment: SentimentScorer,
                   ppl=None) -> np.ndarray:
    """ppl: precomputed log-perplexities (e.g. out-of-fold ones for the training set)."""
    texts = [str(t) for t in texts]
    ratings = list(ratings) if ratings is not None else [None] * len(texts)
    ppl = perplexity.score(texts) if ppl is None else ppl
    rows = []
    for t, r, p in zip(texts, ratings, ppl):
        s = sentiment(t)
        st = stylometric(t)
        try:
            extremity = abs(float(r) - 3.0) / 2.0
            extremity = 0.0 if math.isnan(extremity) else extremity
        except (TypeError, ValueError):
            extremity = 0.0
        rows.append([st[k] for k in STYLO_FEATURES] + [p, s, extremity, rating_mismatch(r, s)])
    return np.asarray(rows, dtype=float)
