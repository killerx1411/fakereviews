"""Stage 1: genuine vs fake classification.

TF-IDF word n-grams + scaled dense features (stylometry, perplexity, rating-sentiment
mismatch) -> logistic regression. A linear model keeps every decision explainable:
each n-gram's contribution (tf-idf value x coefficient) tells us which phrases pushed
the review towards "fake", and each dense feature's contribution gives a plain-English
reason.

Hook for later: `transformer_scorer` can be any callable texts -> P(fake). When it is
set, its probability is averaged with the linear model's (simple late fusion), so the
fine-tuned transformer can be dropped in without touching the rest of the pipeline.
"""
from dataclasses import dataclass, field

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .features import DENSE_FEATURES, STOPWORDS, PerplexityScorer, dense_features
from .sentiment import SentimentScorer

# Plain-English reason for a dense feature, depending on whether its value is high or low.
REASONS = {
    "log_perplexity": ("Unusually unpredictable wording", "Very predictable, template-like wording (low perplexity)"),
    "rating_mismatch": ("Star rating contradicts the sentiment of the text", "Rating agrees with the text"),
    "rating_extremity": ("Extreme star rating (1 or 5)", "Moderate star rating"),
    "sentiment": ("Very positive tone", "Very negative tone"),
    "superlative_ratio": ("Heavy use of superlatives and hype words", "Few superlatives"),
    "hedge_ratio": ("Lots of hedging and caveats", "No hedging or caveats at all"),
    "first_person_ratio": ("Many first-person references", "Little first-person experience (few 'I', 'my')"),
    "second_person_ratio": ("Addresses the reader directly ('you will love...')", "Does not address the reader"),
    "type_token_ratio": ("Varied vocabulary", "Repetitive vocabulary"),
    "avg_sent_len": ("Long, polished sentences", "Short sentences"),
    "sent_len_std": ("Uneven sentence lengths", "Uniform sentence lengths"),
    "avg_word_len": ("Long words", "Short words"),
    "n_words": ("Long review", "Very short review"),
    "n_sentences": ("Many sentences", "Few sentences"),
    "exclamations": ("Many exclamation marks", "No exclamation marks"),
    "questions": ("Contains questions", "No questions"),
    "punct_ratio": ("Heavy punctuation", "Sparse punctuation"),
    "upper_ratio": ("Lots of capital letters", "Few capital letters"),
    "digit_ratio": ("Mentions specific numbers", "No concrete numbers or specifics"),
    "lowercase_sentence_starts": ("Casual typing (lowercase sentence starts)", "Perfectly capitalised sentences"),
    "repeated_chars": ("Repeated characters ('sooo')", "No informal spelling"),
    "stopword_ratio": ("Many function words", "Few function words"),
}


@dataclass
class Explanation:
    fake_prob: float
    phrases: list = field(default_factory=list)  # [(phrase, weight)] pushing towards fake
    reasons: list = field(default_factory=list)  # [(text, weight)] pushing towards fake


class FakeReviewClassifier:
    def __init__(self, perplexity_backend: str = "auto", sentiment_backend: str = "auto",
                 C: float = 2.0, max_features: int = 50000, dense_weight: float = 1.0):
        self.perplexity = PerplexityScorer(perplexity_backend)
        self.sentiment = SentimentScorer(sentiment_backend)
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True,
                                     max_features=max_features, token_pattern=r"(?u)\b\w+(?:'\w+)?\b")
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=C, max_iter=3000, class_weight="balanced", solver="liblinear")
        self.dense_weight = dense_weight
        self.transformer_scorer = None  # optional callable(texts) -> P(fake)
        self.use_text = self.use_dense = True

    # ------------------------------------------------------------------ helpers
    def _matrix(self, texts, ratings, fit=False, ppl=None):
        blocks = []
        if self.use_text:
            blocks.append(self.tfidf.fit_transform(texts) if fit else self.tfidf.transform(texts))
        if self.use_dense:
            d = dense_features(texts, ratings, self.perplexity, self.sentiment, ppl)
            d = self.scaler.fit_transform(d) if fit else self.scaler.transform(d)
            blocks.append(sparse.csr_matrix(d * self.dense_weight))
        return sparse.hstack(blocks).tocsr()

    @property
    def n_text(self):
        return len(self.tfidf.vocabulary_) if self.use_text else 0

    # ------------------------------------------------------------------ API
    def fit(self, texts, ratings, labels):
        """labels: 1 = fake / computer-generated, 0 = genuine."""
        texts = [str(t) for t in texts]
        labels = np.asarray(labels)
        # the bigram perplexity fallback is trained on genuine text only
        self.perplexity.fit([t for t, y in zip(texts, labels) if y == 0])
        ppl = self.perplexity.cross_fit_score(texts, labels) if self.use_dense else None
        self.model.fit(self._matrix(texts, ratings, fit=True, ppl=ppl), labels)
        return self

    def predict_proba(self, texts, ratings=None) -> np.ndarray:
        texts = [str(t) for t in texts]
        p = self.model.predict_proba(self._matrix(texts, ratings))[:, 1]
        if self.transformer_scorer is not None:
            p = (p + np.asarray(self.transformer_scorer(texts))) / 2.0
        return p

    def predict(self, texts, ratings=None, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(texts, ratings) >= threshold).astype(int)

    def explain_batch(self, texts, ratings=None, top_phrases: int = 6, top_reasons: int = 3):
        texts = [str(t) for t in texts]
        ratings = list(ratings) if ratings is not None else [None] * len(texts)
        X = self._matrix(texts, ratings)
        probs = self.predict_proba(texts, ratings)
        coef = self.model.coef_[0]
        vocab = self.tfidf.get_feature_names_out() if self.use_text else []
        n_text = self.n_text
        out = []
        for i in range(len(texts)):
            row = X.getrow(i)
            contrib = row.multiply(coef).tocsr()
            phrases, reasons = [], []
            for j, v in zip(contrib.indices, contrib.data):
                if v <= 0:
                    continue
                if j < n_text:
                    # function words still count in the model, but highlighting them explains nothing
                    if not all(w in STOPWORDS for w in vocab[j].split()):
                        phrases.append((vocab[j], float(v)))
                else:
                    name = DENSE_FEATURES[j - n_text]
                    high = row[0, j] > 0  # standardised value above the training mean
                    hi_txt, lo_txt = REASONS.get(name, (name + " (high)", name + " (low)"))
                    reasons.append((hi_txt if high else lo_txt, float(v)))
            phrases = _drop_overlaps(sorted(phrases, key=lambda x: -x[1]))[:top_phrases]
            reasons = sorted(reasons, key=lambda x: -x[1])[:top_reasons]
            out.append(Explanation(float(probs[i]), phrases, reasons))
        return out

    def explain(self, text, rating=None, **kw) -> Explanation:
        return self.explain_batch([text], [rating], **kw)[0]

    def top_global_ngrams(self, k: int = 25):
        """Most indicative n-grams for each class: useful for the report."""
        coef = self.model.coef_[0][: self.n_text]
        vocab = self.tfidf.get_feature_names_out()
        order = np.argsort(coef)
        return ([(vocab[i], coef[i]) for i in order[::-1][:k]],
                [(vocab[i], coef[i]) for i in order[:k]])

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "FakeReviewClassifier":
        return joblib.load(path)


def _drop_overlaps(phrases):
    """Keep 'highly recommend' and drop 'highly' / 'recommend' when both score."""
    kept = []
    for p, w in phrases:
        toks = set(p.split())
        if any(toks <= set(k.split()) or set(k.split()) <= toks for k, _ in kept):
            continue
        kept.append((p, w))
    return kept
