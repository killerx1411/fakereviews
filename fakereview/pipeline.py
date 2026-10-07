"""Glue: classify -> keep genuine -> summarise + aspect sentiment -> one report object."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .absa import AspectSentimentAnalyzer
from .classifier import FakeReviewClassifier
from .summarizer import OpinionSummarizer


@dataclass
class ReviewResult:
    idx: int
    text: str
    rating: float | None
    fake_prob: float
    is_fake: bool
    phrases: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    sentiment: float = 0.0


@dataclass
class ProductReport:
    reviews: list
    summary: dict
    aspects: list
    stats: dict

    @property
    def genuine(self):
        return [r for r in self.reviews if not r.is_fake]

    @property
    def fake(self):
        return [r for r in self.reviews if r.is_fake]

    def reviews_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "text": r.text, "rating": r.rating, "fake_prob": round(r.fake_prob, 3),
            "verdict": "Fake" if r.is_fake else "Genuine",
            "suspicious_phrases": ", ".join(p for p, _ in r.phrases),
            "reasons": "; ".join(t for t, _ in r.reasons),
        } for r in self.reviews])

    def aspects_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "aspect": a.aspect, "verdict": a.verdict, "score": round(a.score, 2),
            "mentions": a.mentions, "positive": a.positive, "neutral": a.neutral,
            "negative": a.negative, "discovered": a.discovered,
        } for a in self.aspects])

    def to_text(self) -> str:
        s = self.stats
        lines = [
            f"Reviews analysed : {s['n_total']}",
            f"Flagged as fake  : {s['n_fake']} ({s['pct_fake']:.0%})",
            f"Star rating      : {_fmt(s['raw_avg_rating'])} raw -> {_fmt(s['genuine_avg_rating'])} genuine only",
            "",
            f"Public opinion ({self.summary['backend']}, {self.summary['n_reviews']} genuine reviews):",
            "  " + self.summary["summary"],
            "",
            "Aspects:",
        ]
        lines += [f"  {a.aspect:<22} {a.verdict:<6} (score {a.score:+.2f}, {a.mentions} reviews)"
                  for a in self.aspects] or ["  (no aspects with enough mentions)"]
        return "\n".join(lines)


def _fmt(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def _mean_rating(rs):
    vals = [float(r.rating) for r in rs if r.rating is not None and not pd.isna(r.rating)]
    return float(np.mean(vals)) if vals else None


class FakeReviewDetective:
    def __init__(self, classifier: FakeReviewClassifier, summarizer: OpinionSummarizer | None = None,
                 absa: AspectSentimentAnalyzer | None = None, threshold: float = 0.5):
        self.classifier = classifier
        self.summarizer = summarizer or OpinionSummarizer()
        self.absa = absa or AspectSentimentAnalyzer()
        self.threshold = threshold

    @classmethod
    def from_model_file(cls, path, summarizer_backend="auto", absa_backend="auto", threshold=0.5):
        return cls(FakeReviewClassifier.load(path), OpinionSummarizer(summarizer_backend),
                   AspectSentimentAnalyzer(backend=absa_backend), threshold)

    def classify(self, texts, ratings=None) -> list[ReviewResult]:
        texts = [str(t) for t in texts]
        ratings = list(ratings) if ratings is not None else [None] * len(texts)
        exps = self.classifier.explain_batch(texts, ratings)
        return [ReviewResult(i, t, r, e.fake_prob, e.fake_prob >= self.threshold, e.phrases, e.reasons,
                             self.classifier.sentiment(t))
                for i, (t, r, e) in enumerate(zip(texts, ratings, exps))]

    def analyze(self, texts, ratings=None) -> ProductReport:
        reviews = self.classify(texts, ratings)
        return self.build_report(reviews)

    def build_report(self, reviews: list[ReviewResult]) -> ProductReport:
        """Separate from classify() so a dashboard can re-threshold without re-classifying."""
        for r in reviews:
            r.is_fake = r.fake_prob >= self.threshold
        genuine = [r for r in reviews if not r.is_fake]
        g_text = [r.text for r in genuine]
        sent = np.array([r.sentiment for r in genuine]) if genuine else np.array([])
        stats = {
            "n_total": len(reviews),
            "n_fake": len(reviews) - len(genuine),
            "pct_fake": (len(reviews) - len(genuine)) / max(len(reviews), 1),
            "raw_avg_rating": _mean_rating(reviews),
            "genuine_avg_rating": _mean_rating(genuine),
            "genuine_positive": float((sent > 0.1).mean()) if len(sent) else 0.0,
            "genuine_negative": float((sent < -0.1).mean()) if len(sent) else 0.0,
        }
        return ProductReport(reviews, self.summarizer.summarize(g_text), self.absa.analyze(g_text), stats)
