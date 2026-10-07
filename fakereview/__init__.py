"""FakeReview Detective: fake review filtering and trusted opinion summarization."""
from .classifier import FakeReviewClassifier
from .summarizer import OpinionSummarizer
from .absa import AspectSentimentAnalyzer
from .pipeline import FakeReviewDetective, ProductReport

__all__ = [
    "FakeReviewClassifier",
    "OpinionSummarizer",
    "AspectSentimentAnalyzer",
    "FakeReviewDetective",
    "ProductReport",
]
