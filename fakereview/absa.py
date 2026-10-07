"""Stage 3: aspect-based sentiment analysis over genuine reviews.

Aspect extraction
  * a lexicon of common product aspects and their synonyms (editable / extendable), plus
  * optional discovery of product-specific aspects from patterns like "the <noun> is ...".

Aspect sentiment
  * "transformer": a pretrained ABSA model (default yangheng/deberta-v3-base-absa-v1.1)
    that classifies (sentence, aspect) pairs as Positive / Neutral / Negative. Used off
    the shelf, no fine-tuning.
  * "lexicon": sentiment of the clause that mentions the aspect. Reviews are split into
    clauses at "but", "however", ";" ... so "great sound but the volume is too low"
    credits "great" to sound and "too low" to volume.

Each review contributes one vote per aspect (its mentions are averaged), so a single
rambling review cannot dominate an aspect. Mean score -> Great / Good / OK / Poor / Bad.
"""
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from .sentiment import SentimentScorer
from .text_utils import clauses, segments, sentences

DEFAULT_ASPECTS = {
    "Quality": ["quality", "build quality", "build", "construction", "craftsmanship", "material", "materials"],
    "Price / value": ["price", "value", "cost", "money", "worth", "deal"],
    "Battery life": ["battery", "battery life", "charge", "charging", "charger"],
    "Sound": ["sound", "audio", "bass", "treble", "sound quality"],
    "Volume": ["volume", "loudness", "max volume"],
    "Display": ["screen", "display", "resolution", "brightness", "touchscreen"],
    "Performance": ["performance", "speed", "processor", "lag", "responsiveness"],
    "Camera": ["camera", "photos", "pictures", "video quality", "lens"],
    "Design": ["design", "look", "looks", "style", "color", "colour", "appearance", "finish"],
    "Size / weight": ["size", "weight", "portable", "portability", "compact"],
    "Comfort / fit": ["comfort", "fit", "fits", "sizing", "comfortable"],
    "Durability": ["durability", "durable", "lasted", "sturdy", "sturdiness"],
    "Ease of use": ["setup", "set up", "instructions", "controls", "buttons", "interface", "app", "ease of use"],
    "Connectivity": ["bluetooth", "connection", "connectivity", "pairing", "wifi", "wi-fi", "range"],
    "Delivery / packaging": ["delivery", "shipping", "packaging", "package", "box", "arrived"],
    "Customer service": ["customer service", "support", "seller", "warranty", "service"],
    "Taste / smell": ["taste", "flavor", "flavour", "smell", "scent"],
}

_GENERIC = {
    "product", "item", "thing", "things", "one", "it", "this", "unit", "order", "purchase",
    "rest", "only", "whole", "first", "second", "other", "same", "best", "worst", "problem",
    "issue", "reason", "way", "time", "bad", "good", "great", "end", "result", "difference",
}
_DISCOVER_RE = re.compile(
    r"\b(?:the|its|their|this|my)\s+([a-z]{3,}(?:\s[a-z]{3,})?)\s+(?:is|was|are|were|seems|feels|works|lasts|sounds|looks)\b",
    re.IGNORECASE,
)


def verdict(score: float) -> str:
    if score >= 0.5:
        return "Great"
    if score >= 0.25:
        return "Good"
    if score >= -0.1:
        return "OK"
    if score >= -0.4:
        return "Poor"
    return "Bad"


@dataclass
class AspectResult:
    aspect: str
    mentions: int  # number of reviews mentioning it
    positive: int
    neutral: int
    negative: int
    score: float
    verdict: str
    discovered: bool = False
    examples: list = field(default_factory=list)  # [(snippet, score)]


class AspectSentimentAnalyzer:
    def __init__(self, aspects: dict | None = None, backend: str = "auto",
                 model_name: str = "yangheng/deberta-v3-base-absa-v1.1",
                 discover: bool = True, min_mentions: int = 2, max_discovered: int = 5):
        self.aspects = {k: list(v) for k, v in (aspects or DEFAULT_ASPECTS).items()}
        self.discover = discover
        self.min_mentions = min_mentions
        self.max_discovered = max_discovered
        self.sentiment = SentimentScorer("auto")
        self._clf = None
        self.backend = "lexicon"
        if backend in ("auto", "transformer"):
            try:
                from transformers import pipeline
                self._clf = pipeline("text-classification", model=model_name, top_k=None)
                self.backend = "transformer"
            except Exception:
                if backend == "transformer":
                    raise

    # ------------------------------------------------------------------ extraction
    def _known_terms(self):
        return {t.lower() for terms in self.aspects.values() for t in terms}

    def discover_aspects(self, reviews) -> dict:
        known = self._known_terms()
        counts = Counter()
        for r in reviews:
            seen = set()
            for m in _DISCOVER_RE.finditer(str(r)):
                term = m.group(1).lower()
                if term in known or term.split()[-1] in _GENERIC or term.split()[0] in _GENERIC:
                    continue
                if any(w in known for w in term.split()):
                    continue
                seen.add(term)
            counts.update(seen)
        found = [t for t, c in counts.most_common() if c >= self.min_mentions][: self.max_discovered]
        return {t.capitalize(): [t] for t in found}

    @staticmethod
    def _pattern(terms):
        alts = sorted({re.escape(t) for t in terms}, key=len, reverse=True)
        return re.compile(r"\b(?:" + "|".join(alts) + r")(?:s|es)?\b", re.IGNORECASE)

    @staticmethod
    def _match(clause, patterns):
        """Aspects mentioned in a clause; the longest term wins ('sound quality' -> Sound, not Quality)."""
        hits = [(m.start(), m.end(), a, m.group(0)) for a, rx in patterns.items() for m in rx.finditer(clause)]
        hits = [h for h in hits
                if not any(o is not h and o[0] <= h[0] and h[1] <= o[1] and (o[1] - o[0]) > (h[1] - h[0])
                           for o in hits)]
        seen, out = set(), []
        for _, _, a, t in sorted(hits):
            if a not in seen:
                seen.add(a)
                out.append((a, t))
        return out

    def _opinion_span(self, clause, term):
        """The comma segment holding the aspect if it carries an opinion, else the whole clause."""
        for seg in segments(clause):
            if term.lower() in seg.lower():
                return seg if abs(self.sentiment(seg)) > 0.05 else clause
        return clause

    # ------------------------------------------------------------------ sentiment
    def _score_pairs(self, pairs):
        """pairs: [(sentence, clause, term)] -> scores in [-1, 1]."""
        if not pairs:
            return []
        if self._clf is not None:
            inputs = [{"text": s, "text_pair": t} for s, _, t in pairs]
            outs = self._clf(inputs, batch_size=16, truncation=True)
            scores = []
            for res in outs:
                p = {d["label"].lower(): d["score"] for d in res}
                scores.append(p.get("positive", 0.0) - p.get("negative", 0.0))
            return scores
        return [self.sentiment(c) for _, c, _ in pairs]

    # ------------------------------------------------------------------ main
    def analyze(self, reviews, neutral_band: float = 0.1, n_examples: int = 3) -> list[AspectResult]:
        reviews = [str(r) for r in reviews]
        aspects = dict(self.aspects)
        discovered = self.discover_aspects(reviews) if self.discover else {}
        aspects.update(discovered)
        patterns = {a: self._pattern(t) for a, t in aspects.items()}

        # collect every (review, aspect, sentence, opinion span, term) mention
        mentions = []
        for ri, review in enumerate(reviews):
            for sent in sentences(review):
                for cl in clauses(sent):
                    for aspect, term in self._match(cl, patterns):
                        mentions.append((ri, aspect, sent, self._opinion_span(cl, term), term))
        scores = self._score_pairs([(s, c, t) for _, _, s, c, t in mentions])

        per_review = defaultdict(list)  # (aspect, review) -> scores
        snippets = defaultdict(list)
        for (ri, aspect, _, cl, _), sc in zip(mentions, scores):
            per_review[(aspect, ri)].append(sc)
            snippets[aspect].append((cl, sc))

        by_aspect = defaultdict(list)
        for (aspect, _), scs in per_review.items():
            by_aspect[aspect].append(float(np.mean(scs)))

        results = []
        for aspect, votes in by_aspect.items():
            if len(votes) < self.min_mentions:
                continue
            v = np.array(votes)
            mean = float(v.mean())
            ex = sorted(snippets[aspect], key=lambda x: -abs(x[1]))
            # show the strongest opinions on both sides
            pos = [e for e in ex if e[1] > neutral_band][: (n_examples + 1) // 2]
            neg = [e for e in ex if e[1] < -neutral_band][: n_examples // 2 + 1]
            results.append(AspectResult(
                aspect=aspect, mentions=len(v),
                positive=int((v > neutral_band).sum()), neutral=int((abs(v) <= neutral_band).sum()),
                negative=int((v < -neutral_band).sum()), score=mean, verdict=verdict(mean),
                discovered=aspect in discovered, examples=(pos + neg)[:n_examples] or ex[:n_examples],
            ))
        return sorted(results, key=lambda r: -r.mentions)
