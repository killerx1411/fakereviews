"""Sentiment scoring in [-1, 1].

Backends:
  * "vader"   - vaderSentiment, if installed (pip install vaderSentiment)
  * "lexicon" - the built-in review lexicon below (no dependencies)
"auto" picks vader when available, otherwise the lexicon.

The built-in scorer is VADER-style: word valences, negation window, intensifiers,
'but' re-weighting and a handful of review-specific phrases ("stopped working",
"too quiet", "not loud enough"), normalised with x / sqrt(x^2 + alpha).
"""
import math
import re

from .text_utils import tokens_lower

POSITIVE = {
    "amazing": 3.2, "awesome": 3.1, "excellent": 3.2, "fantastic": 3.3, "outstanding": 3.3,
    "superb": 3.2, "perfect": 3.2, "brilliant": 3.0, "incredible": 3.1, "phenomenal": 3.3,
    "wonderful": 3.0, "exceptional": 3.1, "flawless": 3.1, "best": 3.0, "love": 3.0,
    "loved": 2.9, "loves": 2.9, "great": 2.9, "impressive": 2.6, "impressed": 2.5,
    "solid": 2.0, "good": 2.0, "nice": 1.9, "happy": 2.2, "pleased": 2.1, "satisfied": 2.0,
    "recommend": 2.0, "recommended": 2.0, "reliable": 2.1, "sturdy": 2.0, "durable": 2.1,
    "comfortable": 2.0, "comfy": 2.0, "crisp": 1.9, "clear": 1.5, "rich": 1.5, "smooth": 1.7,
    "fast": 1.6, "quick": 1.4, "easy": 1.7, "beautiful": 2.5, "gorgeous": 2.7, "sleek": 1.8,
    "stylish": 1.9, "elegant": 2.0, "worth": 1.8, "bargain": 2.0, "value": 1.0, "works": 1.2,
    "worked": 1.0, "fine": 0.8, "decent": 1.0, "ok": 0.6, "okay": 0.6, "alright": 0.6,
    "adequate": 0.6, "fair": 1.0, "fun": 2.0, "enjoy": 2.2, "enjoyed": 2.2, "helpful": 1.8, "useful": 1.7,
    "favorite": 2.4, "favourite": 2.4, "loud": 0.8, "powerful": 2.0, "bright": 1.2,
    "lightweight": 1.4, "convenient": 1.7, "accurate": 1.6, "premium": 1.8, "lasts": 1.0,
    "fits": 1.0, "soft": 0.8, "delicious": 2.8, "tasty": 2.4, "fresh": 1.5, "pleasant": 1.9,
    "thanks": 1.2, "glad": 1.9, "surprisingly": 0.6, "affordable": 1.6, "cheaper": 0.6,
}

NEGATIVE = {
    "terrible": -3.4, "horrible": -3.4, "awful": -3.3, "worst": -3.4, "useless": -3.0,
    "garbage": -3.2, "junk": -3.0, "trash": -3.1, "hate": -3.0, "hated": -3.0,
    "disappointing": -2.6, "disappointed": -2.6, "disappointment": -2.6, "poor": -2.3,
    "bad": -2.5, "broken": -2.6, "broke": -2.5, "defective": -2.8, "faulty": -2.6,
    "cheaply": -1.8, "flimsy": -2.2, "fragile": -1.6, "weak": -1.6, "mediocre": -1.4,
    "meh": -1.2, "underwhelming": -1.8, "overpriced": -2.2, "expensive": -1.0,
    "uncomfortable": -2.1, "slow": -1.6, "laggy": -1.9, "noisy": -1.5, "muffled": -1.8,
    "distorted": -1.9, "distorts": -1.8, "cuts": -0.8, "distortion": -1.7, "tinny": -1.7, "crackling": -1.8, "dies": -1.8,
    "died": -2.2, "dead": -2.0, "fails": -2.2, "failed": -2.2, "failure": -2.4,
    "problem": -1.5, "problems": -1.6, "issue": -1.3, "issues": -1.4, "annoying": -2.0,
    "frustrating": -2.3, "waste": -2.6, "refund": -1.5, "return": -0.8, "returned": -1.8,
    "returning": -1.6, "scam": -3.2, "fake": -2.0, "leaks": -2.0, "leaked": -2.0,
    "smells": -1.2, "stinks": -2.4, "hard": -0.8, "difficult": -1.3, "confusing": -1.6,
    "dim": -1.0, "short": -0.4, "small": -0.2, "tight": -0.6, "loose": -0.9, "rough": -1.0,
    "scratched": -1.7, "dented": -1.8, "damaged": -2.2, "missing": -1.8, "wrong": -1.8,
    "unreliable": -2.3, "inaccurate": -1.8, "overheats": -2.2, "overheating": -2.2,
    "hot": -0.4, "rude": -2.3, "unhelpful": -2.1, "late": -1.2, "delayed": -1.3,
    "cheap": -0.6, "lacking": -1.4, "lacks": -1.3, "sucks": -2.8, "bland": -1.4,
    "stale": -1.8, "itchy": -1.8, "ripped": -2.1, "torn": -2.0, "unfortunately": -1.2,
}

# Multi-word expressions are scored first and removed before word-level scoring.
PHRASES = [
    (r"\bstopped working\b", -2.8), (r"\bwaste of (?:money|time)\b", -3.0),
    (r"\bfell apart\b", -2.8), (r"\bfalls apart\b", -2.8), (r"\bnot worth\b", -2.4),
    (r"\bworth every (?:penny|rupee|cent)\b", 3.0), (r"\bhighly recommend(?:ed)?\b", 3.0),
    (r"\bwould not recommend\b", -2.8), (r"\bwouldn'?t recommend\b", -2.8),
    (r"\bdon'?t buy\b", -2.8), (r"\bdo not buy\b", -2.8), (r"\bfor the price\b", 0.8),
    (r"\bdoes the job\b", 1.4), (r"\bgets the job done\b", 1.4), (r"\bjust ok(?:ay)?\b", 0.2),
    (r"\bnothing special\b", -0.6), (r"\bcould be better\b", -0.9), (r"\bnot bad\b", 1.2),
    (r"\bno complaints\b", 1.8), (r"\bno issues\b", 1.6), (r"\bno problems?\b", 1.6),
    (r"\bfive stars\b", 2.5), (r"\b5 stars\b", 2.5), (r"\bone star\b", -2.5), (r"\b1 star\b", -2.5),
    (r"\btoo (?:low|quiet|soft|small|short|big|large|heavy|tight|loose|expensive|slow|thin|"
     r"hot|weak|bright|dim|long|bulky)\b", -1.6),
    (r"\bnot (?:\w+ )?enough\b", -1.5), (r"\bbarely (?:audible|works|lasts)\b", -2.0),
    (r"\bdrains? (?:fast|quickly)\b", -2.0), (r"\blasts? (?:all day|forever|for days)\b", 2.2),
]
_PHRASES = [(re.compile(p, re.IGNORECASE), v) for p, v in PHRASES]

NEGATORS = {
    "not", "no", "never", "nothing", "nobody", "none", "neither", "nor", "without", "hardly",
    "barely", "cannot", "cant", "dont", "doesnt", "didnt", "isnt", "wasnt", "arent", "werent",
    "wont", "wouldnt", "shouldnt", "couldnt", "aint",
}
BOOSTERS = {
    "very": 1.3, "really": 1.3, "extremely": 1.5, "super": 1.4, "so": 1.25, "incredibly": 1.5,
    "absolutely": 1.4, "totally": 1.3, "truly": 1.3, "highly": 1.3, "insanely": 1.5,
    "quite": 1.1, "pretty": 1.1, "most": 1.2, "slightly": 0.6, "somewhat": 0.7,
    "fairly": 0.85, "kinda": 0.7, "bit": 0.7, "little": 0.75,
}
_ALPHA = 15.0


def _is_negator(tok: str) -> bool:
    return tok in NEGATORS or tok.endswith("n't")


def lexicon_score(text: str) -> float:
    text = str(text)
    total = 0.0
    for rx, val in _PHRASES:
        n = len(rx.findall(text))
        if n:
            total += val * n
            text = rx.sub(" ", text)

    toks = tokens_lower(text)
    but_idx = max((i for i, t in enumerate(toks) if t in {"but", "however"}), default=-1)
    for i, tok in enumerate(toks):
        val = POSITIVE.get(tok) or NEGATIVE.get(tok)
        if not val:
            continue
        if i > 0 and toks[i - 1] in BOOSTERS:
            val *= BOOSTERS[toks[i - 1]]
        if any(_is_negator(t) for t in toks[max(0, i - 3):i]):
            val *= -0.74
        if but_idx >= 0:
            val *= 1.5 if i > but_idx else 0.5
        total += val

    if total:
        total += math.copysign(min(text.count("!"), 4) * 0.29, total)
    return total / math.sqrt(total * total + _ALPHA)


class SentimentScorer:
    def __init__(self, backend: str = "auto"):
        self.backend = backend
        self._vader = None
        if backend in ("auto", "vader"):
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self._vader = SentimentIntensityAnalyzer()
                self.backend = "vader"
            except ImportError:
                if backend == "vader":
                    raise
                self.backend = "lexicon"

    def score(self, text: str) -> float:
        if self._vader is not None:
            return self._vader.polarity_scores(str(text))["compound"]
        return lexicon_score(text)

    def __call__(self, text: str) -> float:
        return self.score(text)

    def __getstate__(self):
        return {"backend": self.backend}

    def __setstate__(self, state):
        self.__init__(state["backend"])
