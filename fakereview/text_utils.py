"""Small, dependency-free text helpers shared by every stage."""
import html
import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")
_CLAUSE_SPLIT = re.compile(
    r"\s*(?:;|,?\s+\bbut\b|,?\s+\bhowever\b|,?\s+\balthough\b|,\s*\bthough\b|"
    r",?\s+\bwhereas\b|,\s*\bwhile\b|,\s*\byet\b|,?\s+\bexcept\b)\s*",
    re.IGNORECASE,
)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(str(text)) if s and s.strip()]


def clauses(sentence: str) -> list[str]:
    """Split on contrastive conjunctions so 'great sound but weak bass' gives two clauses."""
    return [c.strip() for c in _CLAUSE_SPLIT.split(sentence) if c and c.strip()]


def segments(clause: str) -> list[str]:
    """Comma-level pieces of a clause: 'great quality, ok battery, weak volume'."""
    return [p.strip() for p in re.split(r",|\s+-\s+", clause) if p.strip()]


def words(text: str) -> list[str]:
    return _WORD_RE.findall(str(text))


def tokens_lower(text: str) -> list[str]:
    return [w.lower() for w in words(text)]


def highlight(text: str, phrases: list[str], tag: str = "mark") -> str:
    """Return HTML-escaped text with every occurrence of the given phrases wrapped in <tag>."""
    text = str(text)
    spans = []
    for p in sorted(set(phrases), key=len, reverse=True):
        if not p.strip():
            continue
        pattern = r"\b" + r"\W+".join(re.escape(w) for w in p.split()) + r"\b"
        spans += [(m.start(), m.end()) for m in re.finditer(pattern, text, re.IGNORECASE)]
    if not spans:
        return html.escape(text)
    spans.sort()
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out, pos = [], 0
    for s, e in merged:
        out.append(html.escape(text[pos:s]))
        out.append(f"<{tag}>{html.escape(text[s:e])}</{tag}>")
        pos = e
    out.append(html.escape(text[pos:]))
    return "".join(out)
