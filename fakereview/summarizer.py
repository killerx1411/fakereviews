"""Stage 2: public-opinion summary from genuine reviews only.

Abstractive backend: a pretrained Hugging Face summarizer (default facebook/bart-large-cnn,
used off the shelf, no fine-tuning). Reviews are ranked by centrality, the most
representative ones are packed into chunks that fit the model's input, each chunk is
summarised, and multiple chunk summaries are fused with a second summarisation pass.

Extractive fallback (no transformers installed): TextRank over review sentences with
MMR to avoid picking near-duplicate sentences.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text_utils import sentences, words


def _centrality(texts):
    if len(texts) < 2:
        return np.ones(len(texts)), None
    try:
        X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts)
    except ValueError:  # all stop words
        return np.ones(len(texts)), None
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0.0)
    # PageRank by power iteration
    row = sim.sum(1, keepdims=True)
    P = np.divide(sim, row, out=np.full_like(sim, 1.0 / len(texts)), where=row > 0)
    r = np.full(len(texts), 1.0 / len(texts))
    for _ in range(100):
        r_new = 0.15 / len(texts) + 0.85 * P.T @ r
        if np.abs(r_new - r).sum() < 1e-8:
            break
        r = r_new
    return r, sim


class OpinionSummarizer:
    def __init__(self, backend: str = "auto", model_name: str = "facebook/bart-large-cnn",
                 chunk_words: int = 550, max_chunks: int = 6, n_sentences: int = 4):
        self.model_name = model_name
        self.chunk_words = chunk_words
        self.max_chunks = max_chunks
        self.n_sentences = n_sentences
        self._model = self._tok = None
        self.backend = "extractive"
        if backend in ("auto", "abstractive"):
            try:
                # loaded directly: transformers v5 dropped the "summarization" pipeline
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                self._tok = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()
                self.backend = "abstractive"
            except Exception:
                if backend == "abstractive":
                    raise

    def summarize(self, reviews) -> dict:
        reviews = [str(r).strip() for r in reviews if str(r).strip()]
        if not reviews:
            return {"summary": "No genuine reviews to summarise.", "backend": self.backend, "n_reviews": 0}
        summary = self._abstractive(reviews) if self._model is not None else self._extractive(reviews)
        return {"summary": summary, "backend": self.backend, "n_reviews": len(reviews)}

    # ------------------------------------------------------------------ abstractive
    def _gen(self, text, n_words):
        max_len = int(min(160, max(40, n_words * 0.6)))
        min_len = int(min(max_len - 10, max(15, n_words * 0.15)))
        import torch

        enc = self._tok(text, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            ids = self._model.generate(**enc, max_length=max_len, min_length=min_len, num_beams=4,
                                       no_repeat_ngram_size=3, length_penalty=2.0, early_stopping=True)
        return self._tok.decode(ids[0], skip_special_tokens=True).strip()

    def _abstractive(self, reviews):
        scores, _ = _centrality(reviews)
        budget = self.chunk_words * self.max_chunks
        chunks, cur, cur_n, used = [], [], 0, 0
        for i in np.argsort(-scores):  # most representative reviews first
            n = len(words(reviews[i]))
            if used + n > budget:
                continue
            if cur and cur_n + n > self.chunk_words:
                chunks.append(cur)
                cur, cur_n = [], 0
            cur.append(reviews[i])
            cur_n += n
            used += n
        if cur:
            chunks.append(cur)
        partial = [self._gen(" ".join(c), sum(len(words(r)) for r in c)) for c in chunks]
        if len(partial) == 1:
            return partial[0]
        joined = " ".join(partial)
        return self._gen(joined, len(words(joined)))

    # ------------------------------------------------------------------ extractive
    def _extractive(self, reviews, mmr_lambda: float = 0.7):
        sents = [s for r in reviews for s in sentences(r) if 4 <= len(words(s)) <= 40]
        if not sents:
            return " ".join(reviews)[:500]
        scores, sim = _centrality(sents)
        if sim is None:
            return " ".join(sents[: self.n_sentences])
        scores = scores / scores.max()
        chosen = []
        while len(chosen) < min(self.n_sentences, len(sents)):
            best, best_val = None, -1e9
            for i in range(len(sents)):
                if i in chosen:
                    continue
                redundancy = max((sim[i, j] for j in chosen), default=0.0)
                val = mmr_lambda * scores[i] - (1 - mmr_lambda) * redundancy
                if val > best_val:
                    best, best_val = i, val
            chosen.append(best)
        return " ".join(sents[i] for i in chosen)
