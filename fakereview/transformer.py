"""Fine-tuned transformer scorer: texts -> P(fake).

Wraps a sequence-classification model saved by finetune.py (label 1 = fake). Plug it into
the linear classifier with `clf.transformer_scorer = TransformerScorer(path)`; the two
probabilities are then averaged. Only the model path is pickled, so a classifier.joblib
with a scorer attached stays small and reloads the weights from disk.
"""
import numpy as np


class TransformerScorer:
    def __init__(self, path: str, max_length: int = 256, batch_size: int = 16):
        self.path = path
        self.max_length = max_length
        self.batch_size = batch_size
        self._load()

    def _load(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.path)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.path).eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def __call__(self, texts) -> np.ndarray:
        import torch

        texts = [str(t) for t in texts]
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self._tok(texts[i:i + self.batch_size], return_tensors="pt", padding=True,
                            truncation=True, max_length=self.max_length).to(self._device)
            with torch.no_grad():
                logits = self._model(**enc).logits
            out += torch.softmax(logits, -1)[:, 1].tolist()
        return np.array(out)

    def __getstate__(self):
        return {"path": self.path, "max_length": self.max_length, "batch_size": self.batch_size}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load()
