"""Fine-tune a transformer to classify reviews as genuine (0) or fake (1).

    python finetune.py --data data/fake_reviews_dataset.csv                  # full run (GPU recommended)
    python finetune.py --data data/fake_reviews_dataset.csv --sample 4000    # quick CPU check

Uses the same train/test split as train.py, so the test numbers are directly comparable and
the ensemble in train.py --transformer is evaluated on reviews neither model trained on.
Saves the model + tokenizer to models/transformer (change with --out). See FINETUNING.md.
"""
import argparse
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from fakereview.data import load_labelled, train_test
from fakereview.transformer import TransformerScorer


def batches(texts, labels, tok, bs, max_len, shuffle, seed):
    order = np.random.default_rng(seed).permutation(len(texts)) if shuffle else np.arange(len(texts))
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        enc["labels"] = torch.tensor(labels[idx])
        yield enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="distilbert/distilroberta-base", help="any HF encoder, e.g. FacebookAI/roberta-base")
    ap.add_argument("--out", default="models/transformer")
    ap.add_argument("--sample", type=int, default=0, help="subsample N rows (must match train.py --sample)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, te = train_test(load_labelled(args.data), args.sample)
    X_tr, y_tr = tr.text.tolist(), tr.label.values
    X_te, y_te = te.text.tolist(), te.label.values
    print(f"{len(X_tr)} train / {len(X_te)} test reviews on {device}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label={0: "genuine", 1: "fake"}, label2id={"genuine": 0, "fake": 1}).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(X_tr) + args.batch_size - 1) // args.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        for enc in batches(X_tr, y_tr, tok, args.batch_size, args.max_len, True, args.seed + epoch):
            loss = model(**enc.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step(), opt.zero_grad()
            step += 1
            if step % 50 == 0 or step == steps:
                eta = (time.time() - t0) / step * (steps - step)
                print(f"epoch {epoch + 1} step {step}/{steps} loss {loss.item():.4f} (eta {eta / 60:.0f} min)", flush=True)

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    p = TransformerScorer(args.out, max_length=args.max_len)(X_te)
    print(f"\ntest acc={accuracy_score(y_te, p >= 0.5):.3f}  f1={f1_score(y_te, p >= 0.5):.3f}  "
          f"auc={roc_auc_score(y_te, p):.3f}")
    print(f"saved -> {args.out}\nnext: python train.py --data {args.data} --transformer {args.out}"
          + (f" --sample {args.sample}" if args.sample else ""))


if __name__ == "__main__":
    main()
