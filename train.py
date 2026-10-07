"""Train and evaluate the fake review classifier.

    python train.py --data data/fake_reviews_dataset.csv            # real training
    python train.py --data data/fake_reviews_dataset.csv --ablation # + feature ablation table
    python train.py --demo                                          # synthetic smoke test
    python train.py --data ... --transformer models/transformer     # + fine-tuned model (see FINETUNING.md)

Saves the model to models/classifier.joblib (change with --out).
"""
import argparse
import copy
import time

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

from fakereview.classifier import FakeReviewClassifier
from fakereview.data import load_labelled, synthetic, train_test


def evaluate(clf, X_te, r_te, y_te, name="model"):
    p = clf.predict_proba(X_te, r_te)
    pred = (p >= 0.5).astype(int)
    print(f"{name:<28} acc={accuracy_score(y_te, pred):.3f}  f1={f1_score(y_te, pred):.3f}  "
          f"auc={roc_auc_score(y_te, p):.3f}")
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="labelled CSV (e.g. fake_reviews_dataset.csv)")
    ap.add_argument("--demo", action="store_true", help="train on synthetic data (smoke test only)")
    ap.add_argument("--out", default="models/classifier.joblib")
    ap.add_argument("--perplexity", default="auto", choices=["auto", "gpt2", "bigram"])
    ap.add_argument("--sample", type=int, default=0, help="subsample N rows for a quick run")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--transformer", help="fine-tuned model dir from finetune.py; fused with the linear model")
    args = ap.parse_args()

    if args.demo:
        df = synthetic(800)
        print("Using SYNTHETIC data: scores below only prove the code runs.")
    elif args.data:
        df = load_labelled(args.data)
    else:
        ap.error("pass --data PATH or --demo")
    tr, te = train_test(df, args.sample)
    print(f"{len(tr)} train / {len(te)} test reviews, {tr.label.mean():.1%} fake")
    X_tr, X_te = tr.text.tolist(), te.text.tolist()
    r_tr, r_te = tr.rating.tolist(), te.rating.tolist()

    t0 = time.time()
    clf = FakeReviewClassifier(perplexity_backend=args.perplexity).fit(X_tr, r_tr, tr.label.values)
    print(f"trained in {time.time() - t0:.1f}s (perplexity backend: {clf.perplexity.backend}, "
          f"sentiment backend: {clf.sentiment.backend})\n")
    pred = evaluate(clf, X_te, r_te, te.label.values, "TF-IDF + dense (full)")

    if args.ablation:
        for name, text, dense in [("TF-IDF only", True, False), ("Dense features only", False, True)]:
            m = copy.deepcopy(clf)
            m.use_text, m.use_dense = text, dense
            m.model = copy.deepcopy(clf.model)
            ppl = m.perplexity.cross_fit_score(X_tr, tr.label.values) if dense else None
            m.model.fit(m._matrix(X_tr, r_tr, fit=True, ppl=ppl), tr.label.values)
            evaluate(m, X_te, r_te, te.label.values, name)

    if args.transformer:
        from fakereview.transformer import TransformerScorer
        clf.transformer_scorer = TransformerScorer(args.transformer)
        p_tf = clf.transformer_scorer(X_te)
        print(f"{'Fine-tuned transformer':<28} acc={accuracy_score(te.label.values, p_tf >= 0.5):.3f}  "
              f"f1={f1_score(te.label.values, p_tf >= 0.5):.3f}  auc={roc_auc_score(te.label.values, p_tf):.3f}")
        pred = evaluate(clf, X_te, r_te, te.label.values, "Ensemble (linear + transf.)")

    print("\n" + classification_report(te.label.values, pred, target_names=["genuine", "fake"], digits=3))
    fake_ng, real_ng = clf.top_global_ngrams(15)
    print("Most 'fake' n-grams   :", ", ".join(w for w, _ in fake_ng))
    print("Most 'genuine' n-grams:", ", ".join(w for w, _ in real_ng))

    from fakereview.features import DENSE_FEATURES
    dense_coef = clf.model.coef_[0][clf.n_text:]
    order = np.argsort(-np.abs(dense_coef))[:8]
    print("Strongest dense features:", ", ".join(f"{DENSE_FEATURES[i]} ({dense_coef[i]:+.2f})" for i in order))

    clf.save(args.out)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
