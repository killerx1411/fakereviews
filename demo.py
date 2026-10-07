"""Command-line report for one product's reviews.

    python demo.py --reviews data/sample_product_reviews.csv --model models/classifier.joblib
"""
import argparse

from fakereview.data import load_reviews
from fakereview.pipeline import FakeReviewDetective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", default="data/sample_product_reviews.csv")
    ap.add_argument("--model", default="models/classifier.joblib")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--summarizer", default="auto", choices=["auto", "abstractive", "extractive"])
    ap.add_argument("--absa", default="auto", choices=["auto", "transformer", "lexicon"])
    ap.add_argument("--out", help="optional CSV path for per-review verdicts")
    args = ap.parse_args()

    df = load_reviews(args.reviews)
    det = FakeReviewDetective.from_model_file(args.model, args.summarizer, args.absa, args.threshold)
    report = det.analyze(df.text, df.rating)
    print(report.to_text())
    print("\nFlagged reviews:")
    for r in sorted(report.fake, key=lambda r: -r.fake_prob):
        print(f"  [{r.fake_prob:.2f}] {r.text[:90]}{'...' if len(r.text) > 90 else ''}")
        if r.phrases:
            print("         phrases:", ", ".join(p for p, _ in r.phrases))
        if r.reasons:
            print("         reasons:", "; ".join(t for t, _ in r.reasons))
    if args.out:
        report.reviews_frame().to_csv(args.out, index=False)
        print(f"\nper-review results -> {args.out}")


if __name__ == "__main__":
    main()
