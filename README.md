# FakeReview Detective

FakeReview Detective takes all the reviews for one product, flags the ones that look fake or
machine-generated, and then works only from the genuine ones. From those it writes a short
summary of what buyers think and gives each product feature a verdict
(for example *Sound: Great, Battery life: OK, Volume: Bad*).

It does three things, in order:

1. **Fake review detection**: an explainable classifier (TF-IDF n-grams plus 22 hand-made
   linguistic features fed into logistic regression), with an optional fine-tuned transformer
   averaged in.
2. **Opinion summary**: summarises the genuine reviews, using BART if it is installed or
   TextRank otherwise.
3. **Aspect-based sentiment**: finds product features mentioned in the genuine reviews and
   rates each one from Great to Bad.

You can use it from a **Streamlit dashboard** (`app.py`) or the **command line** (`demo.py`).

---

## Table of contents

- [Quick start](#quick-start)
- [Project layout](#project-layout)
- [Installation in detail](#installation-in-detail)
- [Using it](#using-it)
- [Training your own model](#training-your-own-model)
- [How it works](#how-it-works)
  - [Pipeline overview](#pipeline-overview)
  - [Stage 1: fake review classifier](#stage-1-fake-review-classifier)
  - [Stage 2: opinion summary](#stage-2-opinion-summary)
  - [Stage 3: aspect-based sentiment](#stage-3-aspect-based-sentiment)
  - [Sentiment scorer](#sentiment-scorer)
  - [Pipeline and report](#pipeline-and-report)
- [Backends: with and without transformers](#backends-with-and-without-transformers)
- [Data](#data)
- [Input CSV format](#input-csv-format)
- [Limitations and caveats](#limitations-and-caveats)
- [Troubleshooting](#troubleshooting)

---

## Quick start

You need **Python 3.10 or newer** (developed on 3.13) and git.

```bash
git clone https://github.com/killerx1411/fakereviews.git
cd fakereviews
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

Install the core packages and start the dashboard:

```bash
pip install numpy pandas scipy scikit-learn joblib streamlit altair vaderSentiment
streamlit run app.py
```

A browser tab opens at http://localhost:8501 showing the sample product
(`data/sample_product_reviews.csv`, a portable Bluetooth speaker). Upload your own CSV from the
sidebar to analyse another product.

The repository already includes a trained classifier (`models/classifier.joblib`, trained on
the full 40k-review dataset), so you do **not** need to train anything first.

For the command-line report instead:

```bash
python demo.py
```

> **Want the full transformer models?** Run `pip install -r requirements.txt` instead (see
> [Installation in detail](#installation-in-detail)). The first run then downloads roughly
> 2.5 GB of pretrained models from Hugging Face, and analysis on CPU becomes much slower.

---

## Project layout

```
fakereviews/
├── app.py                         Streamlit dashboard
├── demo.py                        Command-line report for one product
├── train.py                       Train + evaluate the fake review classifier
├── finetune.py                    Fine-tune a transformer classifier (GPU recommended)
├── requirements.txt               Dependencies (core + optional)
├── FINETUNING.md                  Step-by-step guide for fine-tuning on Google Colab
├── data/
│   ├── fake_reviews_dataset.csv   Training data: 40k Amazon reviews, labelled OR / CG
│   └── sample_product_reviews.csv Example product (Bluetooth speaker) for the demo
├── models/
│   └── classifier.joblib          Trained classifier (linear model, bigram perplexity)
└── fakereview/                    The library
    ├── __init__.py                Public API
    ├── data.py                    CSV loading, column detection, train/test split, synthetic data
    ├── text_utils.py              Sentence / clause / word splitting, HTML highlighting
    ├── sentiment.py               Sentiment scoring (VADER or built-in lexicon)
    ├── features.py                Stylometric features, perplexity, rating mismatch
    ├── classifier.py              Stage 1: FakeReviewClassifier + explanations
    ├── transformer.py             Wrapper for the fine-tuned transformer (P(fake))
    ├── summarizer.py              Stage 2: OpinionSummarizer
    ├── absa.py                    Stage 3: AspectSentimentAnalyzer
    └── pipeline.py                FakeReviewDetective: ties the stages into one report
```

Not in the repository (see `.gitignore`): the virtual environment, Python caches, and
fine-tuned transformer weights. The weights are over 300 MB, above GitHub's file size limit.
Generate them with `finetune.py` (see [FINETUNING.md](FINETUNING.md)).

---

## Installation in detail

### Option A: core only (fast, small, recommended to start)

```bash
pip install numpy pandas scipy scikit-learn joblib streamlit altair vaderSentiment
```

Everything works with these packages. Each stage uses its lightweight backend: bigram
perplexity, extractive TextRank summary, and lexicon aspect sentiment.
`vaderSentiment` is optional but gives better sentiment scores than the built-in lexicon.

### Option B: everything, including transformers

PyTorch is best installed first, from the official index for your platform. CPU-only on Windows/Linux:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

(For an NVIDIA GPU, pick the CUDA command from https://pytorch.org/get-started/locally/.)

With `transformers` and `torch` installed, each stage switches to its transformer backend
automatically. The first run downloads the pretrained models:

| Model | Used for | Approx. size |
|---|---|---|
| `distilbert/distilgpt2` | perplexity feature (only when **training** with `--perplexity auto/gpt2`) | 350 MB |
| `facebook/bart-large-cnn` | abstractive opinion summary | 1.6 GB |
| `yangheng/deberta-v3-base-absa-v1.1` | aspect sentiment | 700 MB |

To keep the light backends with transformers installed, pick **Summary: extractive** and
**Aspect sentiment: lexicon** in the dashboard's *Models* section, or pass
`--summarizer extractive --absa lexicon` to `demo.py`.

### A note on the saved model and library versions

`models/classifier.joblib` is a pickled scikit-learn object. It was saved with
**scikit-learn 1.9.1, numpy 2.5, pandas 3.0, Python 3.13**. A very different scikit-learn
version may print an `InconsistentVersionWarning` or fail to load it. If that happens, either
install a matching version (`pip install scikit-learn==1.9.1`) or retrain. On CPU the
bigram model trains in a few minutes:

```bash
python train.py --data data/fake_reviews_dataset.csv --perplexity bigram
```

---

## Using it

### Dashboard (`app.py`)

```bash
streamlit run app.py
```

**Sidebar**
- **Upload a CSV** with one product's reviews. Without one, the sample speaker reviews are shown.
- **Review text column / Star rating column**: guessed automatically, change them if the guess is wrong.
  The rating column is optional, but some features (rating extremity, rating/sentiment
  mismatch) and the before/after star rating need it.
- **Flag as fake when probability is at least**: the decision threshold (default 0.5). A higher
  value flags fewer reviews and lets more of them into the summary. A lower value is stricter.
- **Models** (expander): which classifier file to load, and which summary and aspect backends to use.

**Top metrics**: number of reviews, how many were flagged, the listed average star rating and
the average from genuine reviews only (with the difference).

**Tabs**
- **What buyers think**: the summary of genuine reviews, the share of positive and negative
  genuine reviews, and a colour-coded verdict badge for each aspect.
- **Feature by feature**: bar chart of each aspect's average sentiment (−1 to +1), plus
  expandable example snippets (🟢 positive, 🔴 negative, ⚪ neutral). Aspects marked
  *(found in reviews)* were discovered in this product's reviews rather than taken from the
  built-in list.
- **Every review**: every review with its fake probability and stars. Flagged reviews show
  the **suspicious phrases highlighted** and the **reasons** (e.g. *"Very predictable,
  template-like wording"*). You can filter, sort by suspicion, and **download all results as CSV**.

### Command line (`demo.py`)

```bash
python demo.py                                              # sample product
python demo.py --reviews path/to/reviews.csv                # your own product
python demo.py --threshold 0.7 --out results.csv            # flag fewer reviews, save per-review CSV
python demo.py --summarizer extractive --absa lexicon       # force light backends
```

| Flag | Default | Meaning |
|---|---|---|
| `--reviews` | `data/sample_product_reviews.csv` | CSV of one product's reviews |
| `--model` | `models/classifier.joblib` | trained classifier |
| `--threshold` | `0.5` | fake-probability cut-off |
| `--summarizer` | `auto` | `auto` / `abstractive` / `extractive` |
| `--absa` | `auto` | `auto` / `transformer` / `lexicon` |
| `--out` | none | write per-review verdicts to this CSV |

Real output on the sample product (`python demo.py --summarizer extractive --absa lexicon`, shortened):

```
Reviews analysed : 22
Flagged as fake  : 5 (23%)
Star rating      : 3.86 raw -> 3.82 genuine only

Public opinion (extractive, 17 genuine reviews):
  great quality, ok battery, the volume though is really disappointing at max. sound quality is good
  but volume is weak outdoors. Great product, great price, great quality! Excellent build quality, ...

Aspects:
  Volume                 Poor   (score -0.38, 9 reviews)
  Quality                Great  (score +0.56, 8 reviews)
  Battery life           OK     (score +0.03, 8 reviews)
  Sound                  Good   (score +0.28, 7 reviews)
  Price / value          Good   (score +0.37, 3 reviews)
  Connectivity           Poor   (score -0.24, 2 reviews)

Flagged reviews:
  [0.90] I highly recommend this product. It is the perfect gift and the quality is excellent. You ...
         phrases: quality is, the quality, recommend this, excellent, the perfect, love
         reasons: Very predictable, template-like wording (low perplexity); Very short review; Many function words
  ...
```

(Numbers change with the model and backends you use.)

### As a Python library

```python
from fakereview import FakeReviewDetective

det = FakeReviewDetective.from_model_file("models/classifier.joblib",
                                          summarizer_backend="extractive", absa_backend="lexicon")
report = det.analyze(["Great sound but the volume is too low.", "Best product ever!!! Highly recommend!"],
                     ratings=[4, 5])
print(report.to_text())
report.reviews_frame()   # pandas DataFrame, one row per review
report.aspects_frame()   # pandas DataFrame, one row per aspect
```

---

## Training your own model

### Train and evaluate the classifier (`train.py`)

```bash
# full training: prints accuracy / F1 / ROC-AUC, top n-grams, strongest features, saves the model
python train.py --data data/fake_reviews_dataset.csv

# add an ablation table (TF-IDF only vs dense features only vs both)
python train.py --data data/fake_reviews_dataset.csv --ablation

# quick run on 5,000 reviews with the cheap bigram perplexity (no GPT-2)
python train.py --data data/fake_reviews_dataset.csv --sample 5000 --perplexity bigram

# smoke test with no dataset (synthetic data, scores mean nothing)
python train.py --demo
```

| Flag | Default | Meaning |
|---|---|---|
| `--data` | none | labelled CSV |
| `--demo` | off | use synthetic data instead (code-path check only) |
| `--out` | `models/classifier.joblib` | where to save the model |
| `--perplexity` | `auto` | `gpt2` (needs transformers), `bigram`, or `auto` (GPT-2 if available) |
| `--sample` | `0` (all) | subsample N rows for a quick run |
| `--ablation` | off | also train/evaluate text-only and dense-only models |
| `--transformer` | none | path to a fine-tuned model from `finetune.py`; averaged with the linear model |

The split is always stratified 80/20 with `random_state=42` (`fakereview.data.train_test`), so
results can be reproduced and compared across runs.

### Fine-tune the transformer (`finetune.py`)

```bash
python finetune.py --data data/fake_reviews_dataset.csv            # writes models/transformer
python train.py --data data/fake_reviews_dataset.csv --transformer models/transformer
```

This fine-tunes `distilbert/distilroberta-base` (AdamW, lr 2e-5, 2 epochs, max 256 tokens) on
the **same** train split as `train.py`, then fuses it with the linear model. On CPU this takes
hours. Use a GPU, e.g. free Google Colab. **[FINETUNING.md](FINETUNING.md) has the complete
step-by-step guide.** If you use `--sample`, pass the same value to both scripts so their
train/test splits match.

`classifier.joblib` stores only the *path* to the transformer folder, not the weights, so
keep `models/transformer/` next to it and run from the project root.

---

## How it works

### Pipeline overview

```
 product reviews (text + optional star rating)
              │
              ▼
 ┌─────────────────────────────┐
 │ Stage 1  FakeReviewClassifier│  P(fake) per review + highlighted phrases + reasons
 └─────────────────────────────┘
              │  keep reviews with P(fake) < threshold
              ▼
 ┌──────────────────────┐   ┌──────────────────────────────┐
 │ Stage 2  Summarizer   │   │ Stage 3  Aspect sentiment     │
 │ what buyers think     │   │ Sound: Great, Volume: Bad ... │
 └──────────────────────┘   └──────────────────────────────┘
              │                         │
              └────────────┬────────────┘
                           ▼
                ProductReport (stats, summary, aspects, per-review results)
```

### Stage 1: fake review classifier

File: `fakereview/classifier.py`, features in `fakereview/features.py`.

**Inputs to the model**

1. **TF-IDF word 1- and 2-grams**: up to 50,000 features, `min_df=2`, `max_df=0.9`, sublinear
   term frequency. These capture wording such as *"highly recommend"* or *"must have"*.
2. **22 dense features**, standardised (zero mean, unit variance):

| Group | Features |
|---|---|
| Length / structure | `n_words` (log), `n_sentences`, `avg_word_len`, `avg_sent_len`, `sent_len_std` |
| Vocabulary | `type_token_ratio` (richness), `stopword_ratio` |
| Voice | `first_person_ratio` (I, my…), `second_person_ratio` (you, your…) |
| Tone | `superlative_ratio` (best, perfect, amazing…), `hedge_ratio` (maybe, kinda, though…) |
| Punctuation / typing | `exclamations`, `questions`, `punct_ratio`, `upper_ratio`, `digit_ratio`, `lowercase_sentence_starts`, `repeated_chars` ("sooo") |
| Language model | `log_perplexity`: how predictable the wording is |
| Sentiment & rating | `sentiment`, `rating_extremity` (1 or 5 stars), `rating_mismatch` (5 stars on a negative text, or the reverse) |

**Perplexity.** Machine-generated text tends to be more predictable to a language model.
Two backends:
- `gpt2`: average token negative log-likelihood under `distilgpt2`.
- `bigram`: an interpolated bigram language model (λ = 0.7, add-one unigram backoff) trained
  only on **genuine** training reviews. It needs no dependencies. To avoid leakage, training-set
  scores are computed **out of fold** (5 folds): a review is never scored by an LM that saw it.
  Otherwise genuine training reviews would look unrealistically predictable and the classifier
  would learn that artefact.

**Model.** `LogisticRegression(C=2.0, class_weight="balanced", solver="liblinear")`.

**Why linear?** Every prediction can be explained exactly. For a review, each feature's
contribution to the fake score is `feature value × coefficient`:
- n-gram contributions give the **suspicious phrases** highlighted in the dashboard. Pure stop
  word n-grams are skipped, and overlapping phrases are merged so you see *"highly recommend"*
  rather than *"highly"* and *"recommend"* separately.
- dense feature contributions become **plain-English reasons** from the `REASONS` table, e.g.
  *"Star rating contradicts the sentiment of the text"* or *"No concrete numbers or specifics"*.

**Optional transformer fusion.** If `clf.transformer_scorer` is set (a fine-tuned
DistilRoBERTa from `finetune.py`, wrapped by `fakereview/transformer.py`), the final probability
is the **average** of the linear model's and the transformer's P(fake). Highlights and reasons
still come from the linear half, the part that can be explained.

**Performance.** On the Salminen test split (8,000 reviews) the linear model reaches about
**0.95 accuracy**. Run `train.py --ablation` to see the exact numbers and how much each
feature group contributes.

### Stage 2: opinion summary

File: `fakereview/summarizer.py`. Only **genuine** reviews are summarised.

- **Abstractive** (`facebook/bart-large-cnn`, used as-is without fine-tuning):
  1. Rank reviews by **centrality**: build a TF-IDF cosine-similarity graph between reviews and
     run PageRank (damping 0.85) on it. The highest-ranked reviews are the most representative.
  2. Pack the most central reviews into chunks of about 550 words (BART's 1,024-token input),
     at most 6 chunks.
  3. Summarise each chunk with beam search (4 beams, no repeated trigrams).
  4. If there were several chunks, summarise the concatenated chunk summaries once more.
- **Extractive** (fallback, no dependencies): split reviews into sentences of 4 to 40 words,
  rank them by the same TextRank centrality, then pick 4 with **MMR** (Maximal Marginal
  Relevance, λ = 0.7), which trades centrality against similarity to the sentences already
  chosen to avoid near-duplicates.

### Stage 3: aspect-based sentiment

File: `fakereview/absa.py`.

**Finding aspects**
- A built-in, editable lexicon (`DEFAULT_ASPECTS`) of 17 common aspects and their synonyms:
  Quality, Price / value, Battery life, Sound, Volume, Display, Performance, Camera, Design,
  Size / weight, Comfort / fit, Durability, Ease of use, Connectivity, Delivery / packaging,
  Customer service, Taste / smell.
- **Discovery** of product-specific aspects from patterns like *"the **hinge** is…"*,
  *"my **strap** feels…"*. Generic words (product, item, thing…) are ignored. A discovered
  aspect must appear in at least 2 reviews, and at most 5 are added.

**Matching mentions**
- Each review is split into sentences, then into **clauses** at *but, however, although,
  though, whereas, while, yet, except* and `;`. So in *"great sound but the volume is too
  low"*, "great" goes to Sound and "too low" goes to Volume.
- When terms overlap, the **longest match wins**: *"sound quality"* counts as Sound, not Quality.
- Within a clause, the comma-separated segment containing the aspect is used if it carries an
  opinion of its own (*"great quality, ok battery, weak volume"*).

**Scoring**
- `transformer` backend: the DeBERTa ABSA model classifies each (sentence, aspect) pair as
  Positive / Neutral / Negative. Score = P(positive) − P(negative).
- `lexicon` backend: the sentiment score of the opinion span.
- Each review casts **one vote per aspect** (the mean of its mentions), so one long review
  cannot dominate. An aspect needs votes from at least 2 reviews to be shown.
- Mean score → verdict: **Great** ≥ 0.5 > **Good** ≥ 0.25 > **OK** ≥ −0.1 > **Poor** ≥ −0.4 > **Bad**.
- Example snippets show the strongest opinions on both sides.

### Sentiment scorer

File: `fakereview/sentiment.py`. Returns a score in [−1, 1]. Uses **VADER** when
`vaderSentiment` is installed, otherwise a built-in VADER-style lexicon tuned for product reviews:
- about 180 word valences, plus review phrases scored first (*"stopped working"*,
  *"waste of money"*, *"too quiet"*, *"not loud enough"*, *"lasts all day"*…)
- negation within a 3-word window (`× −0.74`), intensifiers (*very, super, kinda…*)
- *but/however* weighting: words before it × 0.5, words after it × 1.5
- exclamation-mark boost, normalised with `x / sqrt(x² + 15)`

It is used for the classifier's sentiment features, the positive/negative shares of genuine
reviews, and lexicon ABSA.

### Pipeline and report

File: `fakereview/pipeline.py`.

- `FakeReviewDetective.classify()` runs the classifier and its explanations once.
- `FakeReviewDetective.build_report()` applies the threshold and runs summary and ABSA on the
  genuine reviews. It is separate from `classify()` so the dashboard can change the threshold
  without classifying again (Streamlit also caches both steps).
- `ProductReport` holds `reviews` (per-review `ReviewResult`s), `summary`, `aspects` and `stats`
  (counts, % fake, average rating before and after filtering, positive/negative share). It has
  helpers `to_text()`, `reviews_frame()` and `aspects_frame()`.

### Other modules

- `fakereview/data.py`: CSV loading with automatic column detection, the shared train/test
  split, and a tiny **synthetic** dataset used only for `train.py --demo`.
- `fakereview/text_utils.py`: regex-based sentence, clause, segment and word splitting, plus
  `highlight()`, which HTML-escapes a review and wraps the suspicious phrases in `<mark>`.
- `fakereview/transformer.py`: `TransformerScorer(path)`, a callable `texts → P(fake)`. Only
  the path is pickled, so the classifier file stays small.

---

## Backends: with and without transformers

Every heavy component falls back to a light one automatically when `transformers` and `torch`
are not installed. `auto` means "use the transformer if it can be loaded".

| Stage | Light backend (core install) | Transformer backend |
|---|---|---|
| Perplexity feature (training) | bigram LM on genuine reviews | `distilbert/distilgpt2` |
| Fake classifier (optional add-on) | n/a | fine-tuned DistilRoBERTa (`finetune.py`) |
| Opinion summary | TextRank + MMR (extractive) | `facebook/bart-large-cnn` (abstractive) |
| Aspect sentiment | clause-level lexicon scoring | `yangheng/deberta-v3-base-absa-v1.1` |
| Sentiment | built-in lexicon | (VADER if `vaderSentiment` is installed, not a transformer) |

The perplexity backend is fixed when the classifier is **trained** and saved with it. The
included `models/classifier.joblib` uses **bigram** perplexity, so it does not need GPT-2 to run.

---

## Data

**Training data**: the *Fake Reviews Dataset* (Salminen, J., Kandpal, C., Kamel, A. M.,
Jung, S., & Jansen, B. J., 2022, *"Creating and detecting fake reviews of online products"*,
Journal of Retailing and Consumer Services). It has 40,432 Amazon reviews across 10 product
categories, half original human reviews (`OR`) and half computer-generated with GPT-2 (`CG`).
Columns: `category, rating, label, text_`. It is included at `data/fake_reviews_dataset.csv`.
The original source is OSF, and it is also on Kaggle.

**Sample product**: `data/sample_product_reviews.csv` contains 22 hand-written reviews of a
portable Bluetooth speaker. Some are genuine-sounding, some are fake-sounding. It is used as the
dashboard and demo default.

---

## Input CSV format

### For analysis (dashboard / `demo.py`): one product's reviews

| Column | Required | Accepted names (case-insensitive) |
|---|---|---|
| review text | yes | `text_`, `text`, `review`, `review_text`, `reviewText`, `content`, `body` |
| star rating (1–5) | no | `rating`, `stars`, `score`, `overall`, `star_rating` |

```csv
text,rating
"Sound is great but the max volume is too low for outdoors.",4
"Absolutely amazing product!!! Highly recommend to everyone!",5
```

In the dashboard you can pick any column by hand.

### For training (`train.py` / `finetune.py`): labelled reviews

The same text and rating columns, plus a label column named `label`, `is_fake`, `fake`, `class`
or `target`. A row counts as **fake** if its label is one of `CG, fake, deceptive, 1, true,
spam, computer-generated, ai` (case-insensitive). Anything else counts as genuine.

---

## Limitations and caveats

- **"Fake" here means machine-generated.** The training data's fakes were generated with GPT-2.
  The model learns what GPT-2-era generated reviews look like. It is not trained on paid human
  fake reviews, or on text from modern LLMs, which can be much harder to detect.
- **GPT-2 perplexity is partly leaky on this dataset**, because the fakes were generated by
  GPT-2 itself, so GPT-2 finds them unusually predictable. That is why the shipped model uses
  the **bigram** perplexity. Compare `--perplexity gpt2` with `--perplexity bigram` (and use
  `--ablation`) to measure how much it contributes.
- **Threshold trade-off**: a high threshold lets more (possibly fake) reviews into the summary.
  A low one discards genuine reviews. Use the dashboard slider to see the effect.
- **Lexicon aspect sentiment** misses sarcasm and facts stated without an opinion (*"6 hours
  at half volume"*). The DeBERTa ABSA backend handles these better.
- The aspect lexicon is for consumer products in English. Extend `DEFAULT_ASPECTS` in
  `fakereview/absa.py` for other domains.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No trained classifier at models/classifier.joblib` | Run from the project root, or train one: `python train.py --data data/fake_reviews_dataset.csv --perplexity bigram` |
| `InconsistentVersionWarning` / error loading the `.joblib` | scikit-learn version mismatch: `pip install scikit-learn==1.9.1`, or retrain as above |
| First run is very slow / downloads gigabytes | transformers is installed, so BART/DeBERTa are downloading. Choose the extractive/lexicon backends, or uninstall `transformers` |
| `streamlit: command not found` | Activate the venv, or use `python -m streamlit run app.py` |
| PowerShell won't run `Activate.ps1` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or call `.\.venv\Scripts\python.exe` directly |
| Model can't find `models/transformer` | Fine-tuned weights aren't in git. Run `finetune.py` (see FINETUNING.md), and run commands from the project root |
| Out of memory with transformer backends | Use the light backends, or a machine with more RAM / a GPU |
