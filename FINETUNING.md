# Fine-tuning the transformer classifier

## Do you need it?

Yes, if you want the system to match the project description ("a classifier combining
linguistic features ... with a **fine-tuned transformer model**"). Everything else works without it:

| Stage | Needs fine-tuning? | What runs |
|---|---|---|
| Fake review classifier (linear part) | No: trained, not fine-tuned (`train.py`) | TF-IDF + 22 features → logistic regression |
| Fake review classifier (transformer part) | **Yes** (`finetune.py`) | DistilRoBERTa fine-tuned on the Salminen dataset |
| Opinion summary | No | `facebook/bart-large-cnn`, pretrained |
| Aspect sentiment | No | `yangheng/deberta-v3-base-absa-v1.1`, pretrained |

The linear model alone already gets about 0.95 accuracy on the test split. The fine-tuned
transformer adds a few points and gives you the model-comparison table for the report.

Your laptop has no NVIDIA GPU and its CPU runs PyTorch on 2 threads, so a full fine-tune there
would take many hours. **Use Google Colab with a free T4 GPU.** A 2-epoch run on all 32k training
reviews takes about 15–25 minutes there.

The code is already written (`finetune.py`, `fakereview/transformer.py`) and was tested
end to end on a small CPU run. You only need to run it on a GPU.

---

## Steps (Google Colab)

### 1. Zip the project (on your laptop)

Zip the `fakereview_detective` folder **without** `.venv` (it is large and Windows-only).
In PowerShell, from the folder that contains `fakereview_detective`:

```powershell
Compress-Archive -Path fakereview_detective\app.py, fakereview_detective\demo.py, fakereview_detective\train.py, fakereview_detective\finetune.py, fakereview_detective\requirements.txt, fakereview_detective\fakereview, fakereview_detective\data -DestinationPath fakereview_detective.zip -Force
```

(`data/` includes `fake_reviews_dataset.csv`, already downloaded from OSF.)

### 2. Open Colab with a GPU

1. Go to https://colab.research.google.com → **New notebook**.
2. **Runtime → Change runtime type → T4 GPU → Save**.

### 3. Upload and unzip

Run in a cell:

```python
from google.colab import files
files.upload()          # pick fakereview_detective.zip
!mkdir -p fakereview_detective && cd fakereview_detective && unzip -oq ../fakereview_detective.zip
%cd fakereview_detective
!ls
```

`!ls` should show `app.py  data  demo.py  fakereview  finetune.py  requirements.txt  train.py`.
If the zip already contained a `fakereview_detective/` folder level, `%cd` into it one more time.

### 4. Install packages

Colab already has torch with CUDA. Install the rest:

```python
!pip install -q "transformers>=4.40" scikit-learn pandas joblib sentencepiece protobuf vaderSentiment
import torch; print(torch.cuda.is_available())   # must print True
```

### 5. Fine-tune

```python
!python finetune.py --data data/fake_reviews_dataset.csv --epochs 2
```

Expected output: progress lines every 50 steps, then something like

```
test acc=0.97x  f1=0.97x  auc=0.99x
saved -> models/transformer
```

Write down the test numbers for your report. Options if you want to experiment:
`--model FacebookAI/roberta-base` (bigger, slower, usually slightly better), `--epochs 3`, `--batch-size 32`.

### 6. Train the linear model and fuse it with the transformer

Still in Colab (it is faster than your laptop):

```python
!python train.py --data data/fake_reviews_dataset.csv --ablation --transformer models/transformer
```

This prints the full comparison ladder on the same test split:

```
TF-IDF + dense (full)        acc=...
TF-IDF only                  acc=...
Dense features only          acc=...
Fine-tuned transformer       acc=...
Ensemble (linear + transf.)  acc=...
```

and saves `models/classifier.joblib` with the transformer attached (it stores the path
`models/transformer`, not the weights).

Without `--perplexity bigram` this uses GPT-2 (distilgpt2) perplexity. Optional, for the
report's caveat about GPT-2 perplexity on GPT-2-generated fakes:

```python
!python train.py --data data/fake_reviews_dataset.csv --perplexity bigram --transformer models/transformer --out models/classifier_bigram.joblib
```

### 7. Download the models

```python
!zip -rq models.zip models
files.download("models.zip")
```

### 8. Put them in the project (on your laptop)

Unzip `models.zip` into the project so you get:

```
fakereview_detective/
  models/
    classifier.joblib
    transformer/
      config.json
      model.safetensors
      tokenizer.json ...
```

Replace any existing `models/classifier.joblib` (back it up first if you want to keep the
linear-only one, e.g. rename it to `classifier_linear.joblib`).

### 9. Run it

```powershell
.\.venv\Scripts\python.exe demo.py
.\.venv\Scripts\streamlit.exe run app.py
```

Run both from the project folder: the classifier finds the transformer through the relative
path `models/transformer`. Fake probabilities shown in the dashboard are now the average of the
linear model and the fine-tuned transformer. Highlighted phrases and reasons still come from
the linear model (it is the explainable half of the ensemble).

---

## Running it locally instead (not recommended)

Same commands with the venv's Python. A small run to check things work (about 10 minutes on your laptop):

```powershell
.\.venv\Scripts\python.exe finetune.py --data data\fake_reviews_dataset.csv --sample 2000 --epochs 1 --max-len 128 --out models\transformer_smoke
.\.venv\Scripts\python.exe train.py --data data\fake_reviews_dataset.csv --sample 2000 --perplexity bigram --transformer models\transformer_smoke --out models\classifier_smoke.joblib
```

`--sample` must be the same number in both commands, otherwise the two models are trained and
tested on different splits and the ensemble numbers are leaky. A model trained on 1,600 reviews
for one epoch is only a smoke test; do not use it for the report.

## What the code does

`finetune.py`:
1. Loads the dataset and makes the **same** 80/20 split as `train.py` (`fakereview.data.train_test`,
   seed 42, stratified), so the transformer never trains on the linear model's test reviews.
2. Fine-tunes `distilbert/distilroberta-base` with a 2-class head (0 = genuine, 1 = fake): AdamW, lr 2e-5,
   weight decay 0.01, linear schedule with 6% warm-up, gradient clipping at 1.0, max 256 tokens.
3. Saves model + tokenizer to `models/transformer`, then reports test accuracy / F1 / ROC-AUC.

`fakereview/transformer.py`: `TransformerScorer(path)` is a callable `texts -> P(fake)`. Setting
`clf.transformer_scorer = TransformerScorer(path)` makes `FakeReviewClassifier.predict_proba`
average it with the linear model's probability. Only the path is pickled.
