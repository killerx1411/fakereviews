"""Dataset loading.

Recommended training data: the "Fake Reviews Dataset" (Salminen et al., 2022) -
40k Amazon reviews, half original (OR) and half computer-generated (CG) by GPT-2,
with columns category, rating, label, text_. Available on OSF and Kaggle as
fake_reviews_dataset.csv. Any CSV with a text column, a label column and
(optionally) a rating column also works.
"""
import random

import pandas as pd

TEXT_COLS = ["text_", "text", "review", "review_text", "reviewText", "content", "body"]
LABEL_COLS = ["label", "is_fake", "fake", "class", "target"]
RATING_COLS = ["rating", "stars", "score", "overall", "star_rating"]
FAKE_VALUES = {"cg", "fake", "deceptive", "1", "true", "spam", "computer-generated", "ai"}


def find_col(df, candidates, required=True):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise KeyError(f"None of {candidates} found in columns {list(df.columns)}")
    return None


def load_labelled(path) -> pd.DataFrame:
    """Returns a frame with columns text, rating, label (1 = fake)."""
    df = pd.read_csv(path)
    t, l = find_col(df, TEXT_COLS), find_col(df, LABEL_COLS)
    r = find_col(df, RATING_COLS, required=False)
    out = pd.DataFrame({
        "text": df[t].astype(str),
        "rating": pd.to_numeric(df[r], errors="coerce") if r else float("nan"),
        "label": df[l].astype(str).str.strip().str.lower().isin(FAKE_VALUES).astype(int),
    })
    if "category" in df.columns:
        out["category"] = df["category"]
    return out.dropna(subset=["text"]).reset_index(drop=True)


def train_test(df, sample: int = 0):
    """The one train/test split used by train.py and finetune.py, so the fine-tuned
    transformer never sees the linear model's test reviews (and vice versa)."""
    from sklearn.model_selection import train_test_split

    if sample:
        df = df.sample(min(sample, len(df)), random_state=42).reset_index(drop=True)
    return train_test_split(df, test_size=0.2, random_state=42, stratify=df.label)


def load_reviews(path) -> pd.DataFrame:
    """Unlabelled product reviews: returns columns text, rating."""
    df = pd.read_csv(path)
    t = find_col(df, TEXT_COLS)
    r = find_col(df, RATING_COLS, required=False)
    return pd.DataFrame({
        "text": df[t].astype(str),
        "rating": pd.to_numeric(df[r], errors="coerce") if r else float("nan"),
    })


# --------------------------------------------------------------------------- synthetic
# ONLY for smoke-testing the code path without downloading the real dataset.
# Numbers measured on this data mean nothing.
_G_OPEN = ["ok so", "bought this for my", "got it last week for my", "been using it a month,", "honestly", "so"]
_G_OBJ = ["daughter", "kitchen", "desk", "car", "dad", "garage", "dorm room"]
_G_BODY = [
    "it works fine but the {a} is kinda {neg}", "the {a} is {pos} for the price",
    "{a} died after {n} weeks which sucks", "took {n} mins to set up, {a} is {pos} though",
    "returned the first one bc the {a} was {neg}, replacement is better",
    "{a} could be better tbh", "not bad, {a} is {pos} but it gets hot",
]
_F_BODY = [
    "This product is absolutely {sup}!", "I highly recommend it to everyone.",
    "The {a} is {sup} and it exceeded all my expectations.", "It is a must-have for anyone.",
    "Five stars, I would definitely buy it again!", "You will not be disappointed with this purchase.",
    "The {a} is perfect and the quality is outstanding.", "This is the best purchase I have ever made.",
    "It works perfectly and looks great.", "Great product, great price, great quality!",
]
_ASP = ["battery", "sound", "volume", "handle", "lid", "strap", "screen", "cable"]


def synthetic(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rnd = random.Random(seed)
    rows = []
    for _ in range(n // 2):
        a = rnd.choice(_ASP)
        body = rnd.sample(_G_BODY, 2)
        txt = f"{rnd.choice(_G_OPEN)} {rnd.choice(_G_OBJ)}. " + ". ".join(
            b.format(a=a, n=rnd.randint(2, 12), pos=rnd.choice(["good", "decent", "solid", "nice"]),
                     neg=rnd.choice(["weak", "flimsy", "meh", "low"])) for b in body)
        if rnd.random() < 0.5:  # real people capitalise sometimes
            txt = ". ".join(x.strip().capitalize() for x in txt.split(". "))
        rows.append((txt, rnd.choice([2, 3, 4, 4, 5]), 0))
    for _ in range(n // 2):
        a = rnd.choice(_ASP)
        txt = " ".join(s.format(a=a, sup=rnd.choice(["amazing", "excellent", "fantastic", "incredible"]))
                       for s in rnd.sample(_F_BODY, 3))
        rows.append((txt, rnd.choice([5, 5, 5, 4, 1]), 1))
    rnd.shuffle(rows)
    return pd.DataFrame(rows, columns=["text", "rating", "label"])
