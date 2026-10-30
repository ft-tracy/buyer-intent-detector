"""Step 3: train and evaluate the comment classifier.

Usage:  python train.py

Reads   data/labeled.csv         (real comments YOU labeled; columns: text,label)   [optional]
        data/synthetic_seed.csv  (synthetic examples; columns: text,label)           [optional]
Writes  model.joblib, artifacts/metrics.json, artifacts/confusion_matrix.png

Method: TF-IDF on words (1-2 grams) + characters (2-5 grams) -> Logistic Regression.
Character n-grams help with spelling variations, Swahili and Sheng, which word-only models miss.
"""
import json
import re
from pathlib import Path

import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
MODEL_PATH = ROOT / "model.joblib"

LABELS = ["buying_intent", "price_payment", "trust_concern", "complaint", "praise", "other"]
LABEL_NAMES = {
    "buying_intent": "Buying intent",
    "price_payment": "Price / payment question",
    "trust_concern": "Trust / scam concern",
    "complaint": "Complaint",
    "praise": "Praise",
    "other": "Other / noise",
}
ACTIONS = {
    "buying_intent": "Hot lead: reply quickly with a WhatsApp link and offer a site visit.",
    "price_payment": "Reply with pricing and payment-plan details and link the property page.",
    "trust_concern": "Reply calmly with title-deed proof, offer an official search or site visit, share testimonials.",
    "complaint": "Escalate to customer care, reply publicly with empathy, move details to DM or WhatsApp.",
    "praise": "Thank them; consider asking permission to feature them as a testimonial.",
    "other": "No action needed. Monitor.",
}
REVIEW_THRESHOLD = 0.35  # below this confidence the app asks for a human to check


def clean(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+", " url ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_data():
    frames = []
    for filename, source in [("labeled.csv", "real"), ("synthetic_seed.csv", "synthetic")]:
        path = DATA / filename
        if path.exists():
            d = pd.read_csv(path)[["text", "label"]].dropna()
            d["source"] = source
            frames.append(d)
    if not frames:
        raise FileNotFoundError("Put data/labeled.csv and/or data/synthetic_seed.csv in place first.")
    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(LABELS)].drop_duplicates(subset="text")
    return df.reset_index(drop=True)


def build_pipeline():
    features = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=clean, ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=clean, analyzer="char_wb", ngram_range=(2, 5),
                                 min_df=2, sublinear_tf=True)),
    ])
    clf = LogisticRegression(max_iter=3000, C=5.0, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def _stratify(d):
    counts = d["label"].value_counts()
    return d["label"] if len(counts) > 1 and counts.min() >= 2 else None


def split(df):
    """If you have enough real labeled comments, test ONLY on real ones.
    Synthetic examples then only help training and cannot inflate the score."""
    real, synth = df[df.source == "real"], df[df.source == "synthetic"]
    if len(real) >= 60 and real["label"].nunique() > 1:
        tr, te = train_test_split(real, test_size=0.25, stratify=_stratify(real), random_state=42)
        return pd.concat([tr, synth]), te, "Test set contains real comments only"
    tr, te = train_test_split(df, test_size=0.2, stratify=_stratify(df), random_state=42)
    return tr, te, "Test set mixes real and synthetic examples (add more real labels for a stricter test)"


def top_terms(model, k=8):
    names = model.named_steps["features"].get_feature_names_out()
    coefs = model.named_steps["clf"].coef_
    out = {}
    for i, cls in enumerate(model.named_steps["clf"].classes_):
        terms = []
        for j in coefs[i].argsort()[::-1]:
            if names[j].startswith("word__"):
                terms.append(names[j][len("word__"):])
            if len(terms) == k:
                break
        out[cls] = terms
    return out


def predict(model, texts):
    """Returns a list of (label, confidence)."""
    probs = model.predict_proba(list(texts))
    classes = model.classes_
    return [(classes[i], float(probs[r, i])) for r, i in enumerate(probs.argmax(axis=1))]


def cross_validate_real(df, n_splits=5):
    """More stable estimate than one small test split: 5-fold cross-validation on the REAL comments.
    Synthetic examples are added to every training fold but never scored."""
    real, synth = df[df.source == "real"], df[df.source == "synthetic"]
    if len(real) < 60:
        return None
    counts = real["label"].value_counts()
    strat = real["label"].where(real["label"].map(counts) >= n_splits, "rare")
    pred = pd.Series(index=real.index, dtype=object)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=42).split(real, strat):
            fold_train = pd.concat([real.iloc[tr], synth])
            model = build_pipeline().fit(fold_train["text"], fold_train["label"])
            pred.iloc[va] = model.predict(real.iloc[va]["text"])
    rep = classification_report(real["label"], pred, output_dict=True, zero_division=0)
    enough = [l for l in LABELS if counts.get(l, 0) >= 10]
    return {
        "n": int(len(real)),
        "accuracy": round(rep["accuracy"], 3),
        "classes_scored": enough,
        "macro_f1_scored_classes": round(float(np.mean([rep[l]["f1-score"] for l in enough])), 3),
        "per_class": {l: {"f1-score": round(rep[l]["f1-score"], 3), "support": int(rep[l]["support"])}
                      for l in LABELS if l in rep},
    }


def train_and_save():
    df = load_data()
    train_df, test_df, split_note = split(df)

    model = build_pipeline().fit(train_df["text"], train_df["label"])
    pred = model.predict(test_df["text"])
    report = classification_report(test_df["label"], pred, output_dict=True, zero_division=0)
    print(classification_report(test_df["label"], pred, zero_division=0))

    ART.mkdir(exist_ok=True)
    shown = [l for l in LABELS if l in set(test_df["label"]) | set(pred)]
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(test_df["label"], pred, labels=shown, ax=ax,
                                            xticks_rotation=45, colorbar=False)
    fig.tight_layout()
    fig.savefig(ART / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Final model: refit on ALL labeled data so it learns from every example.
    final = build_pipeline().fit(df["text"], df["label"])
    joblib.dump(final, MODEL_PATH)

    metrics = {
        "n_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_real": int((df.source == "real").sum()),
        "n_synthetic": int((df.source == "synthetic").sum()),
        "split_note": split_note,
        "accuracy": round(report["accuracy"], 3),
        "macro_f1": round(report["macro avg"]["f1-score"], 3),
        "per_class": {k: {m: round(v[m], 3) for m in ("precision", "recall", "f1-score")} | {"support": int(v["support"])}
                      for k, v in report.items() if k in LABELS},
        "class_counts": df["label"].value_counts().to_dict(),
        "class_counts_real": df[df.source == "real"]["label"].value_counts().to_dict(),
        "class_counts_synthetic": df[df.source == "synthetic"]["label"].value_counts().to_dict(),
        "cv": cross_validate_real(df),
        "top_terms": top_terms(final),
        "sklearn_version": sklearn.__version__,
    }
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved {MODEL_PATH.name} and artifacts/metrics.json")
    return final, metrics


if __name__ == "__main__":
    train_and_save()
