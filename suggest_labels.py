"""Step 2: prepare a labeling sheet with rule-based SUGGESTED labels.

The rules are deliberately simple. They save you typing, but they are often wrong.
YOUR job is to review every row and fill the `label` column with the correct category.
Those hand-checked labels are what make this project yours.

Usage:
    python suggest_labels.py --n 500
Output: data/to_label.csv   -> open in Google Sheets or Excel, fill `label`, then
        save only the columns text,label as data/labeled.csv
"""
import argparse
import re

import pandas as pd

# Order matters: the first matching rule wins.
RULES = [
    ("trust_concern", ["scam", "conned", "fake", "genuine", "legit", "verified", "dispute", "court case",
                       "tapeli", "mtapeli", "kweli", "proof", "fraud", "cheat", "licensed", "registered",
                       "is it real", "real title", "pigwa", "official search"]),
    ("complaint", ["still no", "not received", "delay", "no response", "not responding", "disappointed",
                   "refund", "terrible", "worst", "poor service", "poor communication", "hamjibu",
                   "sijapata", "waiting for", "nobody", "no one"]),
    ("price_payment", ["how much", "price", "bei", "deposit", "installment", "instalment", "payment plan",
                       "m-pesa", "mpesa", "cost", "ngapi", "lipa", "fees", "pay in", "payments from"]),
    ("buying_intent", ["interested", "want to buy", "i want to", "book", "site visit", "reserve",
                       "contact me", "dm me", "whatsapp", "nataka", "how do i start", "call me",
                       "send me details", "tembelea", "add me"]),
    ("praise", ["thank", "asante", "congrat", "great", "amazing", "love", "god bless", "kazi nzuri",
                "proud", "best", "informative", "inspiring"]),
]


def suggest(text):
    t = str(text).lower()
    for label, words in RULES:
        if any(w in t for w in words):
            return label
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="how many comments to label")
    args = parser.parse_args()

    df = pd.read_csv("data/raw_comments.csv")
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].str.len() >= 3].drop_duplicates("text")

    # Take a random sample so the labeling set is not dominated by one video.
    df = df.sample(n=min(args.n, len(df)), random_state=42)
    df["suggested_label"] = df["text"].apply(suggest)
    df["label"] = ""  # <- you fill this in
    out = df[["text", "suggested_label", "label", "video_title"]]
    out.to_csv("data/to_label.csv", index=False)
    print(f"Wrote {len(out)} rows to data/to_label.csv")
    print(out["suggested_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
