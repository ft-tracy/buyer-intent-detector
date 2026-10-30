"""After you fill the `new_label` column in label_review.csv, run:  python apply_review.py
It updates data/labeled.csv for the rows where you typed a new label, then you retrain with python train.py.
Leave new_label blank to keep your original label. Allowed: buying_intent, price_payment, trust_concern,
complaint, praise, other."""
import pandas as pd

VALID = ["buying_intent", "price_payment", "trust_concern", "complaint", "praise", "other"]
lab = pd.read_csv("data/labeled.csv")
rev = pd.read_csv("label_review.csv", dtype={"new_label": str})
rev["new_label"] = rev["new_label"].fillna("").astype(str).str.strip().str.lower()
todo = rev[rev["new_label"] != ""]
bad = todo[~todo["new_label"].isin(VALID)]
if len(bad):
    print("These new_label values are not valid and were skipped:\n", bad[["row_id", "new_label"]])
todo = todo[todo["new_label"].isin(VALID)]
changed = 0
for _, r in todo.iterrows():
    if lab.loc[r.row_id, "label"] != r.new_label:
        lab.loc[r.row_id, "label"] = r.new_label
        changed += 1
lab.to_csv("data/labeled.csv", index=False)
print(f"Updated {changed} labels. New class counts:\n{lab['label'].value_counts().to_string()}")
