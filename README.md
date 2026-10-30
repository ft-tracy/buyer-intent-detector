# Buyer Intent & Trust Concern Detector

A portfolio project built for a social media marketing application. It classifies public social media
comments for a land and real-estate brand into six categories so a marketing team can spot hot leads,
answer trust concerns and escalate complaints.

> Independent portfolio project using public comments and clearly labeled synthetic examples.
> Not an official AMG Realtors tool and not built with their internal data.

**Live demo:** _add your Streamlit link here_

## Categories
Buying intent, Price / payment question, Trust / scam concern, Complaint, Praise, Other / noise.
Each prediction comes with a confidence score and a suggested action for the community manager.
Low-confidence predictions are flagged for human review.

## Data
- 336 public YouTube comments from the channel, collected with the YouTube Data API (author names not stored),
  labeled by hand. A small set of synthetic examples (English, Swahili, Sheng-style, 10 per class) is added to
  training only. Synthetic rows are never used for scoring.
- Real label mix: other 182, praise 108, buying intent 25 (including questions showing interest), price/payment 19,
  trust concern 2, complaint 0. Scam concerns and complaints are rare in real comments, so those two classes
  are learned mostly from synthetic examples. The app says so in its Methodology tab.
- Raw and labeled comment text is not committed to this repo.

## Results (honest summary)
5-fold cross-validation on the real comments: accuracy about 0.75; macro F1 about 0.68 over the four classes
with 10+ real examples. Main error source: the praise vs other boundary for very short comments.

## Method
1. Clean text and build TF-IDF features on words (1-2 grams) and characters (2-5 grams).
2. Class-balanced logistic regression.
3. Evaluate on a held-out test set: accuracy, per-class precision/recall/F1, confusion matrix.
4. Refit on all labeled data for the final model.

Results and limitations are shown in the app's Methodology tab (`artifacts/metrics.json`).

## Run it yourself
```bash
pip install -r requirements.txt
python train.py            # trains and writes model.joblib + artifacts/
streamlit run app.py
```

To rebuild the dataset from scratch:
```bash
export YOUTUBE_API_KEY="your-key"
python collect_comments.py --max-videos 60 --max-per-video 200
python suggest_labels.py --n 500      # then fix the `label` column by hand
# save text,label columns as data/labeled.csv
python train.py
```

## Project layout
| File | Purpose |
|---|---|
| `collect_comments.py` | Pulls public comments via the YouTube Data API |
| `suggest_labels.py` | Creates a labeling sheet with rule-based suggestions to review |
| `train.py` | Trains, evaluates and saves the model |
| `app.py` | Streamlit app: single comment, bulk analysis, methodology |
| `data/synthetic_seed.csv` | Synthetic labeled examples |
| `data/demo_comments.csv` | Synthetic comments for the demo tab |

## Limitations
Small dataset, one annotator, hard cases (sarcasm, code-switching, very short comments), and
suggested actions are guidance for a human, not automatic replies.

## Next steps
Multilingual sentence embeddings, more labeled data, a second annotator to measure agreement,
and a link into a cross-platform analytics dashboard.
