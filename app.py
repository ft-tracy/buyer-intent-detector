"""Step 4: the interactive app.   Run locally with:  streamlit run app.py"""
import json

import joblib
import pandas as pd
import streamlit as st

from train import (ACTIONS, ART, DATA, LABEL_NAMES, MODEL_PATH, REVIEW_THRESHOLD,
                   predict, train_and_save)

st.set_page_config(page_title="Buyer Intent & Trust Concern Detector", page_icon="🏡", layout="wide")


@st.cache_resource
def get_model():
    """Load the saved model; if it is missing or from a different scikit-learn version, retrain."""
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass
    model, _ = train_and_save()
    return model


def classify_df(df, col):
    preds = predict(model, df[col].astype(str).tolist())
    out = df.copy()
    out["category"] = [LABEL_NAMES[p[0]] for p in preds]
    out["confidence"] = [round(p[1], 2) for p in preds]
    out["needs_review"] = out["confidence"] < REVIEW_THRESHOLD
    actions = [ACTIONS[p[0]] for p in preds]
    out["suggested_action"] = [a if not nr else "Low confidence: review manually." for a, nr in zip(actions, out["needs_review"])]
    return out


model = get_model()

st.title("🏡 Buyer Intent & Trust Concern Detector")
st.caption("Portfolio project: classifies public social media comments for a land and real-estate brand "
           "so a marketing team can spot hot leads, answer trust concerns and escalate complaints. "
           "Not an official AMG Realtors tool.")

tab_try, tab_bulk, tab_method = st.tabs(["Try a comment", "Analyse many comments", "Methodology"])

# ---------------------------------------------------------------- Tab 1
with tab_try:
    examples = [
        "",
        "How much is the 1/8 acre in Juja and can I pay in installments?",
        "Are your title deeds genuine? I have heard of so many scams",
        "Nataka kutembelea site Nanyuki wikendi hii",
        "I paid two months ago and nobody is picking my calls",
        "Asante AMG, nimepata title yangu leo!",
        "Who is watching this in 2026?",
    ]
    choice = st.selectbox("Pick an example (or type your own below)", examples)
    text = st.text_area("Comment", value=choice, height=100, placeholder="Paste a comment here...")
    if text.strip():
        label, conf = predict(model, [text])[0]
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Predicted category", LABEL_NAMES[label])
            st.metric("Confidence", f"{conf:.0%}")
            if conf < REVIEW_THRESHOLD:
                st.warning("Low confidence. A human should check this one.")
        with c2:
            st.markdown("**Suggested action**")
            st.info(ACTIONS[label])
            probs = pd.Series(model.predict_proba([text])[0],
                              index=[LABEL_NAMES[c] for c in model.classes_])
            st.bar_chart(probs.sort_values(ascending=False))

# ---------------------------------------------------------------- Tab 2
with tab_bulk:
    st.write("Upload a CSV of comments (one column of text, optionally a date column), or use the demo file.")
    source = st.radio("Data", ["Demo comments (synthetic)", "Upload my CSV"], horizontal=True)
    df = None
    if source.startswith("Demo"):
        df = pd.read_csv(DATA / "demo_comments.csv")
    else:
        up = st.file_uploader("CSV file", type="csv")
        if up is not None:
            df = pd.read_csv(up)
            st.caption("Tip: do not upload personal data. Public comment text only.")

    if df is not None and len(df):
        text_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
        if not text_cols:
            st.error("No text column found in this file.")
            st.stop()
        default = text_cols.index("text") if "text" in text_cols else 0
        col = st.selectbox("Which column contains the comment text?", text_cols, index=default)
        date_options = ["None"] + list(df.columns)
        date_default = date_options.index("published_at") if "published_at" in date_options else 0
        date_col = st.selectbox("Date column (optional, enables trend chart)", date_options, index=date_default)

        res = classify_df(df, col)
        keep = st.multiselect("Filter categories", sorted(res["category"].unique()),
                              default=sorted(res["category"].unique()))
        view = res[res["category"].isin(keep)]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Comments", len(res))
        k2.metric("Hot leads (intent + price)", int(res["category"].isin([LABEL_NAMES["buying_intent"], LABEL_NAMES["price_payment"]]).sum()))
        k3.metric("Trust concerns", int((res["category"] == LABEL_NAMES["trust_concern"]).sum()))
        k4.metric("Complaints", int((res["category"] == LABEL_NAMES["complaint"]).sum()))

        left, right = st.columns(2)
        with left:
            st.subheader("Category mix")
            st.bar_chart(res["category"].value_counts())
        with right:
            if date_col != "None":
                st.subheader("Trend by month")
                dates = pd.to_datetime(res[date_col], errors="coerce", utc=True).dt.tz_localize(None)
                trend = (res.assign(month=dates.dt.to_period("M").astype(str))
                         .dropna(subset=["month"]).groupby(["month", "category"]).size().unstack(fill_value=0))
                st.bar_chart(trend)

        st.subheader("Comments")
        st.dataframe(view[[col, "category", "confidence", "suggested_action"]])
        st.download_button("Download results (CSV)", res.to_csv(index=False).encode("utf-8"),
                           "classified_comments.csv", "text/csv")

# ---------------------------------------------------------------- Tab 3
with tab_method:
    path = ART / "metrics.json"
    if not path.exists():
        st.write("Run `python train.py` to generate metrics.")
    else:
        m = json.loads(path.read_text())
        st.subheader("Problem")
        st.write("Brands selling land get comments that mix real buying intent, price questions, fear of scams, "
                 "complaints, praise and noise. Sorting them by hand does not scale, and missed leads or unanswered "
                 "trust concerns cost sales.")
        st.subheader("Data")
        st.write(f"{m['n_total']} labeled comments: {m['n_real']} real public comments labeled by hand, "
                 f"{m['n_synthetic']} synthetic examples (written for this project, clearly labeled). "
                 f"{m['split_note']}.")
        st.subheader("Method")
        st.write("Text is turned into TF-IDF features on words and characters (character n-grams cope with spelling "
                 "variation, Swahili and Sheng), then classified with class-balanced logistic regression. "
                 "Predictions under a confidence threshold are flagged for human review.")
        st.subheader("Results on held-out test set")
        a, b, c = st.columns(3)
        a.metric("Accuracy", m["accuracy"])
        b.metric("Macro F1", m["macro_f1"])
        c.metric("Test comments", m["n_test"])
        st.dataframe(pd.DataFrame(m["per_class"]).T)
        if m.get("cv"):
            cv = m["cv"]
            st.subheader("More stable check: 5-fold cross-validation on real comments")
            a1, b1 = st.columns(2)
            a1.metric("Accuracy", cv["accuracy"])
            b1.metric("Macro F1 (classes with 10+ real examples)", cv["macro_f1_scored_classes"])
            st.caption("Scored classes: " + ", ".join(cv["classes_scored"]) +
                       ". Classes with fewer real examples are not scored because the result would be meaningless.")
        st.subheader("Real vs synthetic training examples per class")
        counts = pd.DataFrame({"real": m.get("class_counts_real", {}),
                               "synthetic": m.get("class_counts_synthetic", {})}).fillna(0).astype(int)
        st.dataframe(counts)
        st.caption("Classes with very few real examples (here: trust concern and complaint) are learned mostly "
                   "from synthetic examples, so treat their predictions as indicative only.")
        cm = ART / "confusion_matrix.png"
        if cm.exists():
            st.image(str(cm), caption="Confusion matrix (rows: true, columns: predicted)", width=500)
        st.subheader("What the model keys on (top words per category)")
        st.json(m["top_terms"])
        st.subheader("Limitations")
        st.write("- Small dataset; scores will move as more comments are labeled.\n"
                 "- Real comments on a brand's channel are mostly praise and noise, so scam concerns and "
                 "complaints are rare in the real data.\n"
                 "- The praise vs other boundary is subjective for very short comments (\"Wow\", \"Beautiful\").\n"
                 "- Sarcasm, mixed-language slang and very short comments are hard.\n"
                 "- Labels reflect one annotator's judgement.\n"
                 "- Suggested actions are guidance for a human, not automatic replies.\n"
                 "- Built on public comments only; no private customer data.")
