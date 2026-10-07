"""FakeReview Detective dashboard.

    streamlit run app.py
"""
import io
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from fakereview.data import RATING_COLS, TEXT_COLS, find_col
from fakereview.pipeline import FakeReviewDetective
from fakereview.text_utils import highlight

SAMPLE = Path("data/sample_product_reviews.csv")
VERDICT_COLORS = {"Great": "#2E7D5B", "Good": "#7AAE6E", "OK": "#C9A227", "Poor": "#D9822B", "Bad": "#B83A2E"}

st.set_page_config(page_title="FakeReview Detective", page_icon="🔎", layout="wide")
st.markdown("""
<style>
.review {border-left: 4px solid #7AAE6E; padding: .6rem .9rem; margin: .5rem 0 .9rem;
         background: rgba(127,127,127,.06); border-radius: 0 6px 6px 0;}
.review.fake {border-left-color: #B83A2E;}
.review .meta {font-size: .85rem; opacity: .75; margin-bottom: .3rem;}
.review .why {font-size: .85rem; opacity: .8; margin-top: .4rem;}
.review mark {background: #F6C9C2; color: inherit; padding: 0 2px; border-radius: 2px;}
.verdict {font-weight: 600; padding: 1px 8px; border-radius: 10px; color: white; font-size: .85rem;}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading models…")
def get_detective(model_path, summarizer_backend, absa_backend):
    return FakeReviewDetective.from_model_file(model_path, summarizer_backend, absa_backend)


@st.cache_data(show_spinner="Checking reviews…")
def classify(_det, key, texts, ratings):
    return _det.classify(texts, ratings)


@st.cache_data(show_spinner="Summarising genuine reviews…")
def report_for(_det, key, threshold, _reviews):
    _det.threshold = threshold
    return _det.build_report(_reviews)


def stars(r):
    return "–" if r is None or pd.isna(r) else "★" * int(round(r)) + "☆" * (5 - int(round(r)))


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Reviews")
    upload = st.file_uploader("Upload a CSV of one product's reviews", type="csv")
    raw = upload.getvalue() if upload else SAMPLE.read_bytes()
    df = pd.read_csv(io.BytesIO(raw))
    text_col = st.selectbox("Review text column", df.columns,
                            index=list(df.columns).index(find_col(df, TEXT_COLS, required=False) or df.columns[0]))
    rating_guess = find_col(df, RATING_COLS, required=False)
    rating_opts = ["(none)"] + list(df.columns)
    rating_col = st.selectbox("Star rating column", rating_opts,
                              index=rating_opts.index(rating_guess) if rating_guess else 0)

    st.header("Detection")
    threshold = st.slider("Flag as fake when probability is at least", 0.05, 0.95, 0.5, 0.05)

    with st.expander("Models"):
        model_path = st.text_input("Classifier file", "models/classifier.joblib")
        summ_backend = st.selectbox("Summary", ["auto", "abstractive", "extractive"])
        absa_backend = st.selectbox("Aspect sentiment", ["auto", "transformer", "lexicon"])

if not Path(model_path).exists():
    st.error(f"No trained classifier at `{model_path}`. Train one first: "
             "`python train.py --data data/fake_reviews_dataset.csv` "
             "(or `python train.py --demo` to try the app on synthetic data).")
    st.stop()

texts = df[text_col].astype(str).tolist()
ratings = pd.to_numeric(df[rating_col], errors="coerce").tolist() if rating_col != "(none)" else None
det = get_detective(model_path, summ_backend, absa_backend)
key = (hash(raw), text_col, rating_col, model_path)
reviews = classify(det, key, texts, ratings)
report = report_for(det, key + (summ_backend, absa_backend), threshold, reviews)
s = report.stats

# ----------------------------------------------------------------------------- header
st.title("FakeReview Detective")
st.caption(upload.name if upload else "Sample product: a portable Bluetooth speaker")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews analysed", s["n_total"])
c2.metric("Flagged as fake", f"{s['n_fake']}", f"{s['pct_fake']:.0%} of all reviews", delta_color="off")
if s["raw_avg_rating"] is not None:
    c3.metric("Listed star rating", f"{s['raw_avg_rating']:.2f}")
    if s["genuine_avg_rating"] is not None:
        c4.metric("Rating from genuine reviews", f"{s['genuine_avg_rating']:.2f}",
                  f"{s['genuine_avg_rating'] - s['raw_avg_rating']:+.2f}")

tab_summary, tab_aspects, tab_reviews = st.tabs(["What buyers think", "Feature by feature", "Every review"])

# ----------------------------------------------------------------------------- summary
with tab_summary:
    if not report.genuine:
        st.info("Every review was flagged at this threshold, so there is nothing genuine to summarise. "
                "Raise the threshold in the sidebar.")
    else:
        st.subheader(f"Based on {s['n_total'] - s['n_fake']} genuine reviews")
        st.write(report.summary["summary"])
        st.caption(f"{s['genuine_positive']:.0%} of genuine reviews are positive, "
                   f"{s['genuine_negative']:.0%} negative. "
                   f"Summary method: {report.summary['backend']}.")
        if report.aspects:
            st.markdown(" ".join(
                f"<span class='verdict' style='background:{VERDICT_COLORS[a.verdict]}'>{a.aspect}: {a.verdict}</span>"
                for a in report.aspects), unsafe_allow_html=True)

# ----------------------------------------------------------------------------- aspects
with tab_aspects:
    adf = report.aspects_frame()
    if adf.empty:
        st.info("No product feature was mentioned by enough genuine reviewers to rate it.")
    else:
        chart = alt.Chart(adf).mark_bar().encode(
            x=alt.X("score:Q", scale=alt.Scale(domain=[-1, 1]), title="Average sentiment (−1 to +1)"),
            y=alt.Y("aspect:N", sort="-x", title=None),
            color=alt.Color("verdict:N", scale=alt.Scale(domain=list(VERDICT_COLORS),
                                                          range=list(VERDICT_COLORS.values())),
                            legend=alt.Legend(title="Verdict")),
            tooltip=["aspect", "verdict", "score", "mentions", "positive", "neutral", "negative"],
        ).properties(height=max(160, 36 * len(adf)))
        st.altair_chart(chart, width="stretch")
        for a in report.aspects:
            label = f"{a.aspect}: {a.verdict} · {a.mentions} reviews ({a.positive} positive, {a.negative} negative)"
            with st.expander(label + ("  (found in reviews)" if a.discovered else "")):
                for snippet, sc in a.examples:
                    st.markdown(f"{'🟢' if sc > 0.1 else '🔴' if sc < -0.1 else '⚪'} “{snippet}”")

# ----------------------------------------------------------------------------- reviews
with tab_reviews:
    left, right = st.columns([2, 1])
    show = left.radio("Show", ["All", "Flagged as fake", "Genuine"], horizontal=True)
    order = right.selectbox("Sort by", ["Most suspicious first", "Original order"])
    rows = report.reviews
    if show == "Flagged as fake":
        rows = report.fake
    elif show == "Genuine":
        rows = report.genuine
    if order == "Most suspicious first":
        rows = sorted(rows, key=lambda r: -r.fake_prob)

    for r in rows:
        why = ""
        if r.is_fake and r.reasons:
            why = "<div class='why'>Why: " + "; ".join(t for t, _ in r.reasons) + "</div>"
        body = highlight(r.text, [p for p, _ in r.phrases]) if r.is_fake else highlight(r.text, [])
        st.markdown(
            f"<div class='review {'fake' if r.is_fake else ''}'>"
            f"<div class='meta'>{'Fake' if r.is_fake else 'Genuine'} · {r.fake_prob:.0%} likely fake · "
            f"{stars(r.rating)}</div>{body}{why}</div>", unsafe_allow_html=True)

    st.download_button("Download results as CSV", report.reviews_frame().to_csv(index=False),
                       "fakereview_results.csv", "text/csv")
