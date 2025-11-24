import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------------------------------------
# 0. CONFIG
# ----------------------------------------------------------

BOOSTERS = [
    "✨",
    "🔥",
    "🌟",
    "Captured the moment perfectly.",
    "Living my best life.",
    "One frame, endless stories.",
    "Feeling grateful for this.",
    "Pure vibes only.",
    "Energy speaks louder than words.",
]


# ----------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------

@st.cache_data
def load_data(path="sentimentdataset_10categories_no_other.csv"):
    """
    Loads dataset with corrected column names:
    text, sentiment, hashtags, comments, likes, category
    """
    if not os.path.exists(path):
        st.error(f"❌ Dataset not found at: {path}\nMake sure the CSV is in the same folder as app.py.")
        st.stop()

    df = pd.read_csv(path)

    required_cols = {"text", "sentiment", "hashtags", "comments", "likes", "category"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"Dataset is missing required columns: {missing}")
        st.stop()

    # Clean / standardize
    df["text"] = df["text"].astype(str).str.strip()
    df["hashtags"] = df["hashtags"].astype(str).str.strip()
    df["comments"] = df["comments"].fillna(0).astype(float)
    df["likes"] = df["likes"].fillna(0).astype(float)
    df["category"] = df["category"].astype(str).str.strip()

    df["engagement"] = df["likes"] + df["comments"]

    return df


# ----------------------------------------------------------
# 2. MODEL TRAINING
# ----------------------------------------------------------

@st.cache_resource
def train_models(df):
    """
    Train models:
    - Similarity: TF-IDF(Text)
    - Engagement model: TF-IDF(Text + Hashtags)
    """

    similarity_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
    )
    caption_tfidf = similarity_vectorizer.fit_transform(df["text"])

    combined = df["text"] + " " + df["hashtags"].str.replace("#", " ", regex=False)

    eng_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=7000,
    )
    X = eng_vectorizer.fit_transform(combined)
    y = df["engagement"]

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    return {
        "similarity_vectorizer": similarity_vectorizer,
        "caption_tfidf": caption_tfidf,
        "eng_vectorizer": eng_vectorizer,
        "eng_model": model,
        "min_eng": float(y.min()),
        "max_eng": float(y.max()),
    }


# ----------------------------------------------------------
# 3. CORE FUNCTIONS
# ----------------------------------------------------------

def enhance_caption(text):
    return text + " " + np.random.choice(BOOSTERS)


def generate_captions(topic, df, vectorizer, tfidf_matrix, indices, n=5):
    if len(indices) == 0:
        return []

    if not topic.strip():
        subset = df.iloc[indices].sort_values("engagement", ascending=False)
        return subset["text"].head(n).tolist()

    topic_vec = vectorizer.transform([topic])
    sub_matrix = tfidf_matrix[indices]

    sims = cosine_similarity(topic_vec, sub_matrix).flatten()
    top_local = sims.argsort()[::-1][:n]

    return [df.iloc[indices[i]]["text"] for i in top_local]


def recommend_hashtags(caption, df, vectorizer, tfidf_matrix, top_k=10):
    global_tags = (
        df["hashtags"]
        .str.split()
        .explode()
        .str.strip()
        .value_counts()
    )

    if not caption.strip():
        return global_tags.head(top_k).index.tolist()

    query_vec = vectorizer.transform([caption])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_idx = sims.argsort()[::-1][:5]

    tags = []
    for i in top_idx:
        tags.extend(df.iloc[i]["hashtags"].split())

    if not tags:
        return global_tags.head(top_k).index.tolist()

    return (
        pd.Series([t.strip() for t in tags if t.strip()])
        .value_counts()
        .head(top_k)
        .index
        .tolist()
    )


def score_caption(text, hashtags, vectorizer, model, min_eng, max_eng):
    combined = text + " " + hashtags.replace("#", " ")
    X_new = vectorizer.transform([combined])
    pred = float(model.predict(X_new)[0])

    if max_eng == min_eng:
        return pred, 50.0

    score = 100 * (pred - min_eng) / (max_eng - min_eng)
    score = max(0.0, min(100.0, score))

    return pred, score


def download_output(caps, tags):
    content = "Generated Captions:\n"
    for c in caps:
        content += f"- {c}\n"

    content += "\nRecommended Hashtags:\n"
    content += " ".join(tags)

    st.download_button(
        "⬇ Download suggestions",
        content,
        "caption_suggestions.txt",
        "text/plain",
    )


# ----------------------------------------------------------
# 4. STREAMLIT APP
# ----------------------------------------------------------

def main():
    st.set_page_config(page_title="AI Social Media Generator", page_icon="📱", layout="wide")

    st.title("📱 AI Social Media Generator (Emotion Dataset)")
    st.caption("Caption Generator • Hashtags • Score • Dataset Explorer")

    df = load_data()
    models = train_models(df)

    sim_vec = models["similarity_vectorizer"]
    tfidf = models["caption_tfidf"]
    eng_vec = models["eng_vectorizer"]
    eng_model = models["eng_model"]
    min_eng = models["min_eng"]
    max_eng = models["max_eng"]

    tab1, tab2, tab3, tab4 = st.tabs([
        "✨ Caption Generator",
        "🏷 Hashtags",
        "⭐ Scoring",
        "📊 Dataset",
    ])

    # ------------------------------------------------------
    # TAB 1: CAPTION GENERATOR
    # ------------------------------------------------------
    with tab1:
        st.subheader("✨ Caption Generator")

        topic = st.text_input("What's your post about?")
        categories = sorted(df["category"].unique())
        selected_cat = st.selectbox("Filter by category", ["Any"] + categories)
        ai_boost = st.checkbox("⚡ AI Boost")
        n_suggestions = st.slider("Number of captions", 3, 10, 5)

        if st.button("Generate"):
            if selected_cat == "Any":
                indices = np.arange(len(df))
            else:
                indices = np.where(df["category"] == selected_cat)[0]

            caps = generate_captions(topic, df, sim_vec, tfidf, indices, n_suggestions)

            final_caps = [enhance_caption(c) if ai_boost else c for c in caps]

            if final_caps:
                st.markdown("### Suggested Captions")
                for i, c in enumerate(final_caps, 1):
                    st.markdown(f"**{i}.** {c}")
                st.session_state["generated_captions"] = final_caps
            else:
                st.warning("No captions found.")

    # ------------------------------------------------------
    # TAB 2: HASHTAGS
    # ------------------------------------------------------
    with tab2:
        st.subheader("🏷 Hashtag Recommender")

        caption_text = st.text_area("Enter caption")
        top_k = st.slider("Number of hashtags", 5, 20, 10)

        if st.button("Suggest Hashtags"):
            tags = recommend_hashtags(caption_text, df, sim_vec, tfidf, top_k)

            st.markdown("### Recommended Hashtags")
            st.success(" ".join(tags))

            st.session_state["generated_hashtags"] = tags

            if "generated_captions" in st.session_state:
                download_output(st.session_state["generated_captions"], tags)

    # ------------------------------------------------------
    # TAB 3: CAPTION SCORER
    # ------------------------------------------------------
    with tab3:
        st.subheader("⭐ Caption Scorer")

        cap = st.text_area("Caption")
        ht = st.text_input("Hashtags")

        if st.button("Score"):
            pred, score = score_caption(cap, ht, eng_vec, eng_model, min_eng, max_eng)

            col1, col2 = st.columns(2)
            col1.metric("Predicted Engagement", f"{pred:.1f}")
            col2.metric("Score", f"{score:.1f}/100")

    # ------------------------------------------------------
    # TAB 4: DATASET EXPLORER
    # ------------------------------------------------------
    with tab4:
        st.subheader("📊 Dataset Explorer")

        st.dataframe(df.head(50))

        sent_filter = st.multiselect("Sentiment", sorted(df["sentiment"].unique()))
        cat_filter = st.multiselect("Category", sorted(df["category"].unique()))
        search = st.text_input("Search text")

        filtered = df.copy()
        if sent_filter:
            filtered = filtered[filtered["sentiment"].isin(sent_filter)]
        if cat_filter:
            filtered = filtered[filtered["category"].isin(cat_filter)]
        if search:
            filtered = filtered[filtered["text"].str.contains(search, case=False)]

        st.write(f"{len(filtered)} rows found:")
        st.dataframe(filtered)

        st.markdown("### Category Distribution")
        st.bar_chart(df["category"].value_counts())


if __name__ == "__main__":
    main()
