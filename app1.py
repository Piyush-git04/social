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
def load_data(path: str = "sentimentdataset_10categories_no_other.csv") -> pd.DataFrame:
    """
    Load the sentiment dataset with 10 categories (no 'other').
    Expected columns:
      - Text
      - Sentiment
      - Hashtags
      - Retweets
      - Likes
      - Category
    """
    if not os.path.exists(path):
        st.error(
            f"❌ Dataset not found: {path}\n\n"
            "Make sure 'sentimentdataset_10categories_no_other.csv' is in the same folder as app.py."
        )
        st.stop()

    df = pd.read_csv(path)

    required_cols = {"Text", "Sentiment", "Hashtags", "Retweets", "Likes", "Category"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"Dataset is missing required columns: {missing}")
        st.stop()

    df["Text"] = df["Text"].astype(str).str.strip()
    df["Hashtags"] = df["Hashtags"].astype(str).str.strip()
    df["Retweets"] = df["Retweets"].fillna(0).astype(float)
    df["Likes"] = df["Likes"].fillna(0).astype(float)
    df["Category"] = df["Category"].astype(str).str.strip()

    df["engagement"] = df["Likes"] + df["Retweets"]

    return df


# ----------------------------------------------------------
# 2. MODEL TRAINING
# ----------------------------------------------------------

@st.cache_resource
def train_models(df: pd.DataFrame):
    """
    Train:
    - TF-IDF for similarity search on Text
    - Ridge Regression to predict engagement from Text + Hashtags
    """

    # Similarity vectorizer on Text
    similarity_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
    )
    caption_tfidf = similarity_vectorizer.fit_transform(df["Text"])

    # Combined text for engagement model
    combined = df["Text"] + " " + df["Hashtags"].str.replace("#", " ", regex=False)

    eng_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=7000,
    )
    X = eng_vectorizer.fit_transform(combined)
    y = df["engagement"].astype(float)

    eng_model = Ridge(alpha=1.0)
    eng_model.fit(X, y)

    return {
        "similarity_vectorizer": similarity_vectorizer,
        "caption_tfidf": caption_tfidf,
        "eng_vectorizer": eng_vectorizer,
        "eng_model": eng_model,
        "min_eng": float(y.min()),
        "max_eng": float(y.max()),
    }


# ----------------------------------------------------------
# 3. CORE FUNCTIONS
# ----------------------------------------------------------

def enhance_caption(text: str) -> str:
    booster = np.random.choice(BOOSTERS)
    return (text or "").strip() + " " + booster


def generate_captions(
    topic: str,
    df: pd.DataFrame,
    similarity_vectorizer: TfidfVectorizer,
    caption_tfidf,
    candidate_indices,
    n: int = 5,
):
    """
    Generate captions using Text similarity, restricted to candidate_indices.
    """
    if len(candidate_indices) == 0:
        return []

    # If topic is empty: return top engagement texts from the subset
    if not topic.strip():
        subset = df.iloc[candidate_indices].copy()
        subset = subset.sort_values("engagement", ascending=False)
        return subset["Text"].head(n).tolist()

    topic_vec = similarity_vectorizer.transform([topic])
    sub_matrix = caption_tfidf[candidate_indices]
    sims = cosine_similarity(topic_vec, sub_matrix).flatten()

    top_local = sims.argsort()[::-1][:n]
    top_global_indices = [candidate_indices[i] for i in top_local]

    return [df.iloc[i]["Text"] for i in top_global_indices]


def recommend_hashtags(
    caption: str,
    df: pd.DataFrame,
    similarity_vectorizer: TfidfVectorizer,
    caption_tfidf,
    top_k: int = 10,
):
    """
    Recommend hashtags based on Text similarity over the full dataset.
    """

    global_tags = (
        df["Hashtags"].str.split().explode().str.strip().value_counts()
    )

    if not caption.strip():
        return global_tags.head(top_k).index.tolist()

    query_vec = similarity_vectorizer.transform([caption])
    sims = cosine_similarity(query_vec, caption_tfidf).flatten()
    top_idx = sims.argsort()[::-1][:5]

    tags = []
    for i in top_idx:
        tags.extend(df.iloc[i]["Hashtags"].split())

    if not tags:
        return global_tags.head(top_k).index.tolist()

    return (
        pd.Series([t.strip() for t in tags if t.strip()])
        .value_counts()
        .head(top_k)
        .index
        .tolist()
    )


def score_caption(
    text: str,
    hashtags: str,
    eng_vectorizer: TfidfVectorizer,
    eng_model: Ridge,
    min_eng: float,
    max_eng: float,
):
    hashtags_clean = hashtags.replace("#", " ") if hashtags else ""
    combined = (text or "") + " " + hashtags_clean
    X_new = eng_vectorizer.transform([combined])
    pred = float(eng_model.predict(X_new)[0])

    if max_eng == min_eng:
        score = 50.0
    else:
        score = 100 * (pred - min_eng) / (max_eng - min_eng)
        score = max(0.0, min(100.0, score))

    return pred, score


def download_output(captions, hashtags):
    content = "Generated Captions:\n\n"
    for c in captions:
        content += f"- {c}\n"
    content += "\nRecommended Hashtags:\n"
    content += " ".join(hashtags)

    st.download_button(
        label="⬇ Download captions & hashtags (.txt)",
        data=content,
        file_name="social_media_suggestions.txt",
        mime="text/plain",
    )


# ----------------------------------------------------------
# 4. STREAMLIT APP
# ----------------------------------------------------------

def main():
    st.set_page_config(
        page_title="AI Social Media Assistant",
        page_icon="📱",
        layout="wide",
    )

    st.title("📱 AI Social Media Assistant (Emotion Dataset)")
    st.caption("Caption Generator • Hashtag Recommender • Caption Scorer • Dataset Explorer")

    if "generated_captions" not in st.session_state:
        st.session_state["generated_captions"] = []
    if "generated_hashtags" not in st.session_state:
        st.session_state["generated_hashtags"] = []

    df = load_data()
    models = train_models(df)

    similarity_vectorizer = models["similarity_vectorizer"]
    caption_tfidf = models["caption_tfidf"]
    eng_vectorizer = models["eng_vectorizer"]
    eng_model = models["eng_model"]
    min_eng = models["min_eng"]
    max_eng = models["max_eng"]

    # Sidebar: stats
    st.sidebar.header("📊 Dataset Summary")
    st.sidebar.metric("Total posts", len(df))
    st.sidebar.metric("Avg Likes", f"{df['Likes'].mean():.1f}")
    st.sidebar.metric("Avg Retweets", f"{df['Retweets'].mean():.1f}")
    st.sidebar.metric("Avg Engagement", f"{df['engagement'].mean():.1f}")

    st.sidebar.subheader("Top Categories")
    for cat, cnt in df["Category"].value_counts().head(10).items():
        st.sidebar.write(f"{cat} — {cnt}")

    st.sidebar.subheader("Top Hashtags")
    top_tags = (
        df["Hashtags"].str.split().explode().str.strip().value_counts().head(10)
    )
    for tag, cnt in top_tags.items():
        st.sidebar.write(f"{tag} — {cnt}")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "✨ Caption Generator",
            "🏷 Hashtag Recommender",
            "⭐ Caption Scorer",
            "📊 Dataset Explorer",
        ]
    )

    # ------------------------------------------------------
    # TAB 1: CAPTION GENERATOR
    # ------------------------------------------------------
    with tab1:
        st.subheader("✨ Caption Generator")

        topic = st.text_input(
            "Describe your post (e.g. 'feeling hopeful', 'lost in thoughts tonight')"
        )

        categories = sorted(df["Category"].unique().tolist())
        selected_category = st.selectbox(
            "Filter by Category (optional)",
            options=["Any"] + categories,
        )

        ai_boost = st.checkbox("⚡ AI Boost (make captions catchier)")
        num_suggestions = st.slider("Number of suggestions", 3, 10, 5)

        if st.button("Generate Captions"):
            if selected_category == "Any":
                candidate_indices = np.arange(len(df))
            else:
                candidate_indices = np.where(df["Category"] == selected_category)[0]

            caps = generate_captions(
                topic,
                df,
                similarity_vectorizer,
                caption_tfidf,
                candidate_indices,
                n=num_suggestions,
            )

            final_captions = []
            for c in caps:
                final_captions.append(enhance_caption(c) if ai_boost else c)

            if not final_captions:
                st.warning("No captions found. Try a different topic or category.")
            else:
                st.markdown("### Suggested Captions")
                for i, c in enumerate(final_captions, start=1):
                    st.markdown(f"**{i}.** {c}")

                st.session_state["generated_captions"] = final_captions

    # ------------------------------------------------------
    # TAB 2: HASHTAG RECOMMENDER
    # ------------------------------------------------------
    with tab2:
        st.subheader("🏷 Hashtag Recommender")

        caption_text = st.text_area(
            "Enter caption for hashtag suggestions",
            height=150,
        )
        top_k_tags = st.slider("Number of hashtags", 5, 20, 10)

        if st.button("Suggest Hashtags"):
            tags = recommend_hashtags(
                caption_text,
                df,
                similarity_vectorizer,
                caption_tfidf,
                top_k=top_k_tags,
            )

            if not tags:
                st.warning("No hashtags found. Try a more detailed caption.")
            else:
                st.markdown("### Recommended Hashtags")
                st.success(" ".join(tags))
                st.session_state["generated_hashtags"] = tags

            if st.session_state["generated_captions"]:
                download_output(st.session_state["generated_captions"], tags)

    # ------------------------------------------------------
    # TAB 3: CAPTION SCORER
    # ------------------------------------------------------
    with tab3:
        st.subheader("⭐ Caption Scorer")

        caption_inp = st.text_area("Caption:", height=150)
        hashtag_inp = st.text_input(
            "Hashtags (optional, e.g. '#joy #gratitude')",
            value="",
        )

        if st.button("Score Caption"):
            if not caption_inp.strip():
                st.warning("Please enter a caption first.")
            else:
                raw_pred, score = score_caption(
                    caption_inp,
                    hashtag_inp,
                    eng_vectorizer,
                    eng_model,
                    min_eng,
                    max_eng,
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Engagement (Likes + Retweets)", f"{raw_pred:.1f}")
                with col2:
                    st.metric("Caption Score", f"{score:.1f} / 100")

                if score >= 75:
                    st.success("🔥 Strong caption! Likely to perform very well.")
                elif score >= 50:
                    st.info("👍 Decent caption. You can enhance it by adding more emotion or clarity.")
                else:
                    st.warning(
                        "😐 Score is on the lower side. Try making the caption more specific, emotional, "
                        "and aligning hashtags better with the core feeling."
                    )

    # ------------------------------------------------------
    # TAB 4: DATASET EXPLORER
    # ------------------------------------------------------
    with tab4:
        st.subheader("📊 Dataset Explorer")

        st.markdown("Preview of your dataset:")
        st.dataframe(df.head(50))

        col_a, col_b = st.columns(2)
        with col_a:
            sentiment_filter = st.multiselect(
                "Filter by Sentiment",
                options=sorted(df["Sentiment"].unique()),
                default=[],
            )
        with col_b:
            category_filter = st.multiselect(
                "Filter by Category",
                options=sorted(df["Category"].unique()),
                default=[],
            )

        filtered_df = df.copy()
        if sentiment_filter:
            filtered_df = filtered_df[filtered_df["Sentiment"].isin(sentiment_filter)]
        if category_filter:
            filtered_df = filtered_df[filtered_df["Category"].isin(category_filter)]

        search = st.text_input("Search in Text:")
        if search:
            filtered_df = filtered_df[
                filtered_df["Text"].str.contains(search, case=False, na=False)
            ]

        st.write(f"Showing {len(filtered_df)} rows:")
        st.dataframe(filtered_df)

        st.markdown("### Category distribution")
        st.bar_chart(df["Category"].value_counts())


if __name__ == "__main__":
    main()
