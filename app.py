import streamlit as st
import pandas as pd
import pickle

with open("content_df.pkl", "rb") as f:
    content_df = pickle.load(f)

with open("ratings_df.pkl", "rb") as f:
    ratings_df = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("cosine_similarity_matrix.pkl", "rb") as f:
    cosine_sim = pickle.load(f)

with open("svd_model.pkl", "rb") as f:
    model = pickle.load(f)

content_df['title_clean'] = content_df['title'].str.strip().str.lower()
title_to_index = pd.Series(content_df.index, index=content_df['title_clean'])

def hybrid_recommender(title, user_id, weight_cb=0.5, weight_cf=0.5, top_n=10):
    title_clean = title.strip().lower()
    if title_clean not in title_to_index:
        return f"❌ Movie '{title}' not found in the dataset."

    idx = title_to_index[title_clean]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]

    candidate_ids = [content_df.iloc[i[0]]['id'] for i in sim_scores]
    cb_scores = {content_df.iloc[i[0]]['id']: i[1] for i in sim_scores}

    cf_scores = {}
    for movie_id in candidate_ids:
        try:
            pred = model.predict(user_id, movie_id)
            cf_scores[movie_id] = pred.est
        except:
            cf_scores[movie_id] = 0.0

    hybrid_scores = []
    for movie_id in candidate_ids:
        cb_score = cb_scores.get(movie_id, 0)
        cf_score = cf_scores.get(movie_id, 0)
        final_score = (weight_cb * cb_score) + (weight_cf * cf_score)
        hybrid_scores.append((movie_id, final_score))

    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:top_n]
    top_ids = [movie[0] for movie in hybrid_scores]

    results = content_df[content_df['id'].isin(top_ids)][['id', 'title']].copy()

    results['hybrid_score'] = results['id'].map(dict(hybrid_scores))
    return results.sort_values(by='hybrid_score', ascending=False).reset_index(drop=True)

st.title("🎬 Hybrid Movie Recommender System")

movie_input = st.text_input("Enter a movie title", "The Dark Knight")
user_id = st.number_input("Enter User ID", min_value=1, value=1)
weight_cb = st.slider("Content-Based Weight", 0.0, 1.0, 0.5)
weight_cf = 1.0 - weight_cb
top_n = st.slider("Number of Recommendations", 1, 20, 10)

if st.button("Get Recommendations"):
    recommendations = hybrid_recommender(
        title=movie_input,
        user_id=user_id,
        weight_cb=weight_cb,
        weight_cf=weight_cf,
        top_n=top_n
    )
    if isinstance(recommendations, str):
        st.error(recommendations)
    else:
        st.subheader("Recommended Movies:")
        st.dataframe(recommendations)