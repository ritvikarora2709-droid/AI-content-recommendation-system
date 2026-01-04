import streamlit as st
from src.ai_recommender import recommend

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="AI Content Recommendation System",
    page_icon="🎬",
    layout="centered"
)

# =========================
# Title
# =========================
st.markdown(
    "<h1 style='text-align: center;'>🎬 AI Content Recommendation System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Describe what you want to watch and get AI-powered recommendations</p>",
    unsafe_allow_html=True
)

st.divider()

# =========================
# User Inputs
# =========================
query = st.text_input(
    "🧠 Enter your mood, theme, or idea",
    placeholder="e.g. epic fantasy adventure or romantic bollywood drama"
)

source_filter = st.selectbox(
    "🌍 Content Preference",
    ["All", "Hollywood", "Bollywood"]
)

top_k = st.slider(
    "🎯 Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)

# =========================
# Recommend Button
# =========================
if st.button("🔍 Recommend"):
    if query.strip() == "":
        st.warning("Please enter a valid description.")
    else:
        with st.spinner("AI is thinking... 🤖"):
            results = recommend(query, top_k=top_k)

        # =========================
        # Apply source filter
        # =========================
        if source_filter == "Hollywood":
            results = results[results["source"] == "TMDB"]
        elif source_filter == "Bollywood":
            results = results[results["source"] == "Bollywood_IMDB"]

        if results.empty:
            st.error("No results found for this combination. Try another query.")
        else:
            st.success("Recommended for you:")

        for _, row in results.iterrows():
            with st.container():
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid #333;
                        border-radius: 12px;
                        padding: 16px;
                        margin-bottom: 16px;
                        background-color: #0e1117;
                    ">
                        <h3>🎥 {row['title']}</h3>
                        <p><b>Description:</b> {row['description']}</p>
                        <p>🎭 <b>Genres:</b> {row['genres']}</p>
                        <p>🌍 <b>Source:</b> {row['source']}</p>
                        <p>📊 <b>Relevance Score:</b> {row['score']:.2f}</p>
                        <p style="color: #aaa;"><i>💡 {row['explanation']}</i></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# =========================
# Footer
# =========================
st.markdown(
    "<p style='text-align: center; color: gray;'>Powered by pretrained NLP embeddings & semantic search</p>",
    unsafe_allow_html=True
)
