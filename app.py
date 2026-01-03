import streamlit as st
from src.ai_recommender import recommend

# Page configuration
st.set_page_config(
    page_title="AI Content Recommendation System",
    page_icon="🎬",
    layout="centered"
)

# Title
st.markdown("<h1 style='text-align: center;'>🎬 AI Content Recommendation System</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Describe what you want to watch and get AI-powered recommendations</p>",
    unsafe_allow_html=True
)

st.divider()

# Input box
query = st.text_input(
    "🧠 Enter your mood, theme, or idea",
    placeholder="e.g. space travel with emotional story"
)

# Button
if st.button("🔍 Recommend"):
    if query.strip() == "":
        st.warning("Please enter a valid description.")
    else:
        with st.spinner("AI is thinking... 🤖"):
            results = recommend(query)

        st.success("Recommended for you:")

        for _, row in results.iterrows():
            st.subheader(f"🎥 {row['title']}")
            st.write(f"**Description:** {row['description']}")
            st.write(f"📊 **Relevance Score:** {row['score']:.2f}")
            st.markdown(f"💡 *{row['explanation']}*")
            st.markdown("---")


# Footer
st.markdown(
    "<p style='text-align: center; color: gray;'>Powered by NLP embeddings & semantic search</p>",
    unsafe_allow_html=True
)
