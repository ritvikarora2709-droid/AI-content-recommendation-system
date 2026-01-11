# 🎬 AI-Powered Content Recommendation System

An AI-driven **movie recommendation system** that understands **natural language user intent** and provides **semantic recommendations** across **Hollywood (TMDB)** and **Bollywood (IMDB)** movies using pretrained NLP embeddings.

---

## 📌 Problem Statement

Traditional content recommendation systems rely heavily on user history, ratings, or predefined genres. These approaches struggle in scenarios such as:

* **Cold start** (new users with no history)
* Users expressing **vague or abstract preferences**
* **Intent-based discovery**, e.g., *"epic fantasy adventure with emotional depth"*

This project addresses these limitations by building a **semantic, intent-driven recommendation system** that works **without prior user data**.

---

## 🎯 Project Objectives

* Enable **free-text, intent-based** movie recommendations
* Use **pretrained AI models** instead of training from scratch
* Support **both Hollywood and Bollywood** movies
* Provide **real-time recommendations** via a web interface
* Focus on **practical AI application**, not model re-invention

---

## ✨ Key Features

* 🧠 Natural language movie search
* 🔍 Semantic similarity using NLP embeddings
* 🌍 Dual dataset support (TMDB + Bollywood IMDB)
* 📊 Relevance scoring using cosine similarity
* 💡 Human-readable recommendation explanations
* 🖥️ Interactive **Streamlit** web application

---

## 🛠️ Tech Stack

### Programming Language

* Python

### AI / ML / NLP

* Sentence-Transformers (`all-MiniLM-L6-v2`)
* NLP embeddings for semantic understanding
* Cosine similarity for vector matching

### Libraries

* Pandas
* NumPy
* Scikit-learn
* Streamlit

### Datasets

* **TMDB Movies Dataset** (Hollywood)
* **IMDB Bollywood Movies Dataset**

---

## 🧩 System Architecture

The recommendation pipeline follows these steps:

1. **Dataset Integration**
   Multiple public datasets (TMDB & Bollywood IMDB) are cleaned, standardized, and merged into a unified global dataset.

2. **Text Feature Construction**
   Movie title, description, genres, and metadata are combined into a single textual representation.

3. **Embedding Generation**
   A pretrained transformer model converts movie text into dense numerical embeddings.

4. **Query Understanding**
   User input is embedded using the same pretrained model.

5. **Similarity Computation**
   Cosine similarity is used to rank movies based on semantic closeness to the user query.

6. **Recommendation & Explanation**
   Top-ranked movies are returned with relevance scores and human-readable explanations.

---

## 📁 Repository Structure

```
Content_Recommendation_System/
│
├── data/
│   └── content_global.csv        # Unified TMDB + Bollywood dataset
│
├── src/
│   ├── ai_recommender.py          # Core AI recommendation logic
│   └── build_global_content.py    # Dataset preprocessing & merging
│
├── app.py                         # Streamlit web application
├── notebook.ipynb                 # Evaluation & experimentation notebook
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📊 Sample Output

**User Query:**

> "Hobbits, dwarves, elves, epic fantasy adventure"

**Top Recommendations:**

* *The Hobbit: An Unexpected Journey*
* *The Lord of the Rings: The Fellowship of the Ring*
* *Onward*

Each recommendation includes:

* Movie description
* Dataset source (Hollywood / Bollywood)
* Relevance score
* Explanation of why it was recommended

---

## 📈 Evaluation Metrics

* **Cosine Similarity Score**
  Measures semantic relevance between the user query and movie content.

* **Qualitative Evaluation**
  Manual inspection of recommendation relevance, diversity, and intent alignment.

---

## ⚠️ Limitations

* No user personalization or collaborative filtering
* Embeddings are generated at runtime (initial loading delay)
* Explanations are rule-based, not LLM-generated
* Dataset limited to publicly available metadata

---

## 🔮 Future Enhancements

* User profiling and personalization
* Hybrid recommendation (content-based + collaborative)
* LLM-powered natural language explanations
* Online vector databases (FAISS / Pinecone)
* Streaming platform integration
* Multilingual recommendations

---

## 🤖 Ethical Considerations & Responsible AI

* Uses only publicly available datasets
* No personal user data is collected or stored
* Avoids demographic or behavioral profiling
* Emphasizes transparency and explainability

---

## 📌 Conclusion

This project demonstrates how **pretrained AI models** can be effectively used to build **real-world recommendation systems** without training complex models from scratch. By focusing on **semantic understanding**, the system enables flexible, intuitive, and scalable content discovery across diverse movie datasets.

---

## 👤 Author

**Ritvik Arora**
AI Applications – Individual Open Project Module
