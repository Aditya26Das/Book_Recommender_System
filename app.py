import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv

import asyncio

# Ensure Streamlit thread has an event loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover_not_found.png",
    books["large_thumbnail"],
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db_books = Chroma(
    collection_name="book_recommendation_system",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

if len(db_books.get()["documents"]) == 0:
    with open("tagged_description.txt", "r", encoding="utf-8") as f:
        data = f.readlines()
    documents = [Document(page_content=line.strip()) for line in data if line.strip()]
    db_books.add_documents(documents)


def retrieve_semantic_recommendations(
        query: str,
        category: str = None, # type: ignore
        tone: str = None, # type: ignore
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

st.set_page_config(page_title="📚 Semantic Book Recommender", layout="wide")
st.title("📚 Semantic Book Recommender")

with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        query = st.text_input("🔎 Book Description", placeholder="e.g., A story about forgiveness")
    with col2:
        category = st.selectbox("📂 Category", categories)
    with col3:
        tone = st.selectbox("🎭 Emotional Tone", tones)

if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a valid book description to search.")
    else:
        results = recommend_books(query, category, tone)
        st.markdown("## 🔮 Recommendations")

        num_columns = 3
        for i in range(0, len(results), num_columns):
            cols = st.columns(num_columns)
            for j, (image_url, caption) in enumerate(results[i:i+num_columns]):
                with cols[j]:
                    st.image(image_url, use_container_width=True)
                    st.markdown(caption)

