
# 📚 Semantic Book Recommender System
A Streamlit-based intelligent book recommendation web app that suggests books based on the semantic meaning of a user-provided description, category, and emotional tone. It uses Google Generative AI Embeddings and Chroma vector store to retrieve the most relevant and emotionally aligned books.



## Features

1. 🔍 Semantic Search: Enter a book description to get semantically similar recommendations.
2. 🧠 Emotion-Aware Filtering: Choose from tones like Happy, Sad, Suspenseful, etc., to tailor the mood of the suggestions.
3. 🗂️ Category Filtering: Narrow down recommendations by book category.
4. 🖼️ Thumbnail Previews: View book covers with short summaries.
⚡ Vector Database: Fast and accurate search powered by LangChain's Chroma DB.
5. 🌍 Google Generative AI Embeddings: Captures deep semantic context of user queries.


## Tech Stack
1. Frontend: Streamlit
2. Backend/NLP: LangChain
3. Chroma Vector Store
4. Google Generative AI Embeddings
5. Data: CSV file (books_with_emotions.csv) with metadata and emotional tags for each book.
6. Environment: dotenv for managing API keys.


## Project Structure

```bash
📦 semantic-book-recommender/
├── books_with_emotions.csv
├── tagged_description.txt
├── chroma_langchain_db/
├── main.py
├── .env
└── README.md
```


## Setup Instructions

### Step 1: Fork and Clone

```bash
git clone https://github.com/Aditya26Das/Book_Recommender_System.git
cd Book_Recommender_System
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Create .env file and insert your own GEMINI API KEY

```bash
GOOGLE_API_KEY=your_google_api_key_here
```

### Step 5: Run the App

```bash
streamlit run app.py
```
## How it works ?

1. User Input: User enters a description, selects a category and tone.
2. Embedding Generation: The description is converted into a vector using Google’s embedding model.
3. Similarity Search: Retrieves semantically close documents (books) using ChromaDB.
4. Filtering:
- By category (if selected)
- By emotional tone based on joy, sadness, fear, anger, surprise scores
5. Display: Top book recommendations with thumbnails and summaries are shown.


## Datasets Used

1. books_with_emotions.csv: Contains metadata and emotional scores (e.g., joy, sadness).
2. tagged_description.txt: Used to initially populate the Chroma vector database.