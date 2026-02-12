# RAG with Qdrant (Local Retrieval-Augmented Generation)

## 🚀 Overview

This project implements a **local Retrieval-Augmented Generation (RAG) pipeline** using:

* **Sentence Transformers (MiniLM)** – for embeddings
* **Qdrant (Docker)** – as a vector database
* **Ollama (llama3.2)** – as a local LLM for generation

The system allows you to:

1. Store documents semantically in a vector database
2. Retrieve the most relevant documents using vector similarity search
3. Generate answers grounded in the retrieved context

This makes the model **more accurate, explainable, and fact-grounded** compared to vanilla LLM responses.

---

## 🏗️ Architecture

```
User Query
     ↓
Embedding Model (MiniLM)
     ↓
Qdrant Vector Search (Docker)
     ↓
Retrieve Relevant Context
     ↓
LLM (Ollama - llama3.2)
     ↓
Final Answer (Grounded in Retrieved Data)
```

---

## 🔧 Tech Stack

* Python
* Docker
* Qdrant (Vector DB)
* Sentence Transformers (`all-MiniLM-L6-v2`)
* Ollama (`llama3.2`)

---

## 📦 Setup

### 1️⃣ Start Qdrant (Vector Database)

```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2️⃣ Install dependencies

```bash
pip install -r rag_app/requirements.txt
```

### 3️⃣ Install & start Ollama

Download: https://ollama.com/

Then run:

```bash
ollama pull llama3.2
```

---

## ▶️ Usage

### Add documents & query the system

```python
from rag_app.rag_app import add_document
from rag_app.rag_pipeline import generate_answer

add_document("Endee is a high-performance vector database.")
add_document("Qdrant is a fast open-source vector store with Docker support.")

answer, context = generate_answer("Which database is high performance?")

print("Answer:", answer)
print("\nRetrieved Context:", context)
```

---

## 📂 Project Structure

```
rag_app/
│── embedder.py        # Creates embeddings
│── qdrant_client.py   # Handles vector DB operations
│── rag_app.py         # Document storage & retrieval
│── rag_pipeline.py    # Full RAG pipeline (retrieve + generate)
│── requirements.txt   # Dependencies
```

---

## 🎯 Features

* ✅ Semantic search (not keyword matching)
* ✅ Fully local (no external APIs required)
* ✅ Scalable vector storage via Qdrant
* ✅ LLM responses grounded in real stored data
* ✅ Works with Docker

---

## 🚀 Future Improvements

* Add FastAPI backend
* Build a Streamlit chat UI
* Support PDF ingestion
* Enable authentication
* Support multiple collections

---

## 👨‍💻 Author

Arjun YS
AI/ML Enthusiast | RAG | Vector Databases | Docker
