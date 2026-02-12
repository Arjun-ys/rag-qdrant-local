from rag_app.embedder import get_embedding
from rag_app.qdrant_client import store_vector, search_vector
import requests

# Optional — replace with your own HF token if you want better responses
LLM_API = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HF_HEADERS = {}  # e.g. {"Authorization": "Bearer <YOUR_TOKEN>"}

def add_document(text: str):
    """Embed and store a document in the vector DB."""
    vec = get_embedding(text)
    return store_vector(vec, {"text": text})

def retrieve_context(query: str, k=2):
    """Retrieve most similar stored documents."""
    vec = get_embedding(query)
    results = search_vector(vec, top_k=k)
    return [r["payload"]["text"] for r in results.get("result", [])]

def generate_answer(query: str):
    """Basic RAG: retrieve + ask an LLM."""
    docs = retrieve_context(query)

    prompt = f"""
You are a helpful assistant. Answer using ONLY the context below.

CONTEXT:
{chr(10).join(docs)}

QUESTION: {query}
ANSWER:
"""

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 200}
    }

    r = requests.post(LLM_API, headers=HF_HEADERS, json=payload, timeout=30)
    return r.json()

if __name__ == "__main__":
    print("RAG Demo ready. Use add_document() and retrieve_context().")
