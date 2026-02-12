from rag_app.embedder import get_embedding
from rag_app.qdrant_client import search_vector
import requests

# Use a lightweight open model (no keys needed)
OLLAMA_URL = "http://localhost:11434/api/generate"

def generate_answer(query: str, top_k=3):
    # 1) Retrieve context
    q_vec = get_embedding(query)
    results = search_vector(q_vec, top_k=top_k)

    contexts = []
    for r in results.get("result", []):
        if "payload" in r and "text" in r["payload"]:
            contexts.append(r["payload"]["text"])

    context_text = "\n".join(contexts)

    # 2) Prompt
    prompt = f"""
    Answer the question using only the given context.
    If you don't know, say "I don't know."

    CONTEXT:
    {context_text}

    QUESTION:
    {query}
    """

    # 3) Generate
    r = requests.post(
        OLLAMA_URL,
        json={"model": "llama3.2", "prompt": prompt, "stream": False},
        timeout=30
    )

    return r.json().get("response", ""), contexts
