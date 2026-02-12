import requests
import numpy as np

QDRANT_URL = "http://localhost:6333"

COLLECTION = "endee-demo"

def health():
    r = requests.get(f"{QDRANT_URL}/health", timeout=5)
    return {"status": "ok", "text": r.text}

def create_collection(dim: int):
    r = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}",
        json={
            "vectors": {
                "size": dim,
                "distance": "Cosine"
            }
        },
    )
    return r.json()

def ensure_collection(vec_dim: int):
    r = requests.get(f"{QDRANT_URL}/collections/{COLLECTION}")
    if r.status_code != 200:
        create_collection(vec_dim)

def store_vector(vec, payload):
    vec = list(map(float, vec))

    # Make sure collection exists
    ensure_collection(len(vec))

    data = {
        "points": [
            {
                "id": np.random.randint(1_000_000),
                "vector": vec,
                "payload": payload
            }
        ]
    }

    r = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}/points",
        json=data
    )
    return r.json()

def search_vector(vec, top_k=1):
    vec = list(map(float, vec))

    r = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
        json={
            "vector": vec,
            "limit": top_k,
            "with_payload": True      # <-- THIS IS THE IMPORTANT LINE
        },
    )
    return r.json()

